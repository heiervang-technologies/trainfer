"""Atomically published model snapshots and optional training resume.

A manifest is the commit record. Payloads live in a fresh generation before
replacing it, so overwrites cannot mix old and new tensors. Legacy manifests
remain readable. Trajectories are never truncated.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import uuid
import time
import tempfile

import torch
from safetensors.torch import load_file, save_file

from .state import ModelState
from .trajectory import TrajectoryLog
from .training_state import export_training, restore_training, validate_training
from . import quantized

log = logging.getLogger(__name__)


def _digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SnapshotManager:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, name: str) -> Path:
        if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
            raise ValueError("snapshot name must be a single directory name")
        path = self.root / name
        if path.is_symlink():
            raise ValueError("snapshot directory cannot be a symlink")
        return path

    def save(self, name: str, state: ModelState, log_: TrajectoryLog | None = None,
             *, engine=None, include_training: bool = False) -> Path:
        if include_training and (engine is None or engine.state is not state):
            raise ValueError("training snapshots require the model's training engine")
        with state.mode_lock:
            return self._save(name, state, log_, engine, include_training)

    def _save(self, name, state, log_, engine, include_training):
        d = self._dir(name)
        d.mkdir(parents=True, exist_ok=True)
        generation = d / ("generation-" + uuid.uuid4().hex)
        generation.mkdir()
        files = {}
        manifest = {
            "format_version": 2, "name": name, "created_at": time.time(),
            "base_model": state.base_model_name,
            "base_revision": getattr(state, "base_revision", "original"),
            "lora_rank": state.lora_rank, "lora_alpha": state.lora_alpha,
            "merges_applied": state.merges_applied,
            "residual_fingerprint": state.residual_fingerprint(),
            "trajectory_offset": log_.size() if log_ else 0,
            "trajectory_path": str(log_.path) if log_ else None,
            "files": files, "resumable": include_training,
        }
        temporary_manifest = d / (".manifest-" + uuid.uuid4().hex)
        published = False
        try:
            for role, tensors in (
                ("residual", state.merged_deltas),
                ("adapter", state.extract_active_adapter()),
            ):
                if tensors:
                    path = generation / (role + ".safetensors")
                    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(path))
                    _sync(path)
                    files[role] = {"path": str(path.relative_to(d)), "sha256": _digest(path)}
            # Per-layer files bound CPU workspace to one packed projection.
            # Include originals too: a pre-consolidation snapshot must remain
            # restorable after the live base changes.
            for index, (layer_name, base) in enumerate(quantized.projection_layers(state).items()):
                path = generation / f"base-{index}.safetensors"
                save_file(quantized.pack(base), str(path))
                _sync(path)
                files["base:" + layer_name] = {"path": str(path.relative_to(d)), "sha256": _digest(path)}
            if include_training:
                path = generation / "training.pt"
                torch.save(export_training(engine), path)
                _sync(path)
                files["training"] = {"path": str(path.relative_to(d)), "sha256": _digest(path)}
            _sync(generation)
            temporary_manifest.write_text(json.dumps(manifest, indent=2))
            _sync(temporary_manifest)
            os.replace(temporary_manifest, d / "manifest.json")
            published = True
            _sync(d)
            _sync(self.root)
        finally:
            temporary_manifest.unlink(missing_ok=True)
            if not published:
                shutil.rmtree(generation)
        # Controller serializes saves/loads. Cleanup failure cannot invalidate
        # the newly published generation; retry on the next save.
        for old in d.glob("generation-*"):
            if old != generation and old.is_dir() and not old.is_symlink():
                try:
                    shutil.rmtree(old)
                except OSError:
                    log.warning("could not remove old snapshot generation %s", old)
        log.info("snapshot saved to %s (resumable=%s)", d, include_training)
        return d

    def _payload(self, d, manifest, role):
        if manifest.get("format_version", 1) == 1:
            filename = {"residual": "merged_deltas.safetensors", "adapter": "active_adapter.safetensors"}.get(role)
            path = d / filename if filename else None
            return path if path and path.exists() else None
        record = manifest["files"].get(role)
        if record is None:
            return None
        path = (d / record["path"]).resolve()
        if not path.is_relative_to(d.resolve()):
            raise ValueError("snapshot payload escapes its directory")
        if _digest(path) != record["sha256"]:
            raise ValueError(f"snapshot {role} checksum mismatch")
        return path

    def load(self, name: str, state: ModelState, *, engine=None, resume: bool = False) -> dict:
        if resume and (engine is None or engine.state is not state):
            raise ValueError("resume requires the model's training engine")
        with state.mode_lock:
            return self._load(name, state, engine, resume)

    def _load(self, name, state, engine, resume):
        d = self._dir(name)
        manifest = json.loads((d / "manifest.json").read_text())
        if manifest.get("format_version", 1) not in (1, 2):
            raise ValueError("unsupported snapshot format")
        expected = {
            "base_model": state.base_model_name,
            "lora_rank": state.lora_rank, "lora_alpha": state.lora_alpha,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"snapshot {key} does not match the live model")
        base_layers = quantized.projection_layers(state)
        base_roles = {role[5:]: role for role in manifest.get("files", {}) if role.startswith("base:")}
        base_paths = {}
        if base_roles:
            if set(base_roles) != set(base_layers):
                raise ValueError("snapshot quantized projection names differ from live model")
            for layer_name, role in base_roles.items():
                path = self._payload(d, manifest, role)
                quantized.validate(base_layers[layer_name], load_file(str(path)))
                base_paths[layer_name] = path
        elif manifest.get("base_revision", "original") != state.base_revision:
            raise ValueError("snapshot base revision differs and no packed base is available")
        residual_path = self._payload(d, manifest, "residual")
        adapter_path = self._payload(d, manifest, "adapter")
        residual = load_file(str(residual_path)) if residual_path else {}
        adapter = load_file(str(adapter_path)) if adapter_path else {}
        named = dict(state.model.named_parameters())
        expected_names = {n for n in named if "lora_A" in n or "lora_B" in n}
        if manifest.get("format_version") == 2 and expected_names and not adapter:
            raise ValueError("snapshot is missing its adapter payload")
        if adapter and set(adapter) != expected_names:
            raise ValueError("snapshot adapter parameter names differ")
        for key, tensor in adapter.items():
            if tensor.shape != named[key].shape:
                raise ValueError(f"snapshot adapter shape differs for {key}")
            if manifest.get("format_version") == 2 and tensor.dtype != named[key].dtype:
                raise ValueError(f"snapshot adapter dtype differs for {key}")
        residual_shapes = {}
        for module_name, module in state.model.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                a = module.lora_A["default"].weight
                b = module.lora_B["default"].weight
                residual_shapes[module_name + ".weight"] = (b.shape[0], a.shape[1])
        for key, tensor in residual.items():
            if key not in residual_shapes or tuple(tensor.shape) != residual_shapes[key]:
                raise ValueError(f"snapshot residual shape/name differs for {key}")
        # Read/validate every payload before mutating the live model.
        training = None
        if resume:
            path = self._payload(d, manifest, "training")
            if path is None:
                raise ValueError("snapshot contains weights only; cannot resume training")
            training = torch.load(path, map_location="cpu", weights_only=True)
            validate_training(engine, training)
        previous_adapter = state.extract_active_adapter()
        previous_residual = state.merged_deltas
        previous_merges = state.merges_applied
        previous_revision = state.base_revision
        previous_training = export_training(engine) if engine is not None else None
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=self.root) as temporary:
            backups = {}
            for index, layer_name in enumerate(base_paths):
                backup = Path(temporary) / f"{index}.safetensors"
                save_file(quantized.pack(base_layers[layer_name]), str(backup))
                backups[layer_name] = backup
            try:
                for layer_name, path in base_paths.items():
                    quantized.apply(base_layers[layer_name], load_file(str(path)))
                self._apply(state, residual, adapter, manifest.get("merges_applied", 0))
                state.base_revision = manifest.get("base_revision", "original")
                if engine is not None:
                    if resume:
                        restore_training(engine, training)
                    else:
                        engine.reset_optimizer()
            except Exception:
                for layer_name, backup in backups.items():
                    quantized.apply(base_layers[layer_name], load_file(str(backup)))
                self._apply(state, previous_residual, previous_adapter, previous_merges)
                state.base_revision = previous_revision
                if engine is not None:
                    restore_training(engine, previous_training)
                raise
        log.info("snapshot %s loaded (resume=%s)", name, resume)
        return manifest

    @staticmethod
    def _apply(state, residual, adapter, merges):
        state._clear_residual_from_model()
        state.merged_deltas = residual
        if residual:
            state._apply_residual_to_model()
        if adapter:
            state.load_active_adapter(adapter)
        else:
            state.reset_active_adapter()
        state.merges_applied = merges

    def list(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and (p / "manifest.json").is_file())

    def describe(self, name: str) -> dict:
        return json.loads((self._dir(name) / "manifest.json").read_text())
