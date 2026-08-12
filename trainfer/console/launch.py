"""Launcher: trainfer daemon on Qwen3.5-9B for human QA trial."""

from __future__ import annotations
import os
import sys
import logging
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unsloth  # noqa: F401 — must come before transformers
from trainfer.config import ServeConfig
from trainfer.server import serve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)

cfg = ServeConfig(
    model="unsloth/Qwen3.5-9B",
    max_seq_length=2048,
    host=os.environ.get("TRAINFER_HOST", "127.0.0.1"),
    port=int(os.environ.get("TRAINFER_PORT", "8768")),
    idle_replay=True,
    frozen_ref=False,
    # Dev defaults: hot reload + crash-safe state. Override via env.
    dev_autoreload=os.environ.get("TRAINFER_DEV_AUTORELOAD", "1") == "1",
    autosave_on_exit=os.environ.get("TRAINFER_AUTOSAVE_ON_EXIT", "1") == "1",
    autoload_on_boot=os.environ.get("TRAINFER_AUTOLOAD_ON_BOOT", "1") == "1",
)
print(f"[launch] starting trainfer on http://{cfg.host}:{cfg.port} with {cfg.model}")
serve(cfg)
