"""CLI arg parsing for `python -m trainfer.server ...`.

Pins the contract that Studio's /capsule/start subprocess.Popen relies on —
specifically that ``--port`` and ``--model`` override the ServeConfig defaults.
Before this fix, flags were silently ignored: the daemon always listened on
the ServeConfig default port, so Studio's proxy (configured via ``TRAINFER_PORT``)
could never reach it.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_only


def test_defaults_match_serve_config():
    from trainfer.config import ServeConfig
    from trainfer.cli import parse_cli_args as _parse_cli_args

    default = ServeConfig()
    cfg = _parse_cli_args([])
    assert cfg.host == default.host
    assert cfg.port == default.port
    assert cfg.model == default.model
    assert cfg.load_in_4bit is default.load_in_4bit


def test_port_flag_overrides_default():
    from trainfer.cli import parse_cli_args as _parse_cli_args

    cfg = _parse_cli_args(["--port", "8766"])
    assert cfg.port == 8766


def test_host_and_model_flags_override_defaults():
    from trainfer.cli import parse_cli_args as _parse_cli_args

    cfg = _parse_cli_args(["--host", "0.0.0.0", "--model", "foo/bar"])
    assert cfg.host == "0.0.0.0"
    assert cfg.model == "foo/bar"


def test_data_dir_becomes_path(tmp_path):
    from trainfer.cli import parse_cli_args as _parse_cli_args

    cfg = _parse_cli_args(["--data-dir", str(tmp_path)])
    assert cfg.data_dir == tmp_path


def test_toggle_flags():
    from trainfer.cli import parse_cli_args as _parse_cli_args

    cfg = _parse_cli_args(["--no-4bit", "--idle-replay", "--frozen-ref"])
    assert cfg.load_in_4bit is False
    assert cfg.idle_replay is True
    assert cfg.frozen_ref is True


def test_memory_and_adapter_controls():
    from trainfer.cli import parse_cli_args

    cfg = parse_cli_args(["--lora-rank", "4", "--lora-alpha", "8", "--lora-dropout", "0.1",
                          "--lora-target-modules", "q_proj", "v_proj",
                          "--gradient-checkpointing", "standard", "--model-dtype", "bfloat16",
                          "--memory-reserve-mib", "128", "--training-workspace-mib", "512"])
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout) == (4, 8, 0.1)
    assert cfg.lora_target_modules == ["q_proj", "v_proj"]
    assert cfg.gradient_checkpointing == "standard"
    assert cfg.model_dtype == "bfloat16"
    assert cfg.memory_reserve_bytes == 128 * 1024**2
    assert cfg.training_workspace_bytes == 512 * 1024**2


@pytest.mark.parametrize("args", [["--lora-rank", "0"], ["--lora-alpha", "0"],
                                  ["--lora-dropout", "1"], ["--memory-reserve-mib", "-1"],
                                  ["--training-workspace-mib", "-1"]])
def test_invalid_memory_and_adapter_controls_reject(args):
    from trainfer.cli import parse_cli_args

    with pytest.raises(SystemExit):
        parse_cli_args(args)
