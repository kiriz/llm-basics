"""Config loader tests — YAML + CLI overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_trace import config


def test_defaults_applied_with_no_file():
    cfg = config.load_config(None)
    assert cfg["generation"]["max_new_tokens"] == 50
    assert cfg["collection"]["attention"]["layers"] == [0]
    assert cfg["cache"]["enabled"] is True


def test_yaml_loaded_and_merged_with_defaults(tmp_path: Path):
    p = tmp_path / "trace.yaml"
    p.write_text(
        "generation:\n"
        "  max_new_tokens: 20\n"
        "models:\n"
        "  - id: gpt2\n"
        "    alias: gpt2\n"
    )
    cfg = config.load_config(p)
    assert cfg["generation"]["max_new_tokens"] == 20
    # Non-overridden defaults still present
    assert cfg["generation"]["temperature"] == 0.0
    assert cfg["models"] == [{"id": "gpt2", "alias": "gpt2"}]


def test_cli_override_primitive():
    cfg = config.load_config(None, overrides=["generation.max_new_tokens=25"])
    assert cfg["generation"]["max_new_tokens"] == 25


def test_cli_override_list_via_json():
    cfg = config.load_config(None, overrides=["renderers=[\"html\",\"png\"]"])
    assert cfg["renderers"] == ["html", "png"]


def test_cli_override_bool_via_json():
    cfg = config.load_config(None, overrides=["cache.enabled=false"])
    assert cfg["cache"]["enabled"] is False


def test_cli_override_creates_nested_path():
    cfg = config.load_config(None, overrides=["new.nested.key=42"])
    assert cfg["new"]["nested"]["key"] == 42


def test_override_missing_equals_raises():
    with pytest.raises(ValueError, match="key=value"):
        config.load_config(None, overrides=["just-a-key"])


def test_collection_config_builder_picks_correct_fields():
    cfg = config.load_config(None, overrides=[
        "collection.attention.heads=[0,1,2]",
        "collection.top_k=25",
    ])
    col_cfg = config.collection_config_from(cfg)
    assert col_cfg.attention_heads == (0, 1, 2)
    assert col_cfg.top_k == 25


def test_gen_params_builder_coerces_types():
    cfg = config.load_config(None)
    gp = config.gen_params_from(cfg)
    assert isinstance(gp["max_new_tokens"], int)
    assert isinstance(gp["temperature"], float)
    assert isinstance(gp["stop_on_eos"], bool)


def test_config_module_does_not_import_torch() -> None:
    """Guardrail: config is part of the `render` path, must stay torch-free."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys
        from llm_trace import config  # noqa: F401
        banned = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))
        assert not banned, f'banned: {banned[:5]}'
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
