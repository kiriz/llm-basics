"""Cache round-trip + version-mismatch + list-entries + clear."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm_trace import cache


def _make_arrays() -> dict[str, np.ndarray]:
    return {
        "attention_L0H0": np.random.RandomState(0).rand(5, 5).astype(np.float32),
        "hidden_last_per_layer": np.random.RandomState(1).rand(13, 768).astype(np.float32),
        "logits_top": np.array([-52.41, -54.21, -54.84], dtype=np.float32),
        "token_ids": np.array([16108, 318, 2835, 290, 220], dtype=np.int64),
    }


def _make_meta() -> dict:
    return {
        "model_id": "gpt2",
        "prompt": "Apple is round and",
        "templated_prompt": "Apple is round and",
        "gen_params": {"max_new_tokens": 8, "temperature": 0.0, "seed": 42},
        "tokens": ["Apple", " is", " round", " and", " "],
        "timings": {"forward_ms": 978.1, "per_token_ms": [31, 30, 30, 30]},
    }


def test_roundtrip_preserves_arrays_and_meta(tmp_path: Path) -> None:
    arrays = _make_arrays()
    meta = _make_meta()
    key = cache.cache_key(meta["model_id"], meta["templated_prompt"], meta["gen_params"])

    cache.save(key, arrays, meta, tmp_path)
    loaded = cache.load(key, tmp_path)

    assert loaded is not None, "load returned None for a just-saved key"
    loaded_arrays, loaded_meta = loaded

    assert set(loaded_arrays) == set(arrays)
    for name, arr in arrays.items():
        assert np.array_equal(loaded_arrays[name], arr), f"array {name} differs"
        assert loaded_arrays[name].dtype == arr.dtype, f"dtype {name} differs"

    assert loaded_meta == meta


def test_cache_key_stable_across_calls() -> None:
    k1 = cache.cache_key("gpt2", "Hello", {"max_new_tokens": 5, "temperature": 0.0})
    k2 = cache.cache_key("gpt2", "Hello", {"temperature": 0.0, "max_new_tokens": 5})
    assert k1 == k2, "key must not depend on gen_params dict ordering"


def test_cache_key_changes_when_inputs_change() -> None:
    base = cache.cache_key("gpt2", "Hello", {"max_new_tokens": 5})
    assert base != cache.cache_key("gpt2", "Hello ", {"max_new_tokens": 5})
    assert base != cache.cache_key("gpt2", "Hello", {"max_new_tokens": 6})
    assert base != cache.cache_key("tinyllama", "Hello", {"max_new_tokens": 5})


def test_load_returns_none_for_missing_key(tmp_path: Path) -> None:
    assert cache.load("nonexistent_abc123", tmp_path) is None


def test_load_returns_none_on_version_mismatch(tmp_path: Path, capsys) -> None:
    arrays = _make_arrays()
    meta = _make_meta()
    key = "stale_version_test"
    cache.save(key, arrays, meta, tmp_path)

    # Corrupt the version on disk
    js = tmp_path / f"{key}.json"
    stored = json.loads(js.read_text())
    stored["_cache_version"] = "v0"
    js.write_text(json.dumps(stored))

    assert cache.load(key, tmp_path) is None
    captured = capsys.readouterr()
    assert "version mismatch" in captured.out


def test_save_is_atomic_no_tmp_files_left(tmp_path: Path) -> None:
    key = "atomic_test"
    cache.save(key, _make_arrays(), _make_meta(), tmp_path)
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"expected no .tmp files, found {leftover}"


def test_list_entries_returns_saved_cells(tmp_path: Path) -> None:
    for i in range(3):
        meta = _make_meta()
        meta["gen_params"]["max_new_tokens"] = i + 1
        k = cache.cache_key(meta["model_id"], meta["templated_prompt"], meta["gen_params"])
        cache.save(k, _make_arrays(), meta, tmp_path)

    entries = cache.list_entries(tmp_path)
    assert len(entries) == 3
    for entry in entries:
        assert entry["model_id"] == "gpt2"
        assert entry["size_bytes"] > 0
        assert entry["cache_version"] == cache.CACHE_VERSION


def test_clear_removes_all_entries(tmp_path: Path) -> None:
    meta = _make_meta()
    k = cache.cache_key(meta["model_id"], meta["templated_prompt"], meta["gen_params"])
    cache.save(k, _make_arrays(), meta, tmp_path)
    assert len(cache.list_entries(tmp_path)) == 1

    removed = cache.clear(tmp_path)
    assert removed == 2  # .npz + .json
    assert cache.list_entries(tmp_path) == []


def test_cache_module_does_not_import_torch() -> None:
    """Regression guard: cache.py must stay torch-free so `render` path is clean.

    Runs in a subprocess so prior-imported torch in the pytest session doesn't
    pollute the check.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        from llm_trace import cache  # noqa: F401
        assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}\n{result.stdout}"
