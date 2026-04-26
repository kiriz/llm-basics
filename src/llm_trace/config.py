"""Config loader — trace.yaml with CLI --override key=value (repeatable).

Keeps things flat and dict-based (no pydantic, no omegaconf, no dataclass
ceremony at the top level). Callers pick out the bits they need and build
typed configs (like CollectionConfig) from the dict.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "models": [],
    "prompts": [],
    "generation": {
        "max_new_tokens": 50,     # lets the model complete its loop to EOS
        "temperature": 0.0,
        "seed": 42,
        "stop_on_eos": True,
        "show_eos_experiment": False,
        "extra_temps_for_viz": [0.1, 0.7, 1.5],
    },
    "collection": {
        "attention": {
            # Which attention slices to persist. Use explicit indices for
            # narrow captures, or the string "all" for every layer / every head.
            "layers": [0],
            "heads": "all",     # enables the HTML head-switcher tabs
        },
        "hidden_dims_keep": 32,
        "top_k": 15,
        "generation_top_k": 5,
    },
    "renderers": ["terminal"],
    "out_dir": "./out",
    "open_browser": False,
    "cache": {
        "dir": ".trace_cache",
        "enabled": True,
    },
}


def load_config(path: Path | str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML + apply overrides; fall back to DEFAULT_CONFIG if path is None.

    Overrides are strings of the form 'dotted.path=value'. Value is parsed as
    JSON if possible (for lists, numbers, bools), otherwise treated as a string.
    """
    cfg = deepcopy(DEFAULT_CONFIG)

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = _merge(cfg, loaded)

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got: {ov!r}")
        key, value_str = ov.split("=", 1)
        value = _parse_override_value(value_str)
        _set_dotted(cfg, key.strip(), value)

    return cfg


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts. Lists replace wholesale (not concatenated)."""
    result = deepcopy(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _parse_override_value(s: str) -> Any:
    """Try JSON first (handles numbers, bools, lists, nulls); fall back to str."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def _set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set d[a][b][c] = value for dotted_key='a.b.c'. Creates missing dicts."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


# ── Convenience accessors ──────────────────────────────────────────────────

def collection_config_from(cfg: dict[str, Any]):
    """Build a collector.CollectionConfig from the loaded dict."""
    from llm_trace.collector import CollectionConfig
    col = cfg.get("collection", {})
    gen = cfg.get("generation", {})
    attn = col.get("attention", {})

    def _as_indices(v, default):
        """List → tuple; string 'all' passes through; fallback to default tuple."""
        if v is None:
            return default
        if isinstance(v, str):
            return v
        return tuple(int(x) for x in v)

    return CollectionConfig(
        attention_layers=_as_indices(attn.get("layers"), (0,)),
        attention_heads=_as_indices(attn.get("heads"), "all"),
        hidden_dims_keep=int(col.get("hidden_dims_keep", 32)),
        top_k=int(col.get("top_k", 15)),
        extra_temps_for_viz=tuple(gen.get("extra_temps_for_viz", [0.1, 0.7, 1.5])),
        generation_top_k=int(col.get("generation_top_k", 5)),
    )


def gen_params_from(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the gen_params dict that Collector.collect consumes."""
    gen = cfg.get("generation", {})
    return {
        "max_new_tokens": int(gen.get("max_new_tokens", 50)),
        "temperature": float(gen.get("temperature", 0.0)),
        "seed": int(gen.get("seed", 42)),
        "stop_on_eos": bool(gen.get("stop_on_eos", True)),
    }
