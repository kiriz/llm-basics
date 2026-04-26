"""Disk cache for TraceData.

Layout per cell:
    <cache_dir>/<key>.npz   — numpy arrays
    <cache_dir>/<key>.json  — metadata (tokens, ids, model info, gen params, timings)

Key: sha1(CACHE_VERSION|model_id|templated_prompt|json(gen_params))[:16]

Writes are atomic: we stage to `.tmp` files and `os.replace()` into place.
Version drift returns None with a clear message instead of raising KeyError.

This module has no torch / transformers dependency — safe to import from
renderers and the `llm-trace render` path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = "v7"   # v7: + LM-head top-K weight rows + tied-weights flag


def cache_key(model_id: str, templated_prompt: str, gen_params: dict[str, Any]) -> str:
    payload = "|".join([
        CACHE_VERSION,
        model_id,
        templated_prompt,
        json.dumps(gen_params, sort_keys=True, default=str),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def save(
    key: str,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    cache_dir: Path,
) -> None:
    """Write arrays (.npz) + meta (.json) atomically under cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_with_version = {"_cache_version": CACHE_VERSION, **meta}

    npz_tmp = cache_dir / f"{key}.npz.tmp"
    json_tmp = cache_dir / f"{key}.json.tmp"
    npz_final = cache_dir / f"{key}.npz"
    json_final = cache_dir / f"{key}.json"

    # Open file handle so np.savez writes to the exact path we gave it
    # (passing a string path makes numpy auto-append .npz).
    with open(npz_tmp, "wb") as f:
        np.savez(f, **arrays)
    json_tmp.write_text(
        json.dumps(meta_with_version, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    os.replace(npz_tmp, npz_final)
    os.replace(json_tmp, json_final)


def load(
    key: str,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    """Load (arrays, meta) for key, or None if missing / version-mismatch.

    Prints a clear message on version drift so user knows to run clear-cache.
    """
    npz = cache_dir / f"{key}.npz"
    js = cache_dir / f"{key}.json"
    if not (npz.exists() and js.exists()):
        return None

    try:
        meta = json.loads(js.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[cache] corrupt metadata for {key}: {e}. Treating as miss.")
        return None

    found_version = meta.get("_cache_version")
    if found_version != CACHE_VERSION:
        print(
            f"[cache] version mismatch for {key}: found {found_version!r}, "
            f"expected {CACHE_VERSION!r}. Treating as miss. "
            f"Run `llm-trace clear-cache` to remove stale entries."
        )
        return None

    with np.load(npz) as data:
        arrays = {name: data[name] for name in data.files}

    meta.pop("_cache_version", None)
    return arrays, meta


def list_entries(cache_dir: Path) -> list[dict[str, Any]]:
    """Enumerate cached cells — metadata only, no arrays loaded."""
    if not cache_dir.exists():
        return []

    entries: list[dict[str, Any]] = []
    for js in sorted(cache_dir.glob("*.json")):
        if js.name.endswith(".tmp"):
            continue
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        key = js.stem
        npz = cache_dir / f"{key}.npz"
        size_bytes = npz.stat().st_size if npz.exists() else 0

        entries.append({
            "key": key,
            "size_bytes": size_bytes,
            "cache_version": meta.get("_cache_version"),
            "model_id": meta.get("model_id"),
            "prompt": meta.get("prompt"),
            "gen_params": meta.get("gen_params"),
        })
    return entries


def clear(cache_dir: Path) -> int:
    """Remove all cache files under cache_dir. Returns count removed."""
    if not cache_dir.exists():
        return 0
    removed = 0
    for f in cache_dir.iterdir():
        if f.is_file() and (f.suffix in {".npz", ".json"} or f.name.endswith(".tmp")):
            f.unlink()
            removed += 1
    return removed
