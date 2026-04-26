"""Shared helpers for renderers — HTML escape, filename slug, model slug.

Kept torch-free (it just handles strings) so anything imported from here is
safe inside the `llm-trace render` path. Hosts the *single* canonical
implementation; renderer modules MUST import from here rather than rolling
their own (the inconsistency this file removes silently mismatched the `'`
escape across 5 renderers).
"""

from __future__ import annotations


def html_escape(s: str) -> str:
    """Escape the 5 characters that change meaning in HTML/JS attributes.

    Includes single-quote escape so the result is safe for both `"..."` and
    `'...'`-quoted attribute contexts in the renderer's string templates.
    """
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def slug(s: str, max_len: int = 40) -> str:
    """Filename-safe slug: alnum + underscore, capped length."""
    cleaned = "".join(c if c.isalnum() else "_" for c in s.strip())
    return (cleaned[:max_len] or "untitled").rstrip("_")


def short_model_slug(model_id: str) -> str:
    """Compact lowercase slug for the model half of an output filename.

    Examples:
        'distilgpt2'                            -> 'distilgpt2'
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0'    -> 'tinyllama'
        'meta-llama/Llama-3.2-1B-Instruct'      -> 'llama'
    """
    last = model_id.rsplit("/", 1)[-1]
    head = last.split("-", 1)[0].split(".", 1)[0]
    out = "".join(c.lower() if c.isalnum() else "_" for c in head).strip("_")
    return out or "model"
