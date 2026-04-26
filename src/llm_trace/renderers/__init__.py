"""Renderers — pure TraceData → output functions.

None of these modules may import torch or transformers. Enforced via test.
"""

from __future__ import annotations

# Registry populated lazily to avoid importing matplotlib/rich when not used.
# Consumers: use get_renderer(name) rather than a static dict.
_RENDERERS = ("terminal", "png", "html", "comparison", "animated", "animated_v2", "animated_v3")


def get_renderer(name: str):
    if name == "terminal":
        from llm_trace.renderers.terminal import render as fn
        return fn
    if name == "png":
        from llm_trace.renderers.png import render as fn
        return fn
    if name == "html":
        from llm_trace.renderers.html import render as fn
        return fn
    if name == "comparison":
        from llm_trace.renderers.html import render_comparison as fn
        return fn
    if name == "animated":
        from llm_trace.renderers.animated import render as fn
        return fn
    if name == "animated_v2":
        from llm_trace.renderers.animated_v2 import render as fn
        return fn
    if name == "animated_v3":
        from llm_trace.renderers.animated_v3 import render as fn
        return fn
    raise ValueError(f"unknown renderer: {name!r}. known: {_RENDERERS}")
