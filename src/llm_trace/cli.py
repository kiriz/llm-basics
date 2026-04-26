"""typer CLI — `llm-trace run | render | list-cache | clear-cache`.

Critical discipline: the `render` subcommand must NOT import torch/transformers.
That's the killer feature — iterate on visualizations using only cached traces,
no model load required. All torch-adjacent imports are inside the `run` function
body (lazy imports); the module top imports only torch-free things.

Matrix loop lives inline here — no separate matrix.py per the simplicity bias.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from llm_trace import cache, config
from llm_trace.renderers import get_renderer
from llm_trace.trace_data import TraceData

app = typer.Typer(help="LLM inference tracer — step-by-step visualization.",
                  no_args_is_help=True)


# ── run ────────────────────────────────────────────────────────────────────

@app.command()
def run(
    config_path: Path = typer.Option(
        Path("trace.yaml"), "--config", "-c", help="YAML config file"
    ),
    prompt: List[str] = typer.Option(
        [], "--prompt", "-p",
        help="One-off prompt string. Repeatable. Replaces the config's `prompts:` list.",
    ),
    system: Optional[str] = typer.Option(
        None, "--system", "-s",
        help="System prompt applied to every --prompt in this run. "
             "Example: 'You are a Physicist'.",
    ),
    override: List[str] = typer.Option(
        [], "--override", "-o",
        help="Config override 'dotted.key=value'. Repeatable. JSON values allowed.",
    ),
    offline: bool = typer.Option(
        True, "--offline/--online",
        help="Skip HF Hub staleness check (offline default).",
    ),
) -> None:
    """Collect traces for the matrix in the config, then run the configured renderers."""
    import os
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Lazy-import anything that pulls torch:
    from llm_trace.collector import Collector, load_model

    cfg = config.load_config(config_path, overrides=override)
    if prompt:
        cfg["prompts"] = [
            {
                "name": f"adhoc_{i}" if len(prompt) > 1 else "adhoc",
                "text": t,
                **({"system": system} if system else {}),
            }
            for i, t in enumerate(prompt)
        ]
    elif system:
        # Apply system to all YAML prompts that don't already specify one.
        for p in cfg.get("prompts") or []:
            p.setdefault("system", system)
    models = cfg.get("models") or []
    prompts = cfg.get("prompts") or []
    if not models or not prompts:
        typer.secho("config has no models or prompts", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    cache_dir = Path(cfg["cache"]["dir"])
    use_cache = bool(cfg["cache"]["enabled"])
    col_cfg = config.collection_config_from(cfg)
    gen_params = config.gen_params_from(cfg)

    typer.secho(
        f"Matrix: {len(models)} models × {len(prompts)} prompts = {len(models)*len(prompts)} cells",
        fg=typer.colors.CYAN,
    )

    traces: list[TraceData] = []
    collector = Collector(col_cfg, cache_dir=cache_dir, use_cache=use_cache)

    for m in models:
        loaded = load_model(m["id"])
        for p in prompts:
            prompt_value = _prompt_value(p)
            sys_prompt = p.get("system")
            label = _prompt_label(p)
            if sys_prompt:
                label += f"  [sys: {sys_prompt[:40]!r}]"
            typer.secho(f"  → {m['id']}  |  {label}", fg=typer.colors.YELLOW)
            trace = collector.collect(loaded, prompt_value, gen_params, system=sys_prompt)
            traces.append(trace)

    out_dir = Path(cfg.get("out_dir", "./out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_all(traces, cfg.get("renderers", []), out_dir, cfg)

    typer.secho(f"\n✓ {len(traces)} trace(s) written to {out_dir}", fg=typer.colors.GREEN)


# ── render ─────────────────────────────────────────────────────────────────

@app.command()
def render(
    config_path: Path = typer.Option(Path("trace.yaml"), "--config", "-c"),
    override: List[str] = typer.Option([], "--override", "-o"),
) -> None:
    """Re-render from the cache — does NOT load any model."""
    cfg = config.load_config(config_path, overrides=override)
    models = cfg.get("models") or []
    prompts = cfg.get("prompts") or []
    cache_dir = Path(cfg["cache"]["dir"])
    gen_params = config.gen_params_from(cfg)

    if not models or not prompts:
        typer.secho("config has no models or prompts", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    traces: list[TraceData] = []
    misses: list[str] = []
    for m in models:
        for p in prompts:
            prompt_value = _prompt_value(p)
            templated = prompt_value if isinstance(prompt_value, str) else _encode_chat(prompt_value)
            key = cache.cache_key(m["id"], templated, gen_params)
            hit = cache.load(key, cache_dir)
            if hit is None:
                misses.append(f"{m['id']} × {_prompt_label(p)}")
                continue
            arrays, meta = hit
            traces.append(TraceData.from_cache_payload(arrays, meta))

    if misses:
        typer.secho(
            f"cache miss for {len(misses)} cell(s) — run `llm-trace run` first:",
            fg=typer.colors.YELLOW, err=True,
        )
        for m in misses:
            typer.secho(f"  - {m}", fg=typer.colors.YELLOW, err=True)

    if not traces:
        typer.secho("no cached traces found — nothing to render", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    out_dir = Path(cfg.get("out_dir", "./out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_all(traces, cfg.get("renderers", []), out_dir, cfg)

    typer.secho(f"✓ rendered {len(traces)} cached trace(s)", fg=typer.colors.GREEN)


# ── list-cache ─────────────────────────────────────────────────────────────

@app.command("list-cache")
def list_cache(
    cache_dir: Path = typer.Option(Path(".trace_cache"), "--cache-dir"),
) -> None:
    """Enumerate cached cells with model id, prompt preview, and size."""
    entries = cache.list_entries(cache_dir)
    if not entries:
        typer.echo(f"(no cached entries under {cache_dir})")
        return

    typer.secho(f"{len(entries)} cached entrie(s) in {cache_dir}:", fg=typer.colors.CYAN)
    typer.echo(f"  {'KEY':<18} {'SIZE':>10}  {'MODEL':<36}  PROMPT")
    typer.echo(f"  {'-'*18} {'-'*10}  {'-'*36}  {'-'*40}")
    for e in entries:
        size_kb = e["size_bytes"] / 1024.0
        prompt = (e.get("prompt") or "")[:40]
        typer.echo(
            f"  {e['key']:<18} {size_kb:>9.1f}K  "
            f"{(e.get('model_id') or '?'):<36}  {prompt}"
        )


# ── embeddings (explorer) ──────────────────────────────────────────────────

@app.command("embeddings")
def embeddings(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt whose tokens are highlighted in the scatter."),
    model: str = typer.Option("distilgpt2", "--model", "-m"),
    out: Optional[Path] = typer.Option(None, "--out", "-o",
        help="Output HTML path. Default: out/embeddings_<model-slug>.html"),
    n_neighbors: int = typer.Option(8, "--neighbors", help="Nearest-neighbor count per prompt token."),
    background: int = typer.Option(200, "--background", help="Random vocab tokens for density texture."),
    cap: int = typer.Option(500, "--cap", help="Max tokens to plot."),
    offline: bool = typer.Option(True, "--offline/--online"),
) -> None:
    """Build a self-contained HTML page exploring the model's embedding matrix."""
    import os
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from llm_trace.embeddings import explore
    from llm_trace.renderers.animated_v2 import _short_model_slug
    if out is None:
        out = Path(f"out/embeddings_{_short_model_slug(model)}.html")
    result = explore(
        model_id=model,
        prompt=prompt,
        out_path=out,
        n_neighbors=n_neighbors,
        n_random_background=background,
        cap_total=cap,
    )
    typer.secho(
        f"\n✓ Plotted {result.n_tokens_plotted} tokens "
        f"(PCA captured {result.pca_variance_explained*100:.1f}% of variance) "
        f"-> {result.out_path}",
        fg=typer.colors.GREEN,
    )


# ── clear-cache ────────────────────────────────────────────────────────────

@app.command("clear-cache")
def clear_cache(
    cache_dir: Path = typer.Option(Path(".trace_cache"), "--cache-dir"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Remove all cached trace files under cache_dir."""
    entries = cache.list_entries(cache_dir)
    if not entries:
        typer.echo(f"(nothing to clear in {cache_dir})")
        return

    if not yes:
        confirm = typer.confirm(f"Remove {len(entries)} entrie(s) from {cache_dir}?")
        if not confirm:
            typer.echo("aborted")
            raise typer.Exit(1)

    removed = cache.clear(cache_dir)
    typer.secho(f"✓ removed {removed} file(s)", fg=typer.colors.GREEN)


# ── Helpers ────────────────────────────────────────────────────────────────

def _prompt_value(p: dict):
    """Accept either {text: ...} or {chat: [...]} config entries."""
    if "chat" in p:
        return p["chat"]
    return p["text"]


def _prompt_label(p: dict) -> str:
    name = p.get("name") or ""
    preview = p.get("text") or ""
    if "chat" in p and not preview and p["chat"]:
        preview = str(p["chat"][0].get("content", ""))[:30]
    return f"{name} ({preview[:40]!r})" if name else repr(preview[:40])


def _encode_chat(msgs: list) -> str:
    """Fallback-serialize a chat message list when we cannot use a tokenizer
    (render path, no torch). The collector used the real chat template on save,
    so the cache key was computed from that. To look up by key here we need the
    SAME string the collector hashed. That requires a tokenizer, which we don't
    have in render. Workaround: assume the collector stored the templated_prompt
    in the cache; here we just serialize the list so the render path still
    produces a valid key format. Cache lookups will miss if the prompt was a
    chat list — user must run `llm-trace run` first to warm the cache.
    """
    import json as _json
    return _json.dumps(msgs)


def _render_all(traces: list[TraceData], renderers: list[str], out_dir: Path, cfg: dict) -> None:
    if not renderers:
        typer.echo("(no renderers configured — nothing to do)")
        return

    for r_name in renderers:
        if r_name == "comparison":
            fn = get_renderer("comparison")
            path = fn(traces, cfg={"out_dir": str(out_dir)})
            typer.echo(f"  [comparison] -> {path}")
            continue

        fn = get_renderer(r_name)
        for trace in traces:
            if r_name == "terminal":
                fn(trace, cfg={"out_dir": str(out_dir)})
            else:
                path = fn(trace, cfg={"out_dir": str(out_dir)})
                typer.echo(f"  [{r_name}] -> {path}")


if __name__ == "__main__":
    app()
