"""Renderer smoke tests + no-torch guardrail.

The guardrail runs each import in a fresh subprocess to isolate from the pytest
session's already-imported modules. This protects the `llm-trace render`
killer feature: editing a template and re-rendering must not require
loading torch or any model weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[2] / "src"


def _fresh_import_check(module: str) -> subprocess.CompletedProcess:
    script = textwrap.dedent(f"""
        import sys
        import {module}  # noqa: F401
        banned = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))
        assert not banned, f'banned modules pulled in: {{banned[:5]}}'
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )


@pytest.mark.parametrize("module", ["llm_trace.renderers.terminal",
                                     "llm_trace.renderers.png",
                                     "llm_trace.renderers.html",
                                     "llm_trace.renderers.animated"])
def test_renderer_import_does_not_require_torch(module: str) -> None:
    result = _fresh_import_check(module)
    assert result.returncode == 0, (
        f"{module} pulled torch on import:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def _fake_trace():
    """Minimal TraceData-shaped object for renderer smoke tests."""
    from llm_trace.collector import TraceData

    n = 3
    attn = np.tril(np.ones((n, n), dtype=np.float32) / n)
    return TraceData(
        model_id="fake-model",
        prompt="hi there",
        system_prompt="You are a test fixture",
        templated_prompt="hi there",
        gen_params={"max_new_tokens": 2, "seed": 0, "stop_on_eos": False},
        model_meta={
            "model_type": "fake",
            "vocab_size": 100,
            "n_layer": 2,
            "n_head": 2,
            "n_kv_heads": 2,
            "hidden_size": 8,
            "ffn_intermediate": 32,
            "ctx_window": 64,
            "positional_encoding": "learned",
            "normalization": "layernorm",
            "rope_theta": None,
            "eos_ids": [99],
            "eos_tokens": ["</end>"],
        },
        tokens=["hi", " there", "!"],
        token_ids=[1, 2, 3],
        embeddings_token=np.random.RandomState(0).rand(n, 8).astype(np.float32),
        embeddings_pos=np.random.RandomState(1).rand(n, 8).astype(np.float32),
        embeddings_combined=np.random.RandomState(2).rand(n, 8).astype(np.float32),
        attentions={"L0H0": attn},
        hidden_last=np.random.RandomState(3).rand(3, 8).astype(np.float32),
        hidden_last_norms=np.array([1.0, 5.0, 12.0], dtype=np.float32),
        final_hidden_full=np.random.RandomState(4).rand(8).astype(np.float32),
        hidden_norms=np.array([
            [1.0, 1.1, 1.0],   # input layer norms per-token
            [5.0, 5.5, 5.0],   # after layer 1
            [12.0, 12.5, 12.0], # after layer 2
        ], dtype=np.float32),
        logits_top_ids=np.array([10, 20, 30], dtype=np.int64),
        logits_top_values=np.array([5.2, 4.8, 4.1], dtype=np.float32),
        logits_top_tokens=[" cat", " dog", " bird"],
        probs_top_ids=np.array([10, 20, 30], dtype=np.int64),
        probs_top_values=np.array([0.5, 0.3, 0.1], dtype=np.float32),
        probs_top_tokens=[" cat", " dog", " bird"],
        lm_head_top_rows=np.random.RandomState(5).rand(3, 8).astype(np.float32),
        temp_scan=[{"temperature": 0.7, "top": [{"token": " cat", "id": 10, "prob": 0.4}]}],
        generation=[
            {"step": 1, "token": " cat", "id": 10, "prob": 0.5, "ctx_len": 3,
             "is_eos": False, "top_alts": [{"token": " cat", "id": 10, "prob": 0.5}]},
            {"step": 2, "token": " dog", "id": 20, "prob": 0.3, "ctx_len": 4,
             "is_eos": False, "top_alts": [{"token": " dog", "id": 20, "prob": 0.3}]},
        ],
        generation_text=" cat dog",
        per_step_hidden_norms=np.array([[1.0, 5.0, 12.0], [1.1, 5.5, 13.2]], dtype=np.float32),
        per_step_top_ids=np.array([[10, 20, 30], [20, 10, 30]], dtype=np.int64),
        per_step_top_probs=np.array([[0.5, 0.3, 0.1], [0.4, 0.35, 0.08]], dtype=np.float32),
        timings={"model_load_ms": 1.0, "first_forward_ms": 2.0, "per_token_ms": [3.0, 3.0]},
    )


def test_terminal_runs_without_error(capsys):
    from llm_trace.renderers import terminal
    terminal.render(_fake_trace())
    out = capsys.readouterr().out
    assert "fake-model" in out
    assert "Step 1" in out


def test_png_writes_nonempty_file(tmp_path: Path):
    from llm_trace.renderers import png
    out = png.render(_fake_trace(), out_path=tmp_path / "out.png")
    assert out.exists()
    assert out.stat().st_size > 1000   # more than a couple header bytes


def test_html_writes_nonempty_file_with_valid_structure(tmp_path: Path):
    from llm_trace.renderers import html
    out = html.render(_fake_trace(), out_path=tmp_path / "out.html")
    text = out.read_text()
    assert "<html" in text
    assert "const D =" in text
    assert '"model_id": "fake-model"' in text


def test_html_js_does_not_reference_removed_fields(tmp_path: Path):
    """Guard against the 'top5 renamed to top_alts' class of bugs: a JS ref
    to an undefined field silently blanks the whole page because renderAll
    throws and never recovers.

    Scan the JS body for any read of a field that doesn't appear in the
    payload dict. We match `obj.<word>` patterns on identifiers whose set
    is known to come from the per-step payload.
    """
    import re
    from llm_trace.renderers import html
    out = html.render(_fake_trace(), out_path=tmp_path / "out.html")
    text = out.read_text()

    # Pull the D = {...}; literal, parse it, collect first-step keys.
    import json
    m = re.search(r'const D = (\{.*?\});', text, re.DOTALL)
    assert m
    payload = json.loads(m.group(1).replace(r'<\\/', '</'))
    step_keys = set(payload["generated"][0])

    # These are the specific field names our per-step renderers read. Any
    # reference we make must exist on the payload.
    forbidden = {"top5", "top3"}   # historical names that are now renamed
    for name in forbidden:
        assert f"g.{name}" not in text, (
            f"JS references the removed payload field `g.{name}` — "
            f"current per-step keys are {sorted(step_keys)}"
        )


def test_html_escapes_closing_script_tag_in_prompt(tmp_path: Path):
    from llm_trace.renderers import html
    tr = _fake_trace()
    object.__setattr__(tr, "prompt", "nasty </script><img src=x onerror=alert(1)>")
    out = html.render(tr, out_path=tmp_path / "nasty.html")
    text = out.read_text()
    # The prompt appears in the JSON payload — should be neutralized
    assert "</script><img" not in text
    assert "<\\/script>" in text


def test_animated_writes_nonempty_file(tmp_path: Path):
    from llm_trace.renderers import animated
    out = animated.render(_fake_trace(), out_path=tmp_path / "anim.html")
    text = out.read_text()
    assert "<html" in text
    assert "const D =" in text
    assert "Act I" in text     # scene marker present
    assert "Act II" in text
    assert "Act III" in text


def test_html_comparison_grid_lays_out_traces(tmp_path: Path):
    from llm_trace.renderers import html
    t1 = _fake_trace()
    t2 = _fake_trace()
    object.__setattr__(t2, "model_id", "other-model")
    out = html.render_comparison([t1, t2], out_path=tmp_path / "cmp.html")
    text = out.read_text()
    assert "fake-model" in text
    assert "other-model" in text
    assert 'grid-template-columns: repeat(1' in text  # one unique prompt
