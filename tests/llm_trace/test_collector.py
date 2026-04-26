"""Collector tests: distilgpt2 smoke, chat-template branch, cache-hit, determinism."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_trace import collector

# ── Pure-unit tests (no model download) ────────────────────────────────────

def test_chat_template_applied_when_tokenizer_has_template():
    fake_tok = MagicMock()
    fake_tok.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    fake_tok.apply_chat_template.return_value = "<|user|>\nHi<|assistant|>\n"

    original, templated = collector._resolve_prompt(fake_tok, "Hi", is_chat=True)

    assert original == "Hi"
    assert templated == "<|user|>\nHi<|assistant|>\n"
    fake_tok.apply_chat_template.assert_called_once()


def test_chat_template_not_applied_for_non_chat_model():
    fake_tok = MagicMock()
    original, templated = collector._resolve_prompt(fake_tok, "Hi", is_chat=False)

    assert original == "Hi"
    assert templated == "Hi"
    fake_tok.apply_chat_template.assert_not_called()


def test_chat_template_accepts_message_list_input():
    fake_tok = MagicMock()
    fake_tok.chat_template = "some template"
    fake_tok.apply_chat_template.return_value = "<formatted>"
    msgs = [{"role": "user", "content": "Hi"}]

    original, templated = collector._resolve_prompt(fake_tok, msgs, is_chat=True)

    assert json.loads(original) == msgs
    assert templated == "<formatted>"


def test_top_k_returns_descending_order():
    vals = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7])
    ids, values = collector._top_k(vals, 3)
    assert list(ids) == [3, 5, 1]           # argmax indices
    assert list(values) == [0.9, 0.7, 0.5]  # sorted desc


def test_softmax_sums_to_one_and_is_nonneg():
    vals = np.array([-10.0, 0.0, 5.0, 5.1])
    probs = collector._softmax(vals)
    assert np.isclose(probs.sum(), 1.0)
    assert (probs >= 0).all()


def test_softmax_stable_for_large_logits():
    vals = np.array([1000.0, 1001.0, 999.0])
    probs = collector._softmax(vals)
    assert np.isfinite(probs).all()
    assert np.isclose(probs.sum(), 1.0)


# ── Integration tests (require distilgpt2 download) ────────────────────────

@pytest.fixture(scope="module")
def distilgpt2():
    try:
        return collector.load_model("distilgpt2")
    except Exception as e:
        pytest.skip(f"distilgpt2 unavailable ({e}) — network/offline environment")


def test_collect_distilgpt2_smoke(distilgpt2, tmp_path: Path):
    cfg = collector.CollectionConfig(top_k=5, hidden_dims_keep=16)
    c = collector.Collector(cfg, cache_dir=tmp_path)

    trace = c.collect(
        distilgpt2,
        prompt="The stock market today is",
        gen_params={"max_new_tokens": 2, "seed": 42, "stop_on_eos": False},
    )

    # Tokenization
    n_tok = len(trace.tokens)
    assert n_tok == len(trace.token_ids)
    assert n_tok > 0

    # Embeddings — distilgpt2 has learned positional embeddings (wpe exists).
    # As of cache v5 we keep FULL prompt-token embeddings (no dims_keep
    # truncation), so the second axis is the model's hidden_size.
    hidden_size = trace.model_meta["hidden_size"]
    assert trace.embeddings_token.shape == (n_tok, hidden_size)
    assert trace.embeddings_combined.shape == (n_tok, hidden_size)
    assert trace.embeddings_pos is not None
    assert trace.embeddings_pos.shape == (n_tok, hidden_size)
    # combined ≈ token + pos
    assert np.allclose(
        trace.embeddings_combined,
        trace.embeddings_token + trace.embeddings_pos,
        atol=1e-5,
    )

    # Hidden states: input + n_layers (distilgpt2 has 6 layers)
    n_layer = trace.model_meta["n_layer"]
    assert trace.hidden_last.shape == (n_layer + 1, 16)
    assert trace.hidden_last_norms.shape == (n_layer + 1,)
    # Residual stream grows (not strictly, but last layer > input)
    assert trace.hidden_last_norms[-1] > trace.hidden_last_norms[0]

    # Top-K arrays
    assert trace.logits_top_ids.shape == (5,)
    assert trace.logits_top_values.shape == (5,)
    assert trace.probs_top_ids.shape == (5,)
    # Probs sorted descending
    assert (np.diff(trace.probs_top_values) <= 0).all()

    # Attention slicing (default L0H0)
    assert "L0H0" in trace.attentions
    attn = trace.attentions["L0H0"]
    assert attn.shape == (n_tok, n_tok)
    # Upper triangle zero (causal mask)
    for i in range(n_tok):
        for j in range(i + 1, n_tok):
            assert attn[i, j] < 1e-6, f"attention leak at ({i},{j})"

    # Generation trace
    assert len(trace.generation) == 2
    for step in trace.generation:
        assert "token" in step and "id" in step and "prob" in step
        assert "top_alts" in step and len(step["top_alts"]) > 0
        assert 0 <= step["prob"] <= 1

    # Timings populated
    assert trace.timings["first_forward_ms"] > 0
    assert len(trace.timings["per_token_ms"]) == 2


def test_greedy_generation_is_deterministic(distilgpt2, tmp_path: Path):
    cfg = collector.CollectionConfig(top_k=5, hidden_dims_keep=16)
    gen = {"max_new_tokens": 3, "seed": 42, "stop_on_eos": False}

    c1 = collector.Collector(cfg, cache_dir=tmp_path / "run1", use_cache=False)
    c2 = collector.Collector(cfg, cache_dir=tmp_path / "run2", use_cache=False)

    t1 = c1.collect(distilgpt2, "Hello world", gen)
    t2 = c2.collect(distilgpt2, "Hello world", gen)

    assert [s["id"] for s in t1.generation] == [s["id"] for s in t2.generation]


def test_cache_hit_avoids_model_forward(distilgpt2, tmp_path: Path):
    cfg = collector.CollectionConfig(top_k=5, hidden_dims_keep=16)
    c = collector.Collector(cfg, cache_dir=tmp_path)
    gen = {"max_new_tokens": 1, "seed": 42, "stop_on_eos": False}

    c._forward_calls = 0
    _ = c.collect(distilgpt2, "Hello", gen)
    after_first = c._forward_calls
    assert after_first > 0, "first call must run the model"

    _ = c.collect(distilgpt2, "Hello", gen)
    assert c._forward_calls == after_first, (
        f"cache hit should skip model; forward_calls went from "
        f"{after_first} to {c._forward_calls}"
    )


def test_trace_data_cache_roundtrip(distilgpt2, tmp_path: Path):
    cfg = collector.CollectionConfig(top_k=5, hidden_dims_keep=16)
    c = collector.Collector(cfg, cache_dir=tmp_path)
    trace = c.collect(
        distilgpt2,
        "Round-trip me",
        {"max_new_tokens": 1, "seed": 42, "stop_on_eos": False},
    )

    arrays, meta = trace.to_cache_payload()
    restored = collector.TraceData.from_cache_payload(arrays, meta)

    assert restored.model_id == trace.model_id
    assert restored.tokens == trace.tokens
    assert np.array_equal(restored.hidden_last, trace.hidden_last)
    assert np.array_equal(restored.logits_top_ids, trace.logits_top_ids)
    assert set(restored.attentions) == set(trace.attentions)
    for key in trace.attentions:
        assert np.array_equal(restored.attentions[key], trace.attentions[key])
    assert restored.generation == trace.generation
