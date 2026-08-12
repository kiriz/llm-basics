#!/usr/bin/env python3
"""Regenerate the exhibits in docs/how-llms-are-wired.html.

The page's whole argument rests on those blocks being real, so this exists to
make them checkable: it fetches each model's published tokenizer config from
HuggingFace, renders that model's own chat_template on one fixed pair of
messages, and prints the exact raw string the model would receive.

    python fetch-chat-templates.py

No weights are downloaded — a tokenizer config is a few KB. Meta's Llama and
Google's Gemma are licence-gated and return 401 without a token, which is why
the page cites Meta's documentation instead of showing an artefact, and uses
NVIDIA's Llama-derived Nemotron to display the Llama-3 delimiters.
"""

from __future__ import annotations

import datetime
import json
import urllib.request

from jinja2.sandbox import ImmutableSandboxedEnvironment

MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Name three primary colors."},
]

# Ordered as the page presents them: no template, then simplest to most elaborate.
MODELS = [
    "distilgpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-7B-Instruct",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "openai/gpt-oss-20b",
]

BASE = "https://huggingface.co/{repo}/resolve/main/{name}"


def fetch(repo: str, name: str) -> str | None:
    try:
        with urllib.request.urlopen(BASE.format(repo=repo, name=name), timeout=30) as r:
            return r.read().decode()
    except Exception:
        return None


def template_for(repo: str) -> tuple[str | None, dict]:
    """Returns (template, config). Newer models keep the template in its own file."""
    raw = fetch(repo, "tokenizer_config.json")
    cfg = json.loads(raw) if raw else {}
    tmpl = cfg.get("chat_template")
    if isinstance(tmpl, list):          # some repos ship a list of named templates
        tmpl = tmpl[0]["template"]
    if not tmpl:
        tmpl = fetch(repo, "chat_template.jinja")
    return tmpl, cfg


def environment() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    env.globals["strftime_now"] = lambda f: datetime.datetime.now().strftime(f)
    return env


def main() -> None:
    for repo in MODELS:
        tmpl, cfg = template_for(repo)
        print(f"\n{'=' * 72}\n{repo}\n{'=' * 72}")
        if not tmpl:
            print("no chat template — this model was never taught roles")
            continue
        rendered = environment().from_string(tmpl).render(
            messages=MESSAGES,
            add_generation_prompt=True,
            bos_token=cfg.get("bos_token") or "",
            eos_token=cfg.get("eos_token") or "",
        )
        print(rendered)


if __name__ == "__main__":
    main()
