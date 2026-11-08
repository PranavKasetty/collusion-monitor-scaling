"""Adaption's OpenAI-compatible chat endpoint wired into the sprint-kit LLM client (logging + spend cap).

Adaption documents only /responses, but /chat/completions answers too (checked 7 Nov 2026 with gemma-4-31b-it:
HTTP 200, a vLLM backend). Prices are credits per million tokens from docs.adaptionlabs.ai (1 credit = $1).
Written by Claude during the Swarm sprint as plumbing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor"))  # public repo: vendored pre-sprint llm.py
sys.path.insert(0, str(Path.home() / ".claude/skills/sprint-kit/tools"))

from llm import LLM, RunLog  # noqa: E402

BASE_URL = "https://api.prod.adaptionlabs.ai/api/v1"
PRICES = {  # model: (input, output) credits per 1M tokens
    "gemma-4-31b-it": (0.09, 0.34),
    "gemma-4-26b-a4b-it": (0.0765, 0.255),
    "qwen3.5-397b": (0.55, 3.50),
    "qwen3.8-27b": (0.039, 1.25),
    "deepseek-v4-flash": (0.04444, 0.13332),
    "glm-53-flash": (0.15, 0.50),
    "glm-5.2": (1.40, 4.40),
}


def api_key() -> str:
    k = os.environ.get("ADAPTION_API_KEY")
    if not k and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("ADAPTION_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"').strip()
    if not k:
        raise SystemExit("ADAPTION_API_KEY not set (environment or .env)")
    return k


def make_llm(model: str, run: RunLog, client=None, workers: int = 8, **defaults) -> LLM:
    """An LLM on Adaption. Pass `client` (e.g. a MockClient) to run without the network."""
    if model not in PRICES:
        raise ValueError(f"no price for {model}; add it to PRICES")
    pin, pout = PRICES[model]
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=BASE_URL, api_key=api_key(), timeout=180, max_retries=0)
    return LLM("vllm", model, run, client=client, usd_per_m_in=pin, usd_per_m_out=pout, workers=workers,
               defaults=defaults)
