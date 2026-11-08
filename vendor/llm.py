"""One swappable model call + an append-only run log, shared by every sprint experiment.

    from llm import LLM, RunLog
    run = RunLog("pilot-infection", max_usd=5.0)            # logs under $SPRINT_LOG_DIR or ~/sprint-logs
    m = LLM("openrouter", "openai/gpt-oss-20b", run=run, provider="groq")   # provider: optional OpenRouter pin
    r = m.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=50, meta={"arm": "control", "item": "q1"})
    print(r.text, r.usd)
    replies = m.map([dict(messages=[...], seed=s, meta={...}) for s in range(10)])   # threads, same logging

Backends (all OpenAI-compatible chat completions):
    openrouter  https://openrouter.ai/api/v1         key OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY; reports cost
    hf          https://router.huggingface.co/v1      key HF_TOKEN (e.g. CohereLabs/tiny-aya-global:cohere)
    groq        https://api.groq.com/openai/v1        key GROQ_API_KEY
    vllm        $VLLM_BASE_URL (…/v1)                 key VLLM_API_KEY; self-hosted, API cost 0 (GPU billed elsewhere)
    mock        no network; deterministic replies for tests and dry runs

Every call (success or failure) is one JSON line in <run dir>/calls.jsonl: time, call id, backend, model, provider,
params, seed, messages, reply text, tool calls, finish reason, tokens, usd, latency, error, and your `meta`.
Keys and headers are never logged. Logs live outside the working folder so an agent can't edit them, and the JSONL
uploads to the Hugging Face Hub as a dataset as is (`runlog.py export`).

Spending: the RunLog holds one cap for all models in the run. Cost is OpenRouter's reported cost when present,
else tokens x the price you pass (usd_per_m_in/out), else a deliberately high $1 per million tokens. When the total
reaches the cap, further calls raise BudgetExceeded. Resuming a run (RunLog(..., resume=run_id)) re-reads the spend.
Reasoning models (gpt-oss, Qwen3, ...) spend max_tokens on thinking first: with a small max_tokens the reply is
empty and finish_reason is "length" (seen on gpt-oss-20b with max_tokens=20). Give them room (>= 500) or pass
extra_body={"reasoning": {"effort": "low"}} on OpenRouter. The log keeps the reasoning text and token count.
Replay: LLM(..., replay="<run id or run dir>") answers each call from that run's log instead of the network, matched
on (model, messages, seed, params) and on how many times the same call came before. Seeds alone don't reproduce
API outputs (checked 2026-11-05: Llama-3.1-8B differs across OpenRouter providers, gpt-oss-20b and Tiny Aya differ
even on one provider at T=0), so a "rerun from seed" button has to replay. replay_strict=True raises ReplayMiss on
a call that isn't in the log; otherwise the call goes to the model and is logged as live.
Retries: 429/5xx/timeouts back off (honouring Retry-After) up to `max_attempts`; 400/403 fail that call only;
401/402/404 halt the model (nothing will succeed), except OpenRouter's 404 "every candidate endpoint
was ignored", which means no provider supports this call's parameters (tools, logprobs): that call fails alone.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

BACKENDS = {
    "openrouter": ("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY")),
    "hf": ("https://router.huggingface.co/v1", ("HF_TOKEN",)),
    "groq": ("https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "vllm": (None, ("VLLM_API_KEY",)),
    "mock": (None, ()),
}


class BudgetExceeded(RuntimeError):
    pass


class Halted(RuntimeError):
    pass


class ReplayMiss(KeyError):
    pass


def _replay_key(model: str, messages: list, seed, params: dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "seed": seed, "params": params}, sort_keys=True,
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class RunLog:
    """An append-only log for one run, with the run's spending cap."""

    def __init__(self, name: str, max_usd: float = 5.0, root: str | Path | None = None, resume: str | None = None,
                 config: dict | None = None):
        root = Path(root or os.environ.get("SPRINT_LOG_DIR") or Path.home() / "sprint-logs")
        if resume:
            self.run_id = resume
        else:
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_id = f"{stamp}-{name}-{uuid.uuid4().hex[:6]}"
        self.dir = root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.dir / "calls.jsonl"
        self.max_usd = max_usd
        self.lock = threading.Lock()
        self.usd = 0.0
        self.n_calls = 0
        if resume and self.calls_path.exists():
            for line in self.calls_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.usd += rec.get("usd") or 0.0
                    self.n_calls += 1
        meta_path = self.dir / "run.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                "run_id": self.run_id, "name": name, "started": _dt.datetime.now().astimezone().isoformat(),
                "max_usd": max_usd, "git_sha": _git_sha(), "cwd": os.getcwd(), "argv": sys.argv,
                "config": config or {}}, indent=1), encoding="utf-8")

    def check_budget(self) -> None:
        if self.usd >= self.max_usd:
            raise BudgetExceeded(f"run {self.run_id} spent ${self.usd:.4f} of its ${self.max_usd} cap")

    def write(self, rec: dict) -> None:
        with self.lock:
            self.usd += rec.get("usd") or 0.0
            self.n_calls += 1
            with self.calls_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@dataclass
class Reply:
    text: str
    ok: bool
    usd: float = 0.0
    call_id: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    raw_usage: dict = field(default_factory=dict)


def _as_dict(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: _as_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


class MockClient:
    """Stands in for openai.OpenAI. `reply_fn(messages, kwargs) -> str` (default: echo the last user message)."""

    def __init__(self, reply_fn: Callable | None = None, fail_first: list | None = None, usd_per_call: float | None = None):
        self.reply_fn = reply_fn or (lambda msgs, kw: "echo: " + str(msgs[-1]["content"])[:200])
        self.fail_first = list(fail_first or [])  # exceptions to raise on the first calls, in order
        self.usd_per_call = usd_per_call
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        if self.fail_first:
            raise self.fail_first.pop(0)
        text = self.reply_fn(kw["messages"], kw)
        n_in = sum(len(str(m.get("content", "")).split()) for m in kw["messages"])
        usage = SimpleNamespace(prompt_tokens=n_in, completion_tokens=len(text.split()), cost=self.usd_per_call)
        msg = SimpleNamespace(content=text, tool_calls=None, model_extra={})
        return SimpleNamespace(model=kw["model"], usage=usage, model_extra={"provider": "mock"},
                               choices=[SimpleNamespace(message=msg, finish_reason="stop")])


class LLM:
    def __init__(self, backend: str, model: str, run: RunLog, *, provider: str | None = None,
                 base_url: str | None = None, usd_per_m_in: float | None = None, usd_per_m_out: float | None = None,
                 defaults: dict | None = None, workers: int = 4, max_attempts: int = 6, timeout: float = 180,
                 client: Any = None, sleep: Callable[[float], None] = time.sleep,
                 replay: str | Path | None = None, replay_strict: bool = False):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
        self.backend, self.model, self.run, self.provider = backend, model, run, provider
        self.defaults = defaults or {}
        self.workers, self.max_attempts, self.sleep = workers, max_attempts, sleep
        self.price = (usd_per_m_in, usd_per_m_out)
        if backend == "vllm" and self.price == (None, None):
            self.price = (0.0, 0.0)
        self.halted: str | None = None
        self._replay: dict[str, list[dict]] = {}
        self._replay_seen: dict[str, int] = {}
        self._replay_lock = threading.Lock()
        self.replay_strict = replay_strict
        self.replay_source = None
        if replay is not None:
            rdir = Path(replay)
            if not (rdir / "calls.jsonl").exists():
                rdir = run.dir.parent / str(replay)
            if not (rdir / "calls.jsonl").exists():
                raise FileNotFoundError(f"no calls.jsonl for replay source {replay}")
            self.replay_source = str(rdir)
            with (rdir / "calls.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("ok") and r.get("model") == model:
                        k = _replay_key(model, r["messages"], r.get("seed"), r.get("params") or {})
                        self._replay.setdefault(k, []).append(r)
        if client is not None:
            self.client = client
        elif backend == "mock":
            self.client = MockClient()
        else:
            from openai import OpenAI

            url, key_envs = BACKENDS[backend]
            url = base_url or url or os.environ.get("VLLM_BASE_URL")
            if not url:
                raise ValueError("vllm backend needs base_url= or VLLM_BASE_URL (ending in /v1)")
            key = next((os.environ[k] for k in key_envs if os.environ.get(k)), None)
            if not key:
                raise ValueError(f"set one of {key_envs} in the environment")
            self.client = OpenAI(base_url=url, api_key=key, timeout=timeout, max_retries=0)

    def _cost(self, usage) -> float:
        reported = getattr(usage, "cost", None)
        if reported is not None:
            return float(reported)
        pin, pout = self.price
        tin, tout = getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0
        if pin is None or pout is None:
            return (tin + tout) / 1e6  # unknown price: assume a high $1/M so the cap stays conservative
        return (tin * pin + tout * pout) / 1e6

    def chat(self, messages: list[dict], *, seed: int | None = None, meta: dict | None = None,
             tools: list | None = None, **params) -> Reply:
        if self.halted:
            raise Halted(self.halted)
        self.run.check_budget()
        params = {**self.defaults, **params}
        user_extra = dict(params.pop("extra_body", None) or {})
        kw: dict = {"model": self.model, "messages": messages, **params}
        if seed is not None:
            kw["seed"] = int(seed) % 2**31
        if tools:
            kw["tools"] = tools
        extra: dict = user_extra
        if user_extra:
            params = {**params, "extra_body": user_extra}  # logged with the other params
        if self.backend == "openrouter":
            extra["usage"] = {"include": True}
            if self.provider:
                extra["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        if extra:
            kw["extra_body"] = extra
        call_id = uuid.uuid4().hex[:12]
        if self.replay_source is not None:
            k = _replay_key(self.model, messages, seed, params)
            with self._replay_lock:
                n = self._replay_seen.get(k, 0)
                hits = self._replay.get(k, [])
                hit = hits[n] if n < len(hits) else None
                if hit is not None:
                    self._replay_seen[k] = n + 1
            if hit is not None:
                rec = {k2: hit.get(k2) for k2 in ("backend", "model", "provider_pin", "params", "seed", "messages",
                                                  "prompt_sha", "text", "tool_calls", "reasoning", "finish_reason",
                                                  "served_model", "provider", "prompt_tokens", "completion_tokens")}
                rec.update({"time": time.time(), "call_id": call_id, "run_id": self.run.run_id, "meta": meta or {},
                            "ok": True, "usd": 0.0, "latency_s": 0.0, "attempts": 0, "error": None,
                            "replayed_from": {"source": self.replay_source, "call_id": hit.get("call_id")}})
                self.run.write(rec)
                return Reply(hit.get("text") or "", True, 0.0, call_id, hit.get("tool_calls") or [],
                             hit.get("finish_reason"))
            if self.replay_strict:
                raise ReplayMiss(f"call not in replay log {self.replay_source} (occurrence {n + 1})")
        rec = {"time": time.time(), "call_id": call_id, "run_id": self.run.run_id, "backend": self.backend,
               "model": self.model, "provider_pin": self.provider, "params": params, "seed": seed,
               "tools": [t.get("function", {}).get("name") for t in tools] if tools else None,
               "messages": messages,
               "prompt_sha": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16],
               "meta": meta or {}}
        error = None
        t0 = time.time()
        for attempt in range(self.max_attempts):
            wait = min(120.0, 2.0 * 2**attempt)
            try:
                r = self.client.chat.completions.create(**kw)
                msg, usage = r.choices[0].message, r.usage
                usd = self._cost(usage)
                tool_calls = _as_dict(getattr(msg, "tool_calls", None)) or []
                extra_msg = getattr(msg, "model_extra", None) or {}
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
                rec.update({"ok": True, "text": msg.content or "", "tool_calls": tool_calls,
                            "reasoning": extra_msg.get("reasoning") or getattr(msg, "reasoning_content", None),
                            "reasoning_tokens": reasoning_tokens,
                            "finish_reason": r.choices[0].finish_reason, "served_model": getattr(r, "model", None),
                            "provider": (getattr(r, "model_extra", None) or {}).get("provider"),
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "usd": usd, "latency_s": round(time.time() - t0, 3), "attempts": attempt + 1,
                            "error": None})
                self.run.write(rec)
                return Reply(msg.content or "", True, usd, call_id, tool_calls, r.choices[0].finish_reason)
            except Exception as e:  # noqa: BLE001 - classify below
                status = getattr(e, "status_code", None)
                error = f"{type(e).__name__}" + (f" HTTP {status}" if status else "") + f": {str(e)[:160]}"
                if status == 404 and "every candidate endpoint" in str(e).lower():
                    break  # OpenRouter: no endpoint supports THIS call's parameters (e.g. tools); fail the call only
                if status in (401, 402, 404):
                    self.halted = error
                    break
                if status in (400, 403, 422):
                    break
                resp = getattr(e, "response", None)
                ra = resp.headers.get("retry-after", "") if resp is not None else ""
                if ra.replace(".", "", 1).isdigit():
                    wait = min(300.0, max(wait, float(ra)))
                if attempt < self.max_attempts - 1:
                    self.sleep(wait)
        rec.update({"ok": False, "text": "", "usd": 0.0, "latency_s": round(time.time() - t0, 3), "error": error})
        self.run.write(rec)
        if self.halted:
            raise Halted(self.halted)
        return Reply("", False, 0.0, call_id, error=error)

    def map(self, calls: list[dict]) -> list[Reply]:
        """Run many chat() calls in threads; each dict holds chat()'s arguments. Order is preserved.
        BudgetExceeded / Halted inside a call becomes a failed Reply; check `ok`."""

        def one(kw):
            try:
                return self.chat(**kw)
            except (BudgetExceeded, Halted, ReplayMiss) as e:
                return Reply("", False, error=f"{type(e).__name__}: {e}")

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(one, calls))
