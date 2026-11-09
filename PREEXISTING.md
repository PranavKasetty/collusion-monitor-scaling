# Pre-existing code and prior work

Sprint kickoff: 6 Nov 2026 19:00 UTC (7 Nov 00:30 IST).

## Written before the sprint

| file | what it is | last modified | sha256 |
|---|---|---|---|
| `vendor/llm.py` | the author's LLM client from his sprint kit: one interface for several OpenAI-compatible backends, a per-call JSONL log, spend cap, retries, replay | 2026-11-06 12:09 IST | `6a8c3565ce65ed8c5aa75978a138008626e6d90ac1f76e474de66906ada6cee8` |

## Written during the sprint (6-8 Nov 2026)

Everything in `crowd/`. After submission (9 Nov) two analysis scripts were added that package one-off analyses run during the sprint, with the same code and outputs: `crowd/advocates.py` (lone-advocate tables and behaviour check) and `crowd/nb_narcbench.py` (word-only classifier on NARCBench's transcripts). `crowd/import_narcbench.py` gained a `--core-py` option; its output is unchanged. The README was rewritten for reproduction.

## Third-party prior work used

- NARCBench / HELIOS (Rose et al. 2026, arXiv 2604.01151; github.com/aaronrose227/narcbench, MIT): committee
  scenarios, name pool, honest and colluder prompt templates, discussion templates. Read from its `src/core.py` at run
  time, not copied here.
- UK AISI ControlArena default monitor prompts: shape of our monitor prompts (reasoning before a 0-100 score).
