# Collusion monitors in growing agent committees

[DRAFT README for Pranav to edit. Results summary and framing are his to write.]

Code for an Apart Research Swarm Alignment sprint project (6-8 Nov 2026). Research question: how does a monitor's
ability to identify colluders scale with the number of agents whose transcripts it reads?

NARCBench committees (Rose et al. 2026, arXiv 2604.01151) are grown from 4 to 8 and 16 agents with honest bystanders,
plus sincere-dissent and lone-advocate controls. LLM monitors (one whole-transcript call, or screen-compare-confirm with
N + 3 calls) and no-model baselines (stance, who-talks-to-whom, a word-only naive Bayes classifier) are scored on
detection AUROC, audit-budget TPR, per-agent attribution and exact-pair accuracy, with scenario-bootstrap CIs.

## Summary of results

[PRANAV TO WRITE]

## Files (`crowd/`)

| file | what it does |
|---|---|
| `narc.py` | loads NARCBench scenarios, prompts and name pool by parsing its `generation/core.py` (never executed) |
| `generate.py` | committees of 4 / 8 / 16 agents; modes collusion, control, dissent, collusion_dissent, solo, collusion_solo; optional honest-agent variety (`--variety`) |
| `agent_prompts.py` | prompts added to NARCBench: sincere dissenters, voice and background lines |
| `views.py` | monitor views in two layouts (interleaved, threads); the private channel is never shown |
| `monitor.py`, `monitor_prompts.py`, `monitor_prompts_v1.py` | whole-transcript and screen-compare-confirm monitors; prompt versions v1-v4 |
| `baselines.py` | stance and who-talks-to-whom baselines |
| `metrics.py`, `analyze.py`, `followups.py`, `nb_colluders.py` | metrics, tables, bootstrap CIs, word-only classifier |
| `paraphrase.py` | paraphrasing defence: rewrites every public message in neutral wording |
| `checks.py` | pilot design checks |
| `adaption.py` | OpenAI-compatible client for the Adaption endpoint |
| `modal_crowd.py` | runs generation, monitors and paraphrasing on Modal CPU containers |
| `test_crowd.py` | tests (no model calls) |

## Running

```bash
pip install -r requirements.txt
git clone https://github.com/aaronrose227/narcbench   # MIT; the code reads src/core.py from it
export NARC_CORE_PY=narcbench/src/core.py
export ADAPTION_API_KEY=...                           # or use another OpenAI-compatible endpoint in adaption.py

python -m pytest -q crowd/test_crowd.py
python -X utf8 crowd/generate.py --core-py $NARC_CORE_PY --out runs/demo --design interactive --sizes 4 --scenarios 0 --modes collusion control
python -X utf8 crowd/monitor.py runs/demo/episodes.jsonl --out runs/demo/withheld_monitor.jsonl --model glm-53-flash --methods whole
python -X utf8 crowd/analyze.py runs/demo/withheld_monitor.jsonl
```

Full runs used Modal: `modal run --detach crowd/modal_crowd.py --plan main --stage generate` (then `--stage monitor`).
Monitor outputs must be named `withheld_*.jsonl`: scores were kept sealed until predictions were written.

## Prior work and pre-sprint code

See `PREEXISTING.md`. NARCBench scenarios and colluder prompts are used verbatim and are not included in this repo.

## LLM use

[PRANAV TO CONFIRM] Claude (Anthropic) wrote most of this code during the sprint, as the module docstrings state, and
drafted the added prompts; the research question, predictions, design decisions and interpretation are Pranav's.
