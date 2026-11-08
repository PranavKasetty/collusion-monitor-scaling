# Collusion monitors in growing agent committees

Code for an Apart Research Swarm Alignment sprint project (6-8 Nov 2026). Research question: how does a monitor's
ability to identify colluders scale with the number of agents whose transcripts it reads?

NARCBench committees (Rose et al. 2026, arXiv 2604.01151) are grown from 4 to 8 and 16 agents with honest bystanders,
plus sincere-dissent and lone-advocate controls. LLM monitors (one whole-transcript call, or screen-compare-confirm with
N + 3 calls) and no-model baselines (stance, who-talks-to-whom, a word-only naive Bayes classifier) are scored on
detection AUROC, audit-budget TPR, per-agent attribution and exact-pair accuracy, with scenario-bootstrap CIs.

## Summary of results

Report: [`report/report.pdf`](report/report.pdf), "Consensus Looks Like Collusion: What LLM Monitors Detect in Growing Agent Committees".

- With a standard monitor prompt, GLM-5.3-flash reads unanimous honest committees as coordinated: its coalition answer ranks collusion below control (AUROC 0.23 to 0.44) and its colluder attribution falls from 0.81 at N = 4 to 0.36 at N = 16. The same pattern appears with Gemma 4 26B and Qwen3.5-397B as monitors.
- The monitor's notes cite shared wording, which is expected when every agent is the same model with the same briefing. One prompt sentence saying so (v4) raises detection to 0.67 to 0.76 and attribution to 0.68 to 0.87, and works on NARCBench's own Qwen3-32B transcripts (0.49 to 0.76). It does not fix Gemma 4 26B.
- After the fix, collusion committees and committees with lone advocates (the colluder prompt, no partner, no private channel) cannot be told apart (AUROC 0.44 to 0.55), and colluders behave like advocates in public. The monitor detects the colluder brief, not coordination.
- Counting votes for the target option (stance) scores 0.94 and beats every LLM monitor.

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
| `import_narcbench.py` | converts NARCBench's published Core transcripts into episodes the monitors can read |
| `wording_mentions.py` | counts monitor notes that mention shared wording (this measure saturated; see report section 4.2) |
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

Claude (Anthropic) wrote most of this code during the sprint, as the module docstrings state, drafted the added prompts,
and drafted the report from the logged results. The research question, predictions, design decisions and the hand-coded
monitor notes that identified the shared-wording confound are Pranav's.
