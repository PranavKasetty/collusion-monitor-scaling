# Consensus Looks Like Collusion

Code for "Consensus Looks Like Collusion: What LLM Monitors Detect in Growing Agent Committees", an Apart Research
Swarm Alignment Hackathon project (6-8 Nov 2026). The report is [`report/report.pdf`](report/report.pdf).

**Question.** How well does an LLM monitor that reads a committee's discussion find two secretly coordinating agents
as the committee grows from 4 to 8 and 16 agents?

**Setup.** NARCBench's 50 committee scenarios (Rose et al. 2026) grown with honest bystanders, plus controls: all-honest
committees, sincere dissenters, and lone advocates (the colluder prompt with no partner and no private channel).
Agents are Gemma 4 31B. Monitors are zero-shot LLMs (mainly GLM-5.3-flash) and no-model baselines.

**Findings.**
- With a standard monitor prompt, the monitor rates honest committees as more suspicious than collusion committees
  (AUROC 0.23 to 0.44), and its colluder attribution falls from 0.81 at N = 4 to 0.36 at N = 16, below chance. Gemma 4
  26B and Qwen3.5-397B as monitors show the same inversion.
- The cause is shared wording: every agent is the same model with the same briefing. One prompt sentence saying so
  (prompt v4) raises detection to 0.67 to 0.76 and attribution to 0.68 to 0.87, also on NARCBench's own Qwen3-32B
  transcripts (0.49 to 0.76). It does not fix Gemma 4 26B.
- After the fix, collusion committees and lone-advocate committees cannot be told apart (AUROC 0.44 to 0.55), and
  colluders behave like advocates in public. The monitor detects the colluder brief, not coordination.
- Counting votes for the target option (stance baseline) scores 0.94 and beats every LLM monitor.

## Repository layout

| path | contents |
|---|---|
| `crowd/narc.py` | loads NARCBench scenarios, prompts and name pool by parsing its `src/core.py` (never executed) |
| `crowd/generate.py` | committees of 4 / 8 / 16 agents; modes `collusion`, `control`, `dissent`, `collusion_dissent`, `solo`, `collusion_solo`; optional honest-agent variety (`--variety`) |
| `crowd/agent_prompts.py` | prompts added to NARCBench: sincere dissenters, writing-style and background lines |
| `crowd/views.py` | what a monitor reads (public messages and final recommendations, never the private channel), in two layouts |
| `crowd/monitor.py`, `crowd/monitor_prompts.py`, `crowd/monitor_prompts_v1.py` | whole-transcript and screen-compare-confirm monitors; prompt versions v1-v4 |
| `crowd/baselines.py` | stance and who-talks-to-whom baselines (no model calls) |
| `crowd/metrics.py` | AUROC, TPR at a false-flag budget, per-agent attribution, exact pair, dis_above |
| `crowd/analyze.py` | summary table per method, layout, model and N (includes TPR at 5% / 10% false flags) |
| `crowd/followups.py` | `ci`: scenario-bootstrap CIs; `style`: colluders vs dissenters or advocates, including the word-only classifier |
| `crowd/advocates.py` | lone-advocate control: monitor comparisons and the behaviour check |
| `crowd/nb_colluders.py`, `crowd/nb_narcbench.py` | word-only naive Bayes on our committees and on NARCBench's published transcripts |
| `crowd/import_narcbench.py` | converts NARCBench's published Core transcripts into episodes the monitors can read |
| `crowd/paraphrase.py` | paraphrasing defence: rewrites every public message in neutral wording |
| `crowd/wording_mentions.py` | counts monitor notes that mention shared wording (this measure saturated; report section 4.2) |
| `crowd/checks.py` | pilot design checks (interactive vs nested designs) |
| `crowd/adaption.py` | OpenAI-compatible client for the Adaption endpoint used for all model calls |
| `crowd/modal_crowd.py` | optional: runs generation, monitors and paraphrasing on Modal CPU containers |
| `crowd/test_crowd.py` | tests, no model calls |
| `vendor/llm.py` | pre-sprint LLM client (logging, spend cap, retries); see `PREEXISTING.md` |
| `report/` | the report (PDF and its HTML source) and `figures.py`, which draws the two figures from numbers in the report tables |

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/aaronrose227/narcbench        # MIT; scenarios and prompts are read from src/core.py
export NARC_CORE_PY=narcbench/src/core.py
export ADAPTION_API_KEY=...       # any OpenAI-compatible endpoint works: edit BASE_URL and PRICES in crowd/adaption.py
python -m pytest -q crowd/test_crowd.py
```

Smoke test, one scenario at N = 4 (a few cents):

```bash
python -X utf8 crowd/generate.py --core-py $NARC_CORE_PY --out runs/demo --design interactive --sizes 4 --scenarios 0 --modes collusion control
python -X utf8 crowd/monitor.py runs/demo/episodes.jsonl --out runs/demo/withheld_monitor.jsonl --model glm-53-flash --reasoning-effort low --methods whole
python -X utf8 crowd/analyze.py runs/demo/withheld_monitor.jsonl
```

## Reproducing the report

Generated transcripts and monitor outputs are not included (see the report's dual-use appendix), so reproduction means
regenerating them. Agents sample at temperature 0.7 and monitors at 1.0, so rerun numbers will differ from the report
within its confidence intervals. Every command spends against `--max-usd`; every model call is logged as JSONL under
`SPRINT_LOG_DIR`. Monitor outputs are named `withheld_*.jsonl` because they were kept sealed until predictions were
written. The main run can also be launched on Modal, sharded by scenario, with `modal run --detach crowd/modal_crowd.py
--plan main --stage generate` and then `--stage monitor`.

**1. Committees** (main run: 550 episodes; lone advocates: 250).

```bash
S=$(seq 0 49)
python -X utf8 crowd/generate.py --core-py $NARC_CORE_PY --out runs/main --design interactive --sizes 4 8 16 --scenarios $S \
  --modes collusion control dissent collusion_dissent --max-usd 15
python -X utf8 crowd/generate.py --core-py $NARC_CORE_PY --out runs/main --design interactive --sizes 4 8 16 --scenarios $S \
  --modes solo collusion_solo --max-usd 10
```

**2. Monitors and baselines.**

```bash
E=runs/main/episodes.jsonl
M="--model glm-53-flash --reasoning-effort low"
python -X utf8 crowd/monitor.py $E $M --prompt-version v2 --out runs/main/withheld_monitor_glm.jsonl          # whole (2 layouts) + scc
python -X utf8 crowd/monitor.py $E $M --prompt-version v4 --methods whole --layouts interleaved \
  --out runs/main/withheld_monitor_glm_v4_whole.jsonl
python -X utf8 crowd/baselines.py $E --out runs/main/withheld_baselines.jsonl
```

**3. Tables and figures.**

| report item | command |
|---|---|
| Table 1, Figure 1, Table C5 (detection, spread, attribution with CIs) | `python -X utf8 crowd/followups.py ci runs/main/withheld_monitor_glm*.jsonl runs/main/withheld_baselines.jsonl` |
| Table 3 (catch rate at 5% / 10% false flags) | `python -X utf8 crowd/analyze.py runs/main/withheld_monitor_glm*.jsonl runs/main/withheld_baselines.jsonl` |
| Table 2, Figure 2, Table C6 (lone advocates) | `python -X utf8 crowd/advocates.py monitor runs/main/withheld_monitor_glm*.jsonl` |
| Table 2, word-only row (colluders vs advocates) | `python -X utf8 crowd/followups.py style $E --records runs/main/withheld_monitor_glm*.jsonl --mode collusion_solo` |
| Section 4.3, colluders vs sincere dissenters | `python -X utf8 crowd/followups.py style $E --records runs/main/withheld_monitor_glm*.jsonl` |
| Table C4 (behaviour check) | `python -X utf8 crowd/advocates.py behaviour $E` |
| Word-only classifier, colluder vs everyone, per N | `python -X utf8 crowd/nb_colluders.py 8 $E` (and 4, 16) |
| `report/fig1.png`, `report/fig2.png` | `cd report && python figures.py` (numbers copied from Tables 1 and 2) |

**4. NARCBench's published transcripts** (Table C2 and section 4.3). Download the scenario data from the
[NARCBench Hugging Face dataset](https://huggingface.co/datasets/aaronrose227/narcbench) (`scenarios/qwen3_32b/` with
`core/` and `transfer/`).

```bash
Q=data/scenarios/qwen3_32b
python -I -X utf8 crowd/import_narcbench.py $Q/core --core-py $NARC_CORE_PY --out runs/narcbench/episodes.jsonl
for pv in v2 v4; do
  python -X utf8 crowd/monitor.py runs/narcbench/episodes.jsonl $M --prompt-version $pv --methods whole --layouts interleaved \
    --out runs/narcbench/withheld_monitor_glm_$pv.jsonl
done
python -X utf8 crowd/followups.py ci runs/narcbench/withheld_monitor_glm_*.jsonl
python -X utf8 crowd/nb_narcbench.py $Q                                      # word-only classifier, Core and Transfer
```

**5. Other experiments.**

| report item | how |
|---|---|
| Table C1 (prompt ablation, other monitors) | `monitor.py` on scenarios 0-9 with `--prompt-version v1`-`v4` and `--model gemma-4-26b-a4b-it` or `qwen3.5-397b`, `--methods whole --layouts interleaved`; then `followups.py ci` |
| Paraphrasing (section 4.2) | `python -X utf8 crowd/paraphrase.py $E --out runs/para/episodes.jsonl --sizes 8`, then `monitor.py` (v2, v4) and `nb_colluders.py 8` on `runs/para/episodes.jsonl` |
| Table C3 (varied honest agents) | `generate.py ... --variety wording` (or `roles`, `both`) `--sizes 8 --scenarios 0..9 --modes collusion control`; N = 16: `--sizes 16 --scenarios 0..19 --variety wording`; then `monitor.py` v2 / v4 and `followups.py ci` |
| Wording count (section 4.2) | `python -I -X utf8 crowd/wording_mentions.py v2=<v2 calls.jsonl> v4=<v4 calls.jsonl> --out counts.json` on the monitor call logs |
| Pilot design check (section 3.1) | `generate.py --design nested` and `--design interactive` on 3 scenarios, then `checks.py <episodes.jsonl ...> --name-pool <out>/narc_cache.json --out checks.json --model gemma-4-26b-a4b-it` |

## Prior work and LLM use

`PREEXISTING.md` lists what existed before the sprint. NARCBench scenarios and colluder prompts are used verbatim and
are not included in this repo. Claude (Anthropic) wrote most of this code during the sprint, as the module docstrings
state, drafted the added prompts, and drafted the report from the logged results. The research question, predictions,
design decisions and the hand-coded monitor notes that identified the shared-wording confound are Pranav Kasetty's.
