"""Run the crowd study's generator and monitors on Modal CPU containers, so runs don't depend on the laptop.
Written by Claude (Swarm sprint, 7 Nov 2026) as plumbing.

Each shard is one Function call running generate.py (or monitor.py) with output in the Volume `crowd-runs`:
    /runs/<plan>/<shard>/episodes.jsonl      generator output (resumable: rerunning skips finished episodes)
    /runs/<plan>/logs/<run id>/calls.jsonl   every model call (sprint-kit RunLog)

    modal run --detach swarm/crowd/modal_crowd.py --plan main          # generation, 5 shards of 10 scenarios
    modal volume get crowd-runs main <local dir>/  # copy back
"""

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SPRINT_KIT = Path.home() / ".claude/skills/sprint-kit/tools/llm.py"
if not SPRINT_KIT.exists():  # public repo: vendored copy of the same pre-sprint file
    SPRINT_KIT = HERE.parent / "vendor" / "llm.py"
CORE_PY = os.environ.get("NARC_CORE_PY", "")  # path to NARCBench generation/core.py (only read at deploy time)

image = (modal.Image.debian_slim(python_version="3.12").pip_install("openai==1.109.1")
         .add_local_dir(str(HERE), "/root/crowd", ignore=["__pycache__", "*.pyc"])
         .add_local_file(str(SPRINT_KIT), "/root/.claude/skills/sprint-kit/tools/llm.py"))
if CORE_PY:
    image = image.add_local_file(CORE_PY, "/root/narc/core.py")

app = modal.App("crowd-study")
runs = modal.Volume.from_name("crowd-runs", create_if_missing=True)

ALL_MODES = ["collusion", "control", "dissent", "collusion_dissent"]
PLANS = {  # plan -> list of (shard name, scenario indices)
    "main": [(f"s{i:02d}", list(range(i, i + 10))) for i in range(0, 50, 10)],
    # test 3 pilots (8 Nov): honest-agent variety, scenarios 0-9, N = 8 only
    **{f"var_{v}": [("s00", list(range(10)))] for v in ("wording", "roles", "both")},
    # test 3 at N = 16 (8 Nov evening): wording variety only, scenarios 0-19, where the v2 inversion is strongest
    "var16_wording": [("s00", list(range(10))), ("s10", list(range(10, 20)))],
    # NARCBench's own published Core transcripts (Qwen3-32B, N = 4), converted by import_narcbench.py; monitor only
    "narcbench": [],
}


@app.function(image=image, cpu=2, memory=2048, volumes={"/runs": runs},
              secrets=[modal.Secret.from_name("adaption-api-key")], timeout=3 * 60 * 60, retries=0)
def generate_shard(plan: str, shard: str, scenarios: list, max_usd: float, workers: int, modes: str = "") -> str:
    import subprocess
    import sys

    out = f"/runs/{plan}/{shard}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, SPRINT_LOG_DIR=f"/runs/{plan}/logs")
    cmd = [sys.executable, "-X", "utf8", "/root/crowd/generate.py", "--core-py", "/root/narc/core.py",
           "--out", out, "--design", "interactive", "--sizes", "4", "8", "16",
           "--scenarios", *map(str, scenarios), "--modes", *(modes.split(",") if modes else ALL_MODES),
           "--max-usd", str(max_usd), "--workers", str(workers), "--run-name", f"crowd-{plan}-{shard}"]
    if plan.startswith("var"):  # test 3: var_<variety> = N 8 pilot, var16_<variety> = N 16; one variety per plan
        i = cmd.index("--sizes")
        cmd[i + 1:i + 4] = ["16" if plan.startswith("var16_") else "8"]
        cmd += ["--variety", plan.split("_", 1)[1]]
    with open(f"{out}/driver.log", "a", encoding="utf-8") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd="/root/crowd", env=env)
    runs.commit()
    return f"{shard}: exit {p.returncode}"


MONITORS = {  # name -> (model, reasoning_effort or "")
    "g26": ("gemma-4-26b-a4b-it", ""),
    "glm": ("glm-53-flash", "low"),
    "qwen": ("qwen3.8-27b", "low"),
    "q397": ("qwen3.5-397b", "low"),  # stronger-monitor test (glm-5.2 spent >4000 reasoning tokens without answering)
}


@app.function(image=image, cpu=2, memory=2048, volumes={"/runs": runs},
              secrets=[modal.Secret.from_name("adaption-api-key")], timeout=6 * 60 * 60, retries=0)
def monitor_shard(plan: str, shard: str, mon: str, max_usd: float, workers: int, pv: str = "v2",
                  methods: str = "whole,scc", max_tokens: int = 4000, layouts: str = "interleaved,threads",
                  modes: str = "") -> str:
    """One monitor model over one shard's episodes; output /runs/<plan>/<shard>/withheld_monitor_<mon>.jsonl (sealed)."""
    import subprocess
    import sys

    model, effort = MONITORS[mon]
    out = f"/runs/{plan}/{shard}"
    env = dict(os.environ, SPRINT_LOG_DIR=f"/runs/{plan}/logs")
    tag = mon + ("" if pv == "v2" else f"_{pv}") + ("" if methods == "whole,scc" else "_" + methods.replace(",", "+"))
    cmd = [sys.executable, "-X", "utf8", "/root/crowd/monitor.py", f"{out}/episodes.jsonl",
           "--out", f"{out}/withheld_monitor_{tag}.jsonl", "--model", model, "--max-usd", str(max_usd),
           "--workers", str(workers), "--run-name", f"crowd-mon-{plan}-{shard}-{tag}",
           "--prompt-version", pv, "--methods", *methods.split(","), "--max-tokens", str(max_tokens),
           "--layouts", *layouts.split(",")]
    if modes:
        cmd += ["--modes", *modes.split(",")]
    if effort:
        cmd += ["--reasoning-effort", effort]
    with open(f"{out}/monitor_{tag}.log", "a", encoding="utf-8") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd="/root/crowd", env=env)
    runs.commit()
    return f"{shard}/{tag}: exit {p.returncode}"


@app.function(image=image, cpu=2, memory=2048, volumes={"/runs": runs},
              secrets=[modal.Secret.from_name("adaption-api-key")], timeout=4 * 60 * 60, retries=0)
def paraphrase_job(plan: str, sizes: str, modes: str, max_usd: float, workers: int) -> str:
    """Rewrite the chosen episodes from every shard into /runs/<plan>/para/episodes.jsonl (resumable cache)."""
    import glob
    import subprocess
    import sys

    out = f"/runs/{plan}/para"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, SPRINT_LOG_DIR=f"/runs/{plan}/logs")
    eps = sorted(glob.glob(f"/runs/{plan}/s*/episodes.jsonl"))
    cmd = [sys.executable, "-X", "utf8", "/root/crowd/paraphrase.py", *eps, "--out", f"{out}/episodes.jsonl",
           "--sizes", *sizes.split(","), "--modes", *modes.split(","), "--max-usd", str(max_usd),
           "--workers", str(workers), "--run-name", f"crowd-para-{plan}"]
    with open(f"{out}/paraphrase.log", "a", encoding="utf-8") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd="/root/crowd", env=env)
    runs.commit()
    return f"para: exit {p.returncode}"


@app.local_entrypoint()
def main(plan: str = "main", max_usd: float = 3.0, workers: int = 20, stage: str = "generate", monitors: str = "g26",
         shards: str = "", pv: str = "v2", methods: str = "whole,scc", max_tokens: int = 4000,
         layouts: str = "interleaved,threads", gen_modes: str = ""):
    """shards: comma list (e.g. s00) to run a subset first; empty = all shards of the plan."""
    keep = set(shards.split(",")) if shards else None
    todo = [(s, sc) for s, sc in PLANS[plan] if keep is None or s in keep]
    if stage == "monitor" and keep:  # allow extra shard folders such as "para"
        todo += [(s, []) for s in sorted(keep) if s not in {x for x, _ in PLANS[plan]}]
    if stage == "paraphrase":
        print(paraphrase_job.remote(plan, "8", "collusion,control,dissent,collusion_dissent", max_usd, workers))
        return
    if stage == "generate":
        calls = [generate_shard.spawn(plan, shard, sc, max_usd, workers, gen_modes) for shard, sc in todo]
    else:  # stage == "monitor": monitors is a comma list of MONITORS keys
        calls = [monitor_shard.spawn(plan, shard, m, max_usd, workers, pv, methods, max_tokens, layouts, gen_modes)
                 for m in monitors.split(",") for shard, _ in todo]
    for c in calls:
        print(c.get())
