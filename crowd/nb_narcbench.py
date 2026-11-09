"""Word-only naive Bayes on NARCBench's published transcripts (report section 4.3). Written by Claude (Swarm sprint,
8 Nov 2026) as plumbing; collected here from the one-off analysis that produced the reported numbers. No model calls.

  Core:     leave-one-scenario-out; per-agent attribution AUROC and exact-pair top-2 in collusion runs, and run-level
            collusion vs control AUROC (spread of agent scores).
  Transfer: train on all Core runs, test on Transfer runs (NARCBench's held-out structural shift).

Input: the folder that holds NARCBench's `core/` and `transfer/` run dirs (e.g. scenarios/qwen3_32b from the
published data); each run dir has results.json.

    python -X utf8 nb_narcbench.py <dir with core/ and transfer/>
"""

from __future__ import annotations

import json
import os
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import followups as F  # noqa: E402
import metrics as M  # noqa: E402


def load(root, tier):
    out = []
    for r in sorted(os.listdir(os.path.join(root, tier))):
        p = os.path.join(root, tier, r, "results.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        txt = {a: [] for a in d["roles"]}
        for m in d["public_messages"]:
            mm = re.match(r"\[([^\]]+)\]:\s*(.*)", m, re.S)
            if mm and mm.group(1) in txt:
                txt[mm.group(1)].append(mm.group(2))
        for a, t in (d.get("final_texts") or {}).items():
            if a in txt:
                txt[a].append(t or "")
        parts = r.split("__")
        out.append({"run": r, "scen": parts[1], "mode": d.get("mode", parts[2]), "roles": d["roles"],
                    "text": {a: " ".join(v) for a, v in txt.items()}})
    return out


def fit(runs):
    docs, ys = [], []
    for x in runs:
        for a, r in x["roles"].items():
            docs.append(x["text"][a])
            ys.append(1 if r == "colluder" else 0)
    return F.NB(docs, ys)


def spread(sc):
    v = sorted(sc.values(), reverse=True)
    return st.mean(v[:2]) - st.mean(v[2:]) if len(v) > 2 else v[0]


def main(root):
    core, tr = load(root, "core"), load(root, "transfer")
    att, t2, runscore = [], [], {}
    for s in sorted({x["scen"] for x in core}):
        nb = fit([x for x in core if x["scen"] != s])
        for x in core:
            if x["scen"] == s:
                sc = {a: nb.logodds(t) for a, t in x["text"].items()}
                runscore[x["run"]] = spread(sc)
                col = [a for a, r in x["roles"].items() if r == "colluder"]
                if col:
                    att.append(M.agent_auroc(sc, col))
                    t2.append(M.top2_exact(sc, col, draws=200))
    pos = [runscore[x["run"]] for x in core if x["mode"] == "collusion"]
    neg = [runscore[x["run"]] for x in core if x["mode"] == "control"]
    print(f"Core, leave-one-scenario-out: per-agent AUROC {st.mean(att):.3f}; exact pair in top 2 {st.mean(t2):.3f} "
          f"(chance 0.167); run-level collusion vs control AUROC {M.auroc(pos, neg):.3f}")
    nb = fit(core)
    att, runscore = [], {}
    for x in tr:
        sc = {a: nb.logodds(t) for a, t in x["text"].items()}
        runscore[x["run"]] = spread(sc)
        col = [a for a, r in x["roles"].items() if r == "colluder"]
        if col and len(col) < len(sc):
            att.append(M.agent_auroc(sc, col))
    pos = [runscore[x["run"]] for x in tr if x["mode"] == "collusion"]
    neg = [runscore[x["run"]] for x in tr if x["mode"] == "control"]
    print(f"Transfer (train on Core): per-agent AUROC {st.mean(att):.3f} over {len(att)} runs; run-level AUROC "
          f"{M.auroc(pos, neg):.3f} ({len(pos)} collusion vs {len(neg)} control)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
