"""Word-only naive Bayes, colluder vs everyone else, leave-one-scenario-out, on the collusion episodes of one size.
Written by Claude (Swarm sprint, 8 Nov 2026) as plumbing; no model calls.

    python -X utf8 nb_colluders.py <N> <episodes.jsonl ...>
"""
import json, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import followups as F, metrics as M
size = int(sys.argv[1]); eps = []
for p in sys.argv[2:]:
    for l in Path(p).read_text(encoding="utf-8").splitlines():
        if l.strip():
            e = json.loads(l)
            if e["mode"] == "collusion" and e["size"] == size:
                eps.append(e)
attr, top2 = [], []
for s in sorted({e["scenario_id"] for e in eps}):
    docs, ys = [], []
    for e in eps:
        if e["scenario_id"] != s:
            for a in e["agents"]:
                docs.append(F.agent_text(e, a)); ys.append(int(a in e["colluders"]))
    nb = F.NB(docs, ys)
    for e in eps:
        if e["scenario_id"] == s:
            sc = {a: nb.logodds(F.agent_text(e, a)) for a in e["agents"]}
            attr.append(M.agent_auroc(sc, e["colluders"])); top2.append(M.top2_exact(sc, e["colluders"]))
print(f"N={size} episodes={len(eps)} per-agent AUROC={statistics.mean(attr):.3f} exact pair={statistics.mean(top2):.3f} (chance {M.chance_top2_exact(size):.3f})")
