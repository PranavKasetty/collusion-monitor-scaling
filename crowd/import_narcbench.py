"""Convert NARCBench's published Core transcripts (Qwen3-32B agents, 4 per committee) into crowd episode records, so
the crowd monitors can read the benchmark's own data. Written by Claude (Swarm sprint, 8 Nov 2026) as plumbing.

Input: a folder of run dirs (<domain>__<scenario>__<mode>/ with results.json and run_config.json), as published by
NARCBench (Rose et al. 2026). Scenario context and option text come from the same core.py literals the generator uses
(narc.load), with the run's own label_variant. Public messages are "[Name]: text" strings in speaking order; with 4
agents and 3 rounds, message i is round i // 4 + 1. The private channel is dropped (monitors never see it).

    python -I -X utf8 import_narcbench.py <core runs dir> --cache <narc_cache.json> --out episodes.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import narc  # noqa: E402

MSG = re.compile(r"^\[([^\]]+)\]:\s?(.*)$", re.S)


def convert(run_dir: Path, scenarios: dict) -> dict:
    r = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    c = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    idx, sc = scenarios[c["scenario_id"]]
    lab = narc.labels(sc, c["label_variant"])  # labels() only uses idx % 4, so the variant itself works
    assert (lab["target"], lab["fair"]) == (c["target_option"], c["fair_option"]), run_dir.name
    agents = c["agents"]
    public = []
    for i, s in enumerate(r["public_messages"]):
        m = MSG.match(s)
        assert m and m.group(1) in agents, (run_dir.name, s[:80])
        public.append({"author": m.group(1), "text": m.group(2).strip(), "round": i // len(agents) + 1})
    colluders = [a for a, role in r["roles"].items() if role == "colluder"]
    return {"id": f"narcbench__{c['scenario_id']}__{c['mode']}__n4", "design": "interactive", "size": len(agents),
            "scenario_idx": idx, "scenario_id": c["scenario_id"], "domain": sc["domain"], "title": sc["title"],
            "mode": c["mode"], "model": "qwen3-32b (NARCBench published)", "agents": agents, "core": agents,
            "bystanders": [], "colluders": colluders, "dissenters": [], "context": sc["context"],
            "options_text": lab["options_text"], "target": lab["target"], "fair": lab["fair"], "public": public,
            "finals": r["final_texts"], "votes": r["votes"], "collusion_success": r.get("collusion_success"),
            "source": f"NARCBench published core/{run_dir.name}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path)
    ap.add_argument("--cache", type=Path, required=True, help="narc_cache.json (or pass --core-py)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    lit = narc.load(None, a.cache) if a.cache.exists() else sys.exit(f"no cache at {a.cache}")
    scenarios = {s["id"]: (i, s) for i, s in enumerate(lit["SCENARIOS"])}
    eps = [convert(d, scenarios) for d in sorted(a.runs.iterdir()) if (d / "results.json").exists()]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(json.dumps({"episodes": len(eps), "by_mode": {m: sum(e["mode"] == m for e in eps)
                                                       for m in sorted({e["mode"] for e in eps})}}))


if __name__ == "__main__":
    main()
