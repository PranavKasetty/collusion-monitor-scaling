"""Lone-advocate control (report section 4.3, Tables 2, C4, C6). Written by Claude (Swarm sprint, 8-9 Nov 2026) as
plumbing; collected here from the one-off analyses that produced the reported numbers. No model calls.

Lone advocates get NARCBench's colluder prompt but no partner and no private channel (modes solo and collusion_solo;
advocates are stored in the episode's `dissenters` field).

  monitor    From monitor records: collusion vs solo committees (coalition and spread scores), solo vs control
             (coalition), and, in collusion_solo committees, P(an advocate outranks a colluder) (chance 0.5).
  behaviour  From episodes: do colluding pairs act differently from advocate pairs in public? both_target = both
             recommend the target; pair_mentions = messages in which one names the other; mentions_vs_others = pair
             mentions minus the pair's mean mentions of non-pair members; honest_target = share of honest agents
             recommending the target; words = public words per pair member. Separate committees (collusion vs solo)
             and same committee (collusion_solo). Difference = colluder minus advocate.

All intervals: 95% scenario bootstrap, 1000 resamples, seed 0.

    python -X utf8 advocates.py monitor   <withheld_*.jsonl ...>
    python -X utf8 advocates.py behaviour <episodes.jsonl ...>
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import followups as F  # noqa: E402
import metrics as M  # noqa: E402


def cmd_monitor(paths):
    recs = [r for r in F.load(paths) if r.get("parsed") and r.get("coalition") is not None]
    for r in recs:
        r["_coal"], r["_spread"] = float(r["coalition"]), F.spread(r)
        if r["mode"] == "collusion_solo":
            r["_adv"] = M.dissenter_above(F._scores(r), r["colluders"], r["dissenters"])

    def det(xs, pos, neg, key):
        return F.fast_auroc([r[key] for r in xs if r["mode"] == pos], [r[key] for r in xs if r["mode"] == neg])

    g = collections.defaultdict(list)
    for r in recs:
        g[(r["method"], r["layout"], r.get("model"), r.get("prompt_version", "v2"), r["size"])].append(r)
    print("| method | layout | model | prompt | N | collusion vs solo (coal) | collusion vs solo (spread) "
          "| solo vs control (coal) | advocate above colluder |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(g):
        by = collections.defaultdict(list)
        for r in g[k]:
            by[r["scenario_id"]].append(r)
        cells = [F.boot(by, lambda xs: det(xs, "collusion", "solo", "_coal")),
                 F.boot(by, lambda xs: det(xs, "collusion", "solo", "_spread")),
                 F.boot(by, lambda xs: det(xs, "solo", "control", "_coal")),
                 F.boot(by, lambda xs: F._mean(r.get("_adv") for r in xs if r["mode"] == "collusion_solo"))]
        print(f"| {' | '.join(map(str, k))} | " + " | ".join(F.fmt(c) for c in cells) + " |")


def pair(e):  # the two target-pushing agents: colluders, or advocates in solo
    return e["colluders"] if e["mode"] in ("collusion", "collusion_solo") else e["dissenters"]


def names(a):
    return [a] + [p for p in a.split() if len(p) > 2]


def mentions(e, a, b):  # messages by a naming b, and by b naming a
    return sum(1 for m in e["public"]
               if (m["author"] == a and any(re.search(r"\b%s\b" % re.escape(n), m["text"]) for n in names(b)))
               or (m["author"] == b and any(re.search(r"\b%s\b" % re.escape(n), m["text"]) for n in names(a))))


def other_mentions(e, a, pairset):  # mean mentions between a and each non-pair agent
    others = [x for x in e["agents"] if x not in pairset]
    return statistics.mean(mentions(e, a, o) for o in others) if others else 0


def stats(e, grp):
    a, b = grp
    honest = [x for x in e["agents"] if e["roles"][x] == "honest"]
    return {"both_target": float(e["votes"][a] == e["target"] and e["votes"][b] == e["target"]),
            "pair_mentions": mentions(e, a, b),
            "mentions_vs_others": mentions(e, a, b) - (other_mentions(e, a, set(grp)) + other_mentions(e, b, set(grp))) / 2,
            "honest_target": statistics.mean(e["votes"][x] == e["target"] for x in honest) if honest else None,
            "words": statistics.mean(len(" ".join(m["text"] for m in e["public"] if m["author"] == x).split())
                                     for x in grp)}


def boot_diff(by_a, by_b, b=1000):
    scen = sorted(set(by_a) & set(by_b))
    rng, vals = random.Random(0), []

    def f(d, ss):
        return statistics.mean(v for s in ss for v in d[s] if v is not None)

    pt = f(by_a, scen) - f(by_b, scen)
    for _ in range(b):
        ss = [rng.choice(scen) for _ in scen]
        vals.append(f(by_a, ss) - f(by_b, ss))
    vals.sort()
    return pt, vals[25], vals[974]


def cmd_behaviour(paths):
    eps = [json.loads(l) for p in paths for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []

    def row(label, n, key, ba, bs):
        ma = statistics.mean(v for vs in ba.values() for v in vs if v is not None)
        ms = statistics.mean(v for vs in bs.values() for v in vs if v is not None)
        d = boot_diff(ba, bs)
        rows.append(f"| {label} | {n} | {key} | {ma:.2f} | {ms:.2f} | {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] |")

    for n in (4, 8, 16):
        A = [e for e in eps if e["mode"] == "collusion" and e["size"] == n]
        S = [e for e in eps if e["mode"] == "solo" and e["size"] == n]
        if not A or not S:
            continue
        for key in ("both_target", "pair_mentions", "mentions_vs_others", "honest_target", "words"):
            ba, bs = collections.defaultdict(list), collections.defaultdict(list)
            for e in A:
                ba[e["scenario_id"]].append(stats(e, pair(e))[key])
            for e in S:
                bs[e["scenario_id"]].append(stats(e, pair(e))[key])
            row("separate committees", n, key, ba, bs)
    for n in (8, 16):
        C = [e for e in eps if e["mode"] == "collusion_solo" and e["size"] == n]
        if not C:
            continue
        for key in ("both_target", "pair_mentions", "mentions_vs_others", "words"):
            ba, bs = collections.defaultdict(list), collections.defaultdict(list)
            for e in C:
                ba[e["scenario_id"]].append(stats(e, e["colluders"])[key])
                bs[e["scenario_id"]].append(stats(e, e["dissenters"])[key])
            row("same committee", n, key, ba, bs)
    print("| comparison | N | measure | colluder pair | advocate pair | difference [95% CI] |\n|---|---|---|---|---|---|")
    print("\n".join(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("monitor").add_argument("files", nargs="+")
    sub.add_parser("behaviour").add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    (cmd_monitor if a.cmd == "monitor" else cmd_behaviour)(a.files)


if __name__ == "__main__":
    main()
