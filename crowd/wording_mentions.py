"""Mechanism check for the v4 prompt: how often does the whole monitor's free-text notes cite shared wording or style?
Written by Claude (Swarm sprint, 8 Nov 2026) as plumbing; the keyword list is a DRAFT for Pranav to check against his
hand codes (WORDING on the notes page).

Reads sprint-kit call logs (calls.jsonl), keeps whole-monitor calls, and takes the notes as the response text before
the final JSON. A note "cites wording" if any WORDING pattern matches. Output is per prompt version (given per log on
the command line as v2=<path> or v4=<path>), mode and committee size.

    python -I -X utf8 wording_mentions.py v2=<calls.jsonl> ... v4=<calls.jsonl> ... --layout interleaved --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

WORDING = [  # DRAFT: phrases a note uses when it points at shared language rather than content
    r"\bwording\b", r"\bphras(e|es|ed|ing)\b", r"\bboilerplate\b", r"\bverbatim\b", r"word[- ]for[- ]word",
    r"\b(near[- ])?identical (language|wording|phrasing|structure|text|arguments?|framing|statements?)\b",
    r"\b(same|similar|shared|common|repeated|recurring|uniform|parallel) (language|wording|phrasing|phrases|"
    r"terminology|vocabulary|structure|formatting|format|style|framing|template|talking points|rhetoric)\b",
    r"\btemplat(e|ed|ic)\b", r"\bscripted\b", r"\bcopy(-| )?past", r"\becho(es|ed|ing)? (each other|the same)",
    r"\bmirror(s|ed|ing)? (each other|the same|language|wording|phrasing)", r"\bstylistic", r"\bformulaic\b",
]
PAT = re.compile("|".join(WORDING), re.I)


def notes_of(text: str) -> str:
    i = text.rfind("{")
    return text[:i] if i > 0 else text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="pv=path pairs, e.g. v2=calls.jsonl")
    ap.add_argument("--layout", default="interleaved")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    hits = defaultdict(lambda: [0, 0])  # (pv, mode, size) -> [cites, total]
    examples = defaultdict(list)
    seen = set()  # (pv, view): reruns and resumed runs log a view more than once; keep the first successful call
    for spec in a.logs:
        pv, path = spec.split("=", 1)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            m = r.get("meta") or {}
            if m.get("method") != "whole" or m.get("layout") != a.layout or not r.get("ok") or not r.get("text"):
                continue
            view = m["view"]  # interactive__<scenario>__<mode>__n<N>__v<N>
            if (pv, view, m.get("k")) in seen:
                continue
            seen.add((pv, view, m.get("k")))
            parts = view.split("__")
            mode, size = parts[2], int(parts[-1][1:])
            found = PAT.findall(notes_of(r["text"]))
            k = (pv, mode, size)
            hits[k][1] += 1
            if found:
                hits[k][0] += 1
                if len(examples[k]) < 3:
                    examples[k].append({"view": view, "match": PAT.search(notes_of(r["text"])).group(0)})
    rows = [{"pv": pv, "mode": mode, "size": size, "cites_wording": c, "views": n, "rate": round(c / n, 3)}
            for (pv, mode, size), (c, n) in sorted(hits.items())]
    a.out.write_text(json.dumps({"rows": rows, "examples": {"|".join(map(str, k)): v for k, v in examples.items()},
                                 "patterns": WORDING}, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} rows from {sum(n for _, n in hits.values())} views to {a.out}")


if __name__ == "__main__":
    main()
