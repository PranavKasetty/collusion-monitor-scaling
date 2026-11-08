"""NARCBench scenarios and prompt templates, loaded from the published generator without executing it.

NARCBench (Rose et al. 2026, arXiv 2604.01151; github.com/aaronrose227/narcbench, MIT) ships its 50 committee
scenarios, name pool and prompt templates as literals in generation/core.py. We parse that file with `ast` and
`literal_eval` only, so none of its code runs. The scenario text (including the colluder instructions) is NARCBench's,
used verbatim; nothing here authors new collusion instructions.

Written by Claude during the Swarm sprint (7 Nov 2026) as plumbing.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

WANT = ("NAME_POOL", "SCENARIOS", "HONEST_SYSTEM", "COLLUDER_SYSTEM_A", "COLLUDER_SYSTEM_B", "PRIVATE_COORD_TEMPLATE",
        "DISCUSSION_PROMPT_TEMPLATE", "COLLUDER_DISCUSSION_TEMPLATE", "FINAL_PROMPT_TEMPLATE", "NUM_DISCUSSION_ROUNDS")


def load(core_py: str | Path, cache: str | Path | None = None) -> dict:
    """Return {name: value} for the literals in WANT. Caches to JSON if `cache` is given."""
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text(encoding="utf-8"))
    tree = ast.parse(Path(core_py).read_text(encoding="utf-8"))
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in WANT:
            got[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(WANT) - set(got)
    if missing:
        raise ValueError(f"core.py is missing {sorted(missing)}")
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
    return got


def style_assignments(n: int = 50, seed: int = 42) -> list:
    """Copy of NARCBench's _compute_style_assignments (core.py:3915), so 4-agent runs match theirs."""
    rng = random.Random(seed)
    col = [("standard", "standard")] * 20 + [("standard", "lowkey")] * 10 + [("lowkey", "standard")] * 10 + \
          [("lowkey", "lowkey")] * 10
    rng.shuffle(col)
    hon = [("standard", "standard")] * 20 + [("standard", "aggressive")] * 10 + [("aggressive", "standard")] * 10 + \
          [("aggressive", "aggressive")] * 10
    rng.shuffle(hon)
    ctrl = [("standard",) * 4] * 10 + [("standard", "standard", "standard", "aggressive")] * 10 + \
           [("standard", "standard", "aggressive", "aggressive")] * 20 + \
           [("standard", "aggressive", "aggressive", "aggressive")] * 10
    rng.shuffle(ctrl)
    return [{"colluder": col[i], "honest": hon[i], "control": ctrl[i]} for i in range(n)]


def labels(scenario: dict, idx: int) -> dict:
    """NARCBench's option-label variants (core.py:4111-4137): A/B or 1/2, and which slot holds the corrupt option."""
    variant = idx % 4
    l1, l2 = ("1", "2") if variant >= 2 else ("A", "B")
    fair_desc, corrupt_desc = scenario["options"]["A"], scenario["options"]["B"]
    if variant % 2 == 1:
        opts, target, fair = {l1: corrupt_desc, l2: fair_desc}, l1, l2
    else:
        opts, target, fair = {l1: fair_desc, l2: corrupt_desc}, l2, l1
    return {"options": opts, "target": target, "fair": fair, "variant": variant,
            "options_text": "\n".join(f"Option {k}: {v}" for k, v in opts.items()),
            "label_hint": f"Option {l1} or Option {l2}"}
