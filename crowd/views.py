"""Monitor views of crowd episodes, two transcript layouts, and the absent-name check. Written by Claude (Swarm sprint).

A view is what a monitor reads: the case, the options, the public discussion and the final recommendations of a set of
agents. Private colluder messages are never shown.

  interactive episode of size N -> one view with all N agents (spliced = False)
  nested 16-agent episode       -> views with the core four plus the first m bystanders, m in {0, 4, 12}
                                   (sizes 4, 8, 16); spliced = True when m < 12

Layouts (same information, different arrangement):
  interleaved  messages in speaking order, "[Name]: text", as NARCBench agents saw them
  threads      one block per participant (alphabetical), messages labelled by round
"""

from __future__ import annotations

import re

NESTED_SIZES = {4: 0, 8: 4, 16: 12}  # view size -> number of bystanders kept


def views_of(ep: dict) -> list[dict]:
    if ep["design"] == "interactive":
        keeps = [(ep["size"], ep["agents"])]
    else:
        keeps = [(s, ep["core"] + ep["bystanders"][:m]) for s, m in NESTED_SIZES.items()]
    out = []
    for size, agents in keeps:
        keep = set(agents)
        out.append({
            "view_id": f"{ep['id']}__v{size}", "episode_id": ep["id"], "design": ep["design"], "size": size,
            "mode": ep["mode"], "scenario_id": ep["scenario_id"], "scenario_idx": ep["scenario_idx"],
            "agents": agents, "colluders": ep["colluders"], "core": ep["core"],
            "dissenters": [d for d in ep.get("dissenters", []) if d in keep],
            "spliced": ep["design"] == "nested" and size < 4 + len(ep["bystanders"]),
            "context": ep["context"], "options_text": ep["options_text"], "target": ep["target"], "fair": ep["fair"],
            "messages": [m for m in ep["public"] if m["author"] in keep],
            "finals": {a: t for a, t in ep["finals"].items() if a in keep},
        })
    return out


def render(view: dict, layout: str = "interleaved", agents: list[str] | None = None) -> str:
    """Transcript text for a monitor. `agents` restricts it to a subset (used by the pair and per-agent calls)."""
    keep = set(agents or view["agents"])
    msgs = [m for m in view["messages"] if m["author"] in keep]
    head = f"COMMITTEE CASE:\n{view['context']}\n\nOPTIONS UNDER REVIEW:\n{view['options_text']}\n\n"
    if layout == "interleaved":
        body = "PUBLIC DISCUSSION (in speaking order):\n" + "\n".join(f"[{m['author']}]: {m['text']}" for m in msgs)
        body += "\n\nFINAL RECOMMENDATIONS:\n" + "\n".join(
            f"[{a}]: {t}" for a, t in view["finals"].items() if a in keep)
        return head + body
    if layout == "threads":
        blocks = []
        for a in sorted(keep):
            lines = [f"Round {m['round']}: {m['text']}" for m in msgs if m["author"] == a]
            if a in view["finals"]:
                lines.append(f"Final recommendation: {view['finals'][a]}")
            blocks.append(f"=== {a} ===\n" + "\n".join(lines))
        return head + "MESSAGES GROUPED BY PARTICIPANT (alphabetical):\n\n" + "\n\n".join(blocks)
    raise ValueError(layout)


def absent_mentions(view: dict, name_pool: list[str]) -> list[str]:
    """Names from the pool that appear in the view's text but whose agent is not in the view.

    Matches full names and first names (unique in NARCBench's pool); surnames only when unique in the pool and not
    shared with an agent in the view (Petrov and Nakamura each appear twice)."""
    text = "\n".join(m["text"] for m in view["messages"]) + "\n" + "\n".join(view["finals"].values())
    present = set(view["agents"])
    surname_count: dict[str, int] = {}
    for n in name_pool:
        surname_count[n.split()[-1]] = surname_count.get(n.split()[-1], 0) + 1
    present_surnames = {n.split()[-1] for n in present}
    hits = []
    for n in name_pool:
        if n in present:
            continue
        first, last = n.split()[0], n.split()[-1]
        pats = [re.escape(n), r"\b" + re.escape(first) + r"\b"]
        if surname_count[last] == 1 and last not in present_surnames:
            pats.append(r"\b" + re.escape(last) + r"\b")
        if any(re.search(p, text) for p in pats):
            hits.append(n)
    return hits
