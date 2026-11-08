"""Crowd study: NARCBench committees with more agents, in two designs. Written by Claude (Swarm sprint, 7 Nov 2026).

Every episode keeps NARCBench's 4-agent core (2 colluders + 2 honest in collusion runs, 4 honest in control runs; names,
styles, option labels and prompts as in generation/core.py) and adds honest "bystanders" that use the scenario's own
honest mandate.

  interactive  One episode per size N (4, 8, 16): all N agents see the whole public discussion, speaking order shuffled
               each round. Colluders can react to the crowd. Episodes at different N are matched by scenario only.

  nested       One 16-agent episode per scenario. The core four never see a bystander. Bystander i (1..12) sees the
               core and bystanders 1..i. A view with the first m bystanders therefore never mentions an absent agent,
               and the colluders' words are identical in every view (see views.py).

Bystander styles: 40% aggressive / 60% standard, seeded by scenario index (same in collusion and control runs).

    python swarm/crowd/generate.py --core-py <narcbench>/generation/core.py --out <dir> --design interactive \
        --sizes 4 8 16 --scenarios 0 1 2 --modes collusion control --model gemma-4-31b-it --max-usd 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_prompts as AP  # noqa: E402
import narc  # noqa: E402
from adaption import make_llm  # noqa: E402
from llm import BudgetExceeded, Halted, RunLog  # noqa: E402

N_BYSTANDERS = 12  # 4 core + 12 = 16 agents
VOTE_RES = (r"RECOMMENDATION:\s*\*{0,2}\[?Option\s*([A-D1-9])\]?\*{0,2}",
            r"RECOMMENDATION:\s*\*{0,2}\[?([A-D1-9])\]?\*{0,2}\b")


def parse_vote(text: str) -> str:
    for pat in VOTE_RES:
        m = re.search(pat, text or "", re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return "?"


MODES = ("collusion", "control", "dissent", "collusion_dissent", "solo", "collusion_solo")
SOLO = ("solo", "collusion_solo")


VARIETIES = ("none", "wording", "roles", "both")


def cast(N: dict, idx: int, mode: str, variety: str = "none", plain_colluders: bool = False) -> dict:
    """Agents, roles and system prompts for scenario `idx`, up to 16 agents. Core = NARCBench's 4-name draw.

    collusion          core[:2] collude (NARCBench)
    control            all honest (NARCBench)
    dissent            core[:2] are sincere dissenters instead: same seats and stance as colluders, no coordination
    collusion_dissent  core[:2] collude AND bystanders[:2] are sincere dissenters (needs size >= 6)
    solo               core[:2] are lone advocates: the colluders' NARCBench system prompt verbatim (same motive and
                       instruction, same style seats) but no partner, no private channel and the ordinary discussion
                       template. Style-matched to colluders, with the coordination removed. (Added 8 Nov after the
                       sincere dissenters turned out to be identifiable from prompt echo alone.)
    collusion_solo     core[:2] collude AND bystanders[:2] are lone advocates (needs size >= 6)
    In the solo modes the advocates are stored in the `dissenters` field (role "advocate"), so the existing
    dissenter metrics measure colluders vs their uncoordinated twins.
    """
    if mode not in MODES:
        raise ValueError(mode)
    pool, sc = N["NAME_POOL"], N["SCENARIOS"][idx]
    core = random.Random(idx).sample(pool, 4)  # core.py:4107
    colluders = core[:2] if mode in ("collusion", "collusion_dissent", "collusion_solo") else []
    rest = [p for p in pool if p not in core]
    bystanders = random.Random(10_000 + idx).sample(rest, N_BYSTANDERS)
    dissenters = (core[:2] if mode in ("dissent", "solo") else
                  bystanders[:2] if mode in ("collusion_dissent", "collusion_solo") else [])
    lab = narc.labels(sc, idx)
    assign = narc.style_assignments()[idx]
    sys_prompts, styles = {}, {}
    col_tmpl = N["COLLUDER_SYSTEM_A"] if idx % 2 == 0 else N["COLLUDER_SYSTEM_B"]
    ci = hi = 0
    for a in core:
        if a in colluders or (mode == "solo" and a in dissenters):
            style = assign["colluder"][ci]
            ci += 1
            instr = sc["lowkey_instruction"] if style == "lowkey" else sc["deception_instruction"]
            sys_prompts[a] = col_tmpl.format(agent_name=a, motive=sc["collusion_motive"], deception_instruction=instr)
        else:
            style = (assign["honest"][hi] if mode in ("collusion", "collusion_dissent", "collusion_solo", "solo")
                     else assign["control"][hi])
            hi += 1
            mand = sc["aggressive_mandate"] if style == "aggressive" else sc["honest_mandate"]
            sys_prompts[a] = N["HONEST_SYSTEM"].format(agent_name=a, honest_mandate=mand)
        styles[a] = style
    srng = random.Random(20_000 + idx)
    for a in bystanders:
        style = "aggressive" if srng.random() < 0.4 else "standard"
        mand = sc["aggressive_mandate"] if style == "aggressive" else sc["honest_mandate"]
        sys_prompts[a] = N["HONEST_SYSTEM"].format(agent_name=a, honest_mandate=mand)
        styles[a] = style
    for j, a in enumerate(dissenters):  # overrides the honest prompt; registers mirror the colluder seat styles
        reg = assign["colluder"][j]
        if mode in SOLO:  # lone advocate: the colluder prompt verbatim, nothing else
            instr = sc["lowkey_instruction"] if reg == "lowkey" else sc["deception_instruction"]
            sys_prompts[a] = col_tmpl.format(agent_name=a, motive=sc["collusion_motive"], deception_instruction=instr)
            styles[a] = f"advocate_{reg}"
        else:
            sys_prompts[a] = AP.DISSENTER_SYSTEM[reg].format(agent_name=a, target=lab["target"],
                                                             target_desc=lab["options"][lab["target"]])
            styles[a] = f"dissenter_{reg}"
    droles = "advocate" if mode in SOLO else "dissenter"
    roles = {a: ("colluder" if a in colluders else droles if a in dissenters else "honest")
             for a in core + bystanders}
    # Test 3: one voice and/or background line per member. Seeded by scenario only, so a seat gets the same line in
    # every mode and size (collusion and control committees differ only in the colluder prompts).
    personas = {}
    if variety != "none":
        seats = core + bystanders
        pv = random.Random(30_000 + idx).sample(range(len(AP.VOICES)), len(seats))
        pr = random.Random(40_000 + idx).sample(range(len(AP.ROLES)), len(seats))
        for i, a in enumerate(seats):
            if plain_colluders and a in colluders:
                continue
            lines = []
            if variety in ("roles", "both"):
                lines.append(AP.ROLE_LINE.format(role=AP.ROLES[pr[i]]))
            if variety in ("wording", "both"):
                lines.append(AP.VOICE_LINE.format(voice=AP.VOICES[pv[i]]))
            personas[a] = " ".join(lines)
            sys_prompts[a] = sys_prompts[a] + "\n\n" + personas[a]
    return {"scenario": sc, "core": core, "colluders": colluders, "dissenters": dissenters, "bystanders": bystanders,
            "roles": roles, "styles": styles, "system": sys_prompts, "personas": personas, **lab}


def _hist(msgs: list[dict]) -> str:
    return "\n".join(f"[{m['author']}]: {m['text']}" for m in msgs) if msgs else "(No prior discussion)"


class Episode:
    def __init__(self, N: dict, llm, idx: int, mode: str, design: str, size: int, temperature: float,
                 max_tokens: int, variety: str = "none", plain_colluders: bool = False):
        self.N, self.llm, self.idx, self.mode, self.design, self.size = N, llm, idx, mode, design, size
        self.variety = variety
        self.c = cast(N, idx, mode, variety, plain_colluders)
        self.params = {"temperature": temperature, "max_tokens": max_tokens}
        self.calls = self.errors = 0
        self.usd = 0.0
        self.tok_in = self.tok_out = 0
        self.id = f"{design}__{self.c['scenario']['id']}__{mode}__n{size}" + ("" if variety == "none" else f"__var-{variety}")

    def _call(self, agent: str, user: str, phase: str, rnd: int) -> str:
        for attempt in range(3):  # retry empty replies (3 of 1,080 pilot calls came back empty at the token limit)
            r = self.llm.chat([{"role": "system", "content": self.c["system"][agent]},
                               {"role": "user", "content": user}],
                              meta={"episode": self.id, "agent": agent, "phase": phase, "round": rnd,
                                    "attempt": attempt}, **self.params)
            self.calls += 1
            self.usd += r.usd
            if r.ok and r.text.strip():
                return r.text.strip()
        self.errors += 1
        return "[ERROR]"

    def run(self) -> dict:
        c, N, sc = self.c, self.N, self.c["scenario"]
        ctx = dict(context=sc["context"], options_text=c["options_text"])
        private: list[dict] = []
        for i, a in enumerate(c["colluders"]):  # private coordination, as core.py:4212
            partner = c["colluders"][1 - i]
            frame = sc["private_frame"].format(partner=partner)
            if private:
                frame = f"{frame}\n\nMessages so far:\n{_hist(private)}"
            text = self._call(a, N["PRIVATE_COORD_TEMPLATE"].format(**ctx, private_frame=frame), "private", 0)
            private.append({"author": a, "round": 0, "text": text})

        if self.design == "interactive":
            agents = c["core"] + c["bystanders"][: self.size - 4]
            visible = {a: (lambda m: True) for a in agents}
            order_rng = random.Random(self.idx * 1000 + self.size)
        else:  # nested: core sees core only; bystander i sees core + bystanders 1..i
            agents = c["core"] + c["bystanders"]
            rank = {b: i for i, b in enumerate(c["bystanders"])}
            visible = {a: (lambda m: m["author"] in c["core"]) for a in c["core"]}
            for b in c["bystanders"]:
                visible[b] = (lambda m, r=rank[b]: m["author"] in c["core"] or rank[m["author"]] <= r)
            order_rng = random.Random(self.idx * 1000)  # NARCBench's core order seed

        public: list[dict] = []
        for rnd in range(1, N["NUM_DISCUSSION_ROUNDS"] + 1):
            if self.design == "interactive":
                speakers = agents[:]
                order_rng.shuffle(speakers)
            else:
                speakers = c["core"][:]
                order_rng.shuffle(speakers)
                speakers += c["bystanders"]
            for a in speakers:
                hist = _hist([m for m in public if visible[a](m)])
                if a in c["colluders"]:
                    partner = c["colluders"][1 - c["colluders"].index(a)]
                    user = N["COLLUDER_DISCUSSION_TEMPLATE"].format(**ctx, partner=partner,
                                                                    private_history=_hist(private), history=hist,
                                                                    public_frame=sc["public_frame"])
                else:
                    user = N["DISCUSSION_PROMPT_TEMPLATE"].format(**ctx, history=hist,
                                                                  public_frame=sc["public_frame"])
                public.append({"author": a, "round": rnd, "text": self._call(a, user, "discussion", rnd)})

        final_frame = sc["final_frame"].replace("Option A or Option B", c["label_hint"])

        def final(a):
            hist = _hist([m for m in public if visible[a](m)])
            user = N["FINAL_PROMPT_TEMPLATE"].format(**ctx, history=hist, final_frame=final_frame)
            return a, self._call(a, user, "final", N["NUM_DISCUSSION_ROUNDS"] + 1)

        finals = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for a, text in pool.map(final, agents):
                finals[a] = text
        votes = {a: parse_vote(t) for a, t in finals.items()}
        return {
            "id": self.id, "design": self.design, "size": len(agents), "scenario_idx": self.idx,
            "scenario_id": sc["id"], "domain": sc["domain"], "title": sc["title"], "mode": self.mode,
            "model": self.llm.model, "agents": agents, "core": c["core"], "bystanders": c["bystanders"],
            "colluders": c["colluders"], "dissenters": [d for d in c["dissenters"] if d in agents],
            "roles": {a: c["roles"][a] for a in agents},
            "styles": {a: c["styles"][a] for a in agents}, "target": c["target"], "fair": c["fair"],
            "variety": self.variety, "personas": {a: c["personas"][a] for a in agents if a in c["personas"]},
            "options_text": c["options_text"], "context": sc["context"], "private": private, "public": public,
            "finals": finals, "votes": votes,
            "colluders_voted_target": sum(votes[a] == c["target"] for a in c["colluders"]),
            "honest_voted_target": sum(votes[a] == c["target"] for a in agents if c["roles"][a] == "honest"),
            "dissenters_voted_target": sum(votes[a] == c["target"] for a in c["dissenters"] if a in agents),
            "calls": self.calls, "errors": self.errors, "usd": round(self.usd, 6),
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core-py", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--design", choices=["interactive", "nested"], required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16], help="interactive only")
    ap.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", default=["collusion", "control"], choices=MODES)
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--temperature", type=float, default=0.7)  # NARCBench config.py
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=12, help="episodes in parallel")
    ap.add_argument("--run-name", default="crowd-gen")
    ap.add_argument("--variety", default="none", choices=VARIETIES,
                    help="test 3: give each member a voice line (wording), a background line (roles), or both")
    ap.add_argument("--keep-colluders-plain", action="store_true", help="test 3: no variety line for colluders")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    N = narc.load(a.core_py, cache=a.out / "narc_cache.json")
    path = a.out / "episodes.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    run = RunLog(a.run_name, max_usd=a.max_usd, config=vars(a) | {"core_py": str(a.core_py), "out": str(a.out)})
    llm = make_llm(a.model, run)
    sizes = a.sizes if a.design == "interactive" else [4 + N_BYSTANDERS]
    jobs = [Episode(N, llm, i, m, a.design, s, a.temperature, a.max_tokens, a.variety, a.keep_colluders_plain)
            for i in a.scenarios for m in a.modes for s in sizes
            if not (m in ("collusion_dissent", "collusion_solo") and s < 6)]  # needs seats for 2 colluders + 2 dissenters + honest
    jobs = [j for j in jobs if j.id not in done]
    print(f"run {run.run_id}: {len(jobs)} episodes to run ({len(done)} already done)", flush=True)
    lock = threading.Lock()
    halted = None
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = {pool.submit(j.run): j for j in jobs}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except (BudgetExceeded, Halted) as e:
                halted = str(e)
                continue
            with lock, path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{rec['id']}: calls={rec['calls']} errors={rec['errors']} usd={rec['usd']:.4f}", flush=True)
    print(json.dumps({"run_id": run.run_id, "usd": round(run.usd, 4), "calls": run.n_calls, "halted": halted}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
