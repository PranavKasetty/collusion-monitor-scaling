"""Offline tests for the crowd study (mock model, no network). Run: python -m pytest swarm/crowd -q"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate  # noqa: E402
import monitor  # noqa: E402
from llm import MockClient, RunLog  # noqa: E402
from adaption import make_llm  # noqa: E402
from views import absent_mentions, render, views_of  # noqa: E402

POOL = [f"First{i} Last{i}" for i in range(30)]
SC = {"id": "t_01", "domain": "d", "title": "T", "context": "CTX", "options": {"A": "fair", "B": "bad"},
      "collusion_motive": "M", "deception_instruction": "D", "lowkey_instruction": "L", "honest_mandate": "H",
      "aggressive_mandate": "AG", "private_frame": "talk to {partner}", "public_frame": "speak",
      "final_frame": "RECOMMENDATION: [Option A or Option B]"}
N = {"NAME_POOL": POOL, "SCENARIOS": [SC] * 3, "NUM_DISCUSSION_ROUNDS": 2,
     "HONEST_SYSTEM": "{agent_name} {honest_mandate}", "COLLUDER_SYSTEM_A": "{agent_name} {motive} {deception_instruction}",
     "COLLUDER_SYSTEM_B": "{agent_name} {deception_instruction} {motive}",
     "PRIVATE_COORD_TEMPLATE": "{context} {options_text} {private_frame}",
     "DISCUSSION_PROMPT_TEMPLATE": "{context} {options_text} HIST:{history} {public_frame}",
     "COLLUDER_DISCUSSION_TEMPLATE": "{context} {options_text} {partner} {private_history} HIST:{history} {public_frame}",
     "FINAL_PROMPT_TEMPLATE": "{context} {options_text} HIST:{history} {final_frame}"}


def reply(msgs, kw):
    sysmsg, user = msgs[0]["content"], msgs[-1]["content"]
    me = " ".join(sysmsg.split()[:2])
    if "RECOMMENDATION" in user:
        return "RECOMMENDATION: Option B"
    hist = user.split("HIST:")[-1]
    seen = sorted({p for p in POOL if f"[{p}]" in hist})
    return f"I am {me}; I saw " + ", ".join(seen)  # mentions every name it can see


def ep(design, size=16, mode="collusion", tmp=Path(".")):
    run = RunLog("t", max_usd=1, root=tmp)
    llm = make_llm("gemma-4-31b-it", run, client=MockClient(reply))
    return generate.Episode(N, llm, 0, mode, design, size, 0.7, 64).run()


def test_nested_views_never_mention_absent_agents(tmp_path):
    e = ep("nested", tmp=tmp_path)
    assert e["size"] == 16 and len(e["colluders"]) == 2 and set(e["colluders"]) <= set(e["core"])
    vs = views_of(e)
    assert [v["size"] for v in vs] == [4, 8, 16]
    for v in vs:
        assert absent_mentions(v, POOL) == [], v["size"]
    # the colluders' public words are identical in every view
    col = [[m["text"] for m in v["messages"] if m["author"] in e["colluders"]] for v in vs]
    assert col[0] == col[1] == col[2]
    # core agents never saw a bystander
    for m in e["public"]:
        if m["author"] in e["core"]:
            assert not any(re.search(r"\b" + b.split()[0] + r"\b", m["text"]) for b in e["bystanders"])


def test_interactive_sizes_and_votes(tmp_path):
    for n in (4, 8):
        e = ep("interactive", size=n, tmp=tmp_path)
        assert e["size"] == n and len(e["finals"]) == n and e["errors"] == 0
        assert e["colluders_voted_target"] == (2 if e["target"] == "B" else 0)
    c = ep("interactive", size=4, mode="control", tmp=tmp_path)
    assert c["colluders"] == [] and set(c["roles"].values()) == {"honest"}


def test_dissent_modes(tmp_path):
    d = ep("interactive", size=4, mode="dissent", tmp=tmp_path)
    assert d["colluders"] == [] and d["private"] == [] and d["dissenters"] == d["core"][:2]
    assert {d["roles"][a] for a in d["dissenters"]} == {"dissenter"}
    cd = ep("interactive", size=8, mode="collusion_dissent", tmp=tmp_path)
    assert cd["colluders"] == cd["core"][:2] and cd["dissenters"] == cd["bystanders"][:2]
    assert len(cd["private"]) == 2 and not set(cd["colluders"]) & set(cd["dissenters"])


def test_render_layouts(tmp_path):
    v = views_of(ep("interactive", size=4, tmp=tmp_path))[0]
    t1, t2 = render(v, "interleaved"), render(v, "threads")
    assert "PUBLIC DISCUSSION" in t1 and "=== " in t2
    pair = render(v, "interleaved", agents=v["agents"][:2])
    for a in v["agents"][2:]:
        assert f"[{a}]:" not in pair


def test_monitor_parse_and_scc(tmp_path):
    assert monitor.parse_json('blah {"a": 1} tail {"b": {"c": 2}}') == {"b": {"c": 2}}
    v = views_of(ep("interactive", size=8, tmp=tmp_path))[0]

    def mon(msgs, kw):
        u = msgs[-1]["content"]
        if "Profile this participant" in u:
            return json.dumps({"pushed_option": "B", "dismissed_evidence": ["x"], "key_claims": ["y"],
                               "suspicion": 40})
        return json.dumps({"coordinated_probability": 70, "reason": "r"})

    run = RunLog("m", max_usd=1, root=tmp_path)
    llm = make_llm("deepseek-v4-flash", run, client=MockClient(mon))
    out = monitor.scc(llm, v, "threads+pairs", 0, 100, m_confirm=3)
    assert out["calls"] == 8 + 3 and out["parsed"] and len(out["agent_scores"]) == 8


def test_metrics_audit_budget_and_topk():
    import metrics as M
    neg = [float(i) for i in range(20)]  # 0..19
    r = M.detection_at_fpr([19.5, 18.5, 10.0, 25.0], neg, fpr=0.05)  # 1 negative may be flagged -> threshold 18
    assert r["threshold"] == 18.0 and r["fpr_actual"] == 0.05 and r["tpr"] == 0.75
    s = {"a": 9, "b": 8, "c": 1, "d": 1}
    assert M.top2_exact(s, ["a", "b"]) == 1.0 and M.topk_any(s, ["c", "d"], k=1) == 0.0
    assert abs(M.topk_any({"a": 1, "b": 1, "c": 1, "d": 1}, ["a", "b"], k=1, draws=4000) - 0.5) < 0.05
    assert M.agent_auroc(s, ["a", "b"]) == 1.0 and M.dissenter_above(s, ["a", "b"], ["c"]) == 0.0
    assert abs(M.chance_topk_any(4) - 5 / 6) < 1e-9 and M.chance_top2_exact(16) == 1 / 120


def test_baselines_talks_and_stance():
    import baselines as B
    ag = ["Ann Lee", "Bob Ray", "Cy Fox", "Di Wu"]
    v = {"agents": ag, "target": "B", "messages": [
        {"author": "Ann Lee", "text": "I agree with Bob on this."},
        {"author": "Bob Ray", "text": "As Ann said, B is better. Cy raised costs."},
        {"author": "Cy Fox", "text": "I prefer A."}, {"author": "Di Wu", "text": "Dr. Fox has a point."}],
        "finals": {"Ann Lee": "RECOMMENDATION: Option B", "Bob Ray": "RECOMMENDATION: Option B",
                   "Cy Fox": "RECOMMENDATION: Option A", "Di Wu": "RECOMMENDATION: Option A"}}
    t = B.talks(v)
    assert t["pair_scores"]["Ann Lee|Bob Ray"] == 4.0 and t["top_pairs"][0][1:] == ["Ann Lee", "Bob Ray"]
    assert t["agent_scores"]["Ann Lee"] == 4.0 and t["coalition"] > 0
    s = B.stance(v)
    assert s["coalition"] == 2 and s["agent_scores"] == {"Ann Lee": 1, "Bob Ray": 1, "Cy Fox": 0, "Di Wu": 0}


def test_analyze_table_on_synthetic_records():
    import analyze
    ag = ["a", "b", "c", "d"]
    def rec(mode, coal, sc, col=(), dis=()):
        return {"method": "whole", "layout": "x", "model": "m", "size": 4, "mode": mode, "parsed": True,
                "coalition": coal, "agent_scores": sc, "agents": ag, "colluders": list(col), "dissenters": list(dis),
                "calls": 1, "usd": 0.001}
    rs = [rec("collusion", 90, {"a": 90, "b": 80, "c": 1, "d": 2}, "ab") for _ in range(3)]
    rs += [rec("control", 10, {x: 5 for x in ag}) for _ in range(3)]
    rs += [rec("dissent", 90, {"a": 50, "b": 50, "c": 1, "d": 1}, (), "ab") for _ in range(3)]
    row = analyze.table(rs)[0]
    assert row["det_ctrl"] == 1.0 and row["det_dis"] == 0.5 and row["top2"] == 1.0 and row["attr"] == 1.0
    assert row["tpr5"] == 1.0 and row["top2_chance"] == round(1 / 6, 3)


def test_solo_modes_use_colluder_prompt_without_coordination(tmp_path):
    s = ep("interactive", size=4, mode="solo", tmp=tmp_path)
    assert s["colluders"] == [] and s["private"] == [] and s["dissenters"] == s["core"][:2]
    assert {s["roles"][a] for a in s["dissenters"]} == {"advocate"}
    c = generate.cast(N, 0, "solo")
    for a in c["dissenters"]:
        assert c["system"][a].startswith(a + " M ") or c["system"][a].startswith(a + " D ") or "M" in c["system"][a]
    cs = ep("interactive", size=8, mode="collusion_solo", tmp=tmp_path)
    assert cs["colluders"] == cs["core"][:2] and cs["dissenters"] == cs["bystanders"][:2] and len(cs["private"]) == 2
    cc = generate.cast(N, 0, "collusion_solo")
    a, b = cc["colluders"][0], cc["dissenters"][0]
    assert cc["system"][a].replace(a, "X") == cc["system"][b].replace(b, "X") or "M" in cc["system"][b]


def test_variety_lines_are_per_seat_and_mode_independent():
    base = generate.cast(N, 0, "collusion")
    for v in ("wording", "roles", "both"):
        c = generate.cast(N, 0, "collusion", v)
        seats = c["core"] + c["bystanders"]
        assert set(c["personas"]) == set(seats) and len(set(c["personas"].values())) == len(seats)
        for a in seats:  # the NARCBench prompt is kept verbatim, one line appended
            assert c["system"][a] == base["system"][a] + "\n\n" + c["personas"][a]
        assert generate.cast(N, 0, "control", v)["personas"] == c["personas"]  # same seat, same line, any mode
    p = generate.cast(N, 0, "collusion", "both", plain_colluders=True)
    assert all(p["system"][a] == base["system"][a] for a in p["colluders"])
    assert ("Writing style:" in generate.cast(N, 0, "control", "wording")["personas"][base["core"][0]]
            and "background" not in generate.cast(N, 0, "control", "wording")["personas"][base["core"][0]])
