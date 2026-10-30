"""Reproducibility report across repeated CoralMind judge runs of the same transcript.

Given several results JSONL files (each from one `corallab judge` run on the same
input), the script answers two questions:

1. **Were the inputs identical?** Prompt-template hash, ``kl``, and
   transcript hash from ``run_meta``, and — from each run's ``*.trace.jsonl`` — the
   SHA-256 of every Stage-1 prompt actually sent, per definition group. If the
   per-group prompt hashes match across runs, any output difference originates
   in the model's sampling, not in our pipeline.
2. **How stable were the outputs?** Per-run present sets and their pairwise
   Jaccard; per-behavior agreement (present in k of n runs); turn-anchor
   agreement for behaviors present in >= 2 runs (Jaccard of turn sets, split by
   detection window); negatives agreement; Stage-2 pathway / instance agreement;
   and how much of the observed variance is already in the RAW model answers
   (before quote verification and speaker anchoring) versus introduced by
   normalization.

Usage::

    pixi run python scripts/coralmind_repro.py results/<a>/<a>.jsonl results/<b>/<b>.jsonl ...
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from corallab.judging.judges.kl_two_stage import parse_json_lenient


def _load(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    recs = [r for r in rows if r.get("coralmind_run")]
    if len(recs) != 1:
        raise SystemExit(f"{path}: expected exactly one judged record, found {len(recs)}")
    rec: dict[str, Any] = recs[0]
    return rec


def _trace_prompts(results_path: Path) -> dict[str, dict[str, Any]]:
    """role → {prompt_sha, response_sha, prompt_tokens, completion_tokens, latency_ms}."""
    trace = results_path.with_suffix(".trace.jsonl")
    out: dict[str, dict[str, Any]] = {}
    if not trace.exists():
        return out
    for line in trace.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "llm_call" or not str(row.get("role", "")).startswith("coralmind_"):
            continue
        role = row["role"]
        # Keep the LAST attempt per role (retries reuse the role).
        out[role] = {
            "prompt_sha": hashlib.sha256(str(row.get("prompt", "")).encode()).hexdigest(),
            "response_sha": hashlib.sha256(str(row.get("response", "")).encode()).hexdigest(),
            "response": row.get("response", ""),
            "prompt_tokens": row.get("prompt_tokens"),
            "completion_tokens": row.get("completion_tokens"),
            "latency_ms": row.get("latency_ms"),
        }
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _pairwise_jaccard(sets: list[set]) -> list[float]:
    return [_jaccard(a, b) for a, b in itertools.combinations(sets, 2)]


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return f"mean {statistics.mean(xs):.2f} min {min(xs):.2f} max {max(xs):.2f}"


def _raw_present(traces: dict[str, dict[str, Any]]) -> dict[str, set[int]]:
    """Behavior ids → turns the model itself reported (definition-major answers), before
    any normalization."""
    raw: dict[str, set[int]] = defaultdict(set)
    for role, info in traces.items():
        if not role.startswith("coralmind_s1_"):
            continue
        parsed = parse_json_lenient(info["response"] or "")
        if not parsed or not isinstance(parsed.get("behaviors"), list):
            continue
        for it in parsed["behaviors"]:
            if not isinstance(it, dict) or it.get("present") is False:
                continue
            bid = str(it.get("behavior_id") or "")
            turns = it.get("turns") or [
                e.get("turn") for e in it.get("evidence") or [] if isinstance(e, dict)
            ]
            for t in turns or []:
                with contextlib.suppress(TypeError, ValueError):
                    raw[bid].add(int(t))
            raw.setdefault(bid, set())
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="Results JSONL files of repeated runs")
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.results]
    recs = [_load(p) for p in paths]
    runs = [r["coralmind_run"] for r in recs]
    metas = [r["run_meta"] for r in runs]
    n = len(runs)
    out: list[str] = []
    p = out.append

    p(f"# Reproducibility across {n} runs — {recs[0].get('test_case_display_name')}")
    p("")
    # ---------------- 1. identical inputs? ----------------
    p("## 1. Were the inputs identical?")
    for key in (
        "prompt_version",
        "prompt_hash",
        "kl",
        "transcript_hash",
        "model",
    ):
        vals = {str(m.get(key)) for m in metas}
        p(
            f"- {key}: {'IDENTICAL' if len(vals) == 1 else 'DIFFERENT ' + str(vals)}"
            + (f" ({next(iter(vals))[:16]}…)" if len(vals) == 1 and key.endswith("hash") else "")
        )
    cfgs = {json.dumps(m.get("config"), sort_keys=True) for m in metas}
    p(f"- options: {'IDENTICAL' if len(cfgs) == 1 else 'DIFFERENT'}")
    traces = [_trace_prompts(pth) for pth in paths]
    roles = sorted(set().union(*[set(t) for t in traces]))
    s1_roles = [r for r in roles if r.startswith("coralmind_s1_")]
    same_prompt = 0
    diff_prompt: list[str] = []
    for role in s1_roles:
        shas = {t[role]["prompt_sha"] for t in traces if role in t}
        if len(shas) == 1 and all(role in t for t in traces):
            same_prompt += 1
        else:
            diff_prompt.append(role)
    resp_same = sum(
        1
        for role in s1_roles
        if all(role in t for t in traces) and len({t[role]["response_sha"] for t in traces}) == 1
    )
    p(
        f"- Stage-1 prompts sent (from traces): {same_prompt}/{len(s1_roles)} groups "
        "byte-identical across runs" + (f"; differing: {diff_prompt}" if diff_prompt else "")
    )
    p(f"- Stage-1 responses byte-identical across runs: {resp_same}/{len(s1_roles)} groups")
    if traces and all(traces):
        lat = [sum((t[r]["latency_ms"] or 0) for r in t) / 1000 for t in traces]
        ctoks = [sum((t[r]["completion_tokens"] or 0) for r in t) for t in traces]
        p(f"- per-run total latency (s): {[round(x) for x in lat]}; completion tokens: {ctoks}")
    p("")

    # ---------------- 2. Stage 1 ----------------
    p("## 2. Stage 1 — presence")
    present_sets = [{o["behavior_id"] for o in r["stage1"]["observations"]} for r in runs]
    did = {o["behavior_id"]: o["behavior_did"] for r in runs for o in r["stage1"]["observations"]}
    window = {
        o["behavior_id"]: o.get("detection_window") or "?"
        for r in runs
        for o in r["stage1"]["observations"]
    }
    p(f"- behaviors present per run: {[len(s) for s in present_sets]}")
    p(f"- pairwise Jaccard of present sets: {_fmt(_pairwise_jaccard(present_sets))}")
    count = Counter(b for s in present_sets for b in s)
    dist = Counter(count.values())
    p(
        "- agreement: "
        + ", ".join(f"{k}/{n} runs: {dist.get(k, 0)} behaviors" for k in range(n, 0, -1))
    )
    for win in sorted({window[b] for b in count}):
        ids = [b for b in count if window[b] == win]
        sets_w = [{b for b in s if window.get(b) == win} for s in present_sets]
        p(
            f"  - {win}: {len(ids)} behaviors ever present; "
            f"Jaccard {_fmt(_pairwise_jaccard(sets_w))}; "
            f"in all runs: {sum(1 for b in ids if count[b] == n)}"
        )
    unstable = sorted((b for b in count if count[b] < n), key=lambda b: (-count[b], b))
    if unstable:
        p(
            "- present in only some runs: "
            + ", ".join(f"{did[b]}({count[b]}/{n},{window[b]})" for b in unstable)
        )
    neg_sets = [{x["behavior_id"] for x in r["stage1"].get("negatives", [])} for r in runs]
    unassessed_n = [len(r["stage1"].get("unassessed", [])) for r in runs]
    p(f"- negatives per run: {[len(s) for s in neg_sets]}; unassessed per run: {unassessed_n}")
    ig = [r["stage1"].get("integrity", {}) for r in runs]
    qv = [f"{i.get('quotes_verified')}/{i.get('quotes_total')}" for i in ig]
    p(
        f"- integrity per run: quotes_verified/total {qv}; "
        f"turns_dropped_unverified {[i.get('turns_dropped_unverified') for i in ig]}; "
        f"turns_dropped_wrong_speaker {[i.get('turns_dropped_wrong_speaker') for i in ig]}"
    )
    p("")

    p("## 3. Stage 1 — turn anchors (behaviors present in >= 2 runs)")
    turns_by_run = [
        {o["behavior_id"]: set(o["turns"]) for o in r["stage1"]["observations"]} for r in runs
    ]
    anchor_j: dict[str, list[float]] = defaultdict(list)
    exact: dict[str, list[int]] = defaultdict(list)
    for b in count:
        if count[b] < 2:
            continue
        sets_b = [t[b] for t in turns_by_run if b in t]
        js = _pairwise_jaccard(sets_b)
        anchor_j[window[b]].extend(js)
        exact[window[b]].extend(int(j == 1.0) for j in js)
    for win in sorted(anchor_j):
        p(
            f"- {win}: pairwise turn-set Jaccard {_fmt(anchor_j[win])}; identical anchor sets in "
            f"{sum(exact[win])}/{len(exact[win])} pairs"
        )
    rows = []
    for b in sorted(count, key=lambda x: did[x]):
        if count[b] < 2:
            continue
        rows.append(
            f"  - {did[b]} [{window[b]}]: "
            + " | ".join(
                (",".join(map(str, sorted(t[b]))) if b in t else "—") for t in turns_by_run
            )
        )
    p("- anchors per run (turn lists, one column per run):")
    out.extend(rows)
    p("")

    # ---------------- 4. raw vs normalized ----------------
    p("## 4. Where does the variance come from? raw model answers vs after normalization")
    if all(traces):
        raws = [_raw_present(t) for t in traces]
        raw_sets = [set(r) for r in raws]
        p(
            "- RAW present sets (what the model reported before verification): "
            f"sizes {[len(s) for s in raw_sets]}; "
            f"pairwise Jaccard {_fmt(_pairwise_jaccard(raw_sets))}"
        )
        p(
            f"- NORMALIZED present sets: sizes {[len(s) for s in present_sets]}; "
            f"pairwise Jaccard {_fmt(_pairwise_jaccard(present_sets))}"
        )
        raw_count = Counter(b for s in raw_sets for b in s)
        removed = [
            (did.get(b, b), raw_count[b], count.get(b, 0))
            for b in raw_count
            if count.get(b, 0) < raw_count[b]
        ]
        if removed:
            p(
                "- behaviors reported by the model but removed (or partly removed) by "
                "normalization (name, runs reported raw, runs kept): "
                + ", ".join(f"{a}({r}->{k})" for a, r, k in removed)
            )
        raw_anchor_j: list[float] = []
        for b in raw_count:
            if raw_count[b] < 2:
                continue
            sets_b = [r[b] for r in raws if b in r]
            raw_anchor_j.extend(_pairwise_jaccard(sets_b))
        all_norm = [j for js in anchor_j.values() for j in js]
        p(f"- turn-anchor Jaccard, RAW: {_fmt(raw_anchor_j)}; NORMALIZED: {_fmt(all_norm)}")
        p(
            "- reading: if prompts are byte-identical and RAW variance is already at this "
            "level, the instability originates in the model's sampling; normalization only "
            "removes anchors it "
            "cannot verify (compare the two lines above)."
        )
    else:
        p("- traces missing for some runs; raw comparison skipped")
    p("")

    # ---------------- 5. Stage 2 ----------------
    p("## 5. Stage 2 — pathways and instances")
    pw_sets = [
        {a["pathway_id"] for a in r["stage2"]["assessments"] if a.get("instances")} for r in runs
    ]
    p(
        f"- pathways with >=1 instance per run: {[len(s) for s in pw_sets]}; "
        f"pairwise Jaccard {_fmt(_pairwise_jaccard(pw_sets))}"
    )
    inst_sets = [
        {
            (a["pathway_id"], tuple(i["turn_window"]))
            for a in r["stage2"]["assessments"]
            for i in a.get("instances", [])
        }
        for r in runs
    ]
    p(
        f"- instances per run: {[len(s) for s in inst_sets]}; "
        f"pairwise Jaccard on (pathway, window): {_fmt(_pairwise_jaccard(inst_sets))}"
    )
    cnt = Counter(pw for s in pw_sets for pw in s)
    p(
        "- pathway agreement: "
        + ", ".join(f"{k}/{n}: {sum(1 for v in cnt.values() if v == k)}" for k in range(n, 0, -1))
    )
    s2fail = [sum(1 for c in r["stage2"]["calls"] if c["status"] == "judge_error") for r in runs]
    cands = [m["counts"].get("candidates") for m in metas]
    p(f"- Stage-2 failed calls per run: {s2fail}; candidates per run: {cands}")
    p("")
    p("## 6. Cost per run")
    for i, m in enumerate(metas, start=1):
        t = m.get("tokens") or {}
        p(
            f"- run {i}: {m['elapsed_ms'] / 1000:.0f} s, prompt {t.get('prompt')}, "
            f"completion {t.get('completion')}, cached {t.get('cached')}, "
            f"calls {m['counts'].get('stage1_calls')}+{m['counts'].get('stage2_calls')}"
        )

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
