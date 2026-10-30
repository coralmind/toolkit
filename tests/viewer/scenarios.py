"""Named results trees for the viewer's page tests, built with the record builders in ``_results``.

``write_fixtures.py`` runs each through the ingest into ``viewer/tests/fixtures/generated``;
the vitest page tests render from that output. Each scenario writes into ``data``, the data
repository the ingest resolves paths against, with its results under ``data/results``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from _results import (
    OTHER,
    SUT,
    TARGET,
    assessment,
    case,
    judge,
    transcript,
    write_jsonl,
    write_persona,
    write_run,
)

KL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kl"


@dataclass
class Scenario:
    build: Callable[[Path], None]
    """Writes the tree under the data repository it is given."""
    after: Callable[[Path], None] | None = None
    """Changes the tree after the ingest, for tests of files that changed since."""
    kl_path: Callable[[Path], Path] | None = None
    """Another Knowledge Library to load, built under the data repository."""
    records_use_kl_path: bool = False
    """Pin the records to that library, rather than to the fixture library."""


SCENARIOS: dict[str, Scenario] = {}


def scenario(
    name: str,
    *,
    after: Callable[[Path], None] | None = None,
    kl_path: Callable[[Path], Path] | None = None,
    records_use_kl_path: bool = False,
) -> Callable[[Callable[[Path], None]], Callable[[Path], None]]:
    def register(build: Callable[[Path], None]) -> Callable[[Path], None]:
        SCENARIOS[name] = Scenario(build, after, kl_path, records_use_kl_path)
        return build

    return register


def results(data: Path) -> Path:
    return data / "results"


def _call(
    seq: int, role: str, stamp: str, prompt: str, test_case_id: str = "aimhef-0000000a"
) -> dict:
    return {
        "type": "llm_call",
        "seq": str(seq),
        "test_case_id": test_case_id,
        "role": role,
        "timestamp": stamp,
        "prompt": prompt,
        "response": f"response {seq}",
        "prompt_tokens": "100",
        "completion_tokens": "10",
        "latency_ms": "1500",
    }


# ---------------------------------------------------------------------------
# Content, failures and the case column
# ---------------------------------------------------------------------------


@scenario("escaped")
def _escaped(data: Path) -> None:
    content = transcript("aimhef-0000000a")
    content["turns"][0]["content"] = "I feel <terrible> today </script><script>alert(1)</script>"
    number = transcript("aimhef-0000000a", 2)
    number["turns"][0]["turn_number"] = "<script>alert(1)</script>"
    write_run(results(data), "run-a", cases=[case("aimhef-0000000a")], sims=[content, number])


@scenario("sim-failure")
def _sim_failure(data: Path) -> None:
    rec = transcript(
        "aimhef-0000000a", sim_failure={"reason": "timeout", "detail": "took too long"}
    )
    write_run(results(data), "run-a", cases=[case("aimhef-0000000a")], sims=[rec])


@scenario("person-echo-exit")
def _person_echo_exit(data: Path) -> None:
    rec = transcript("aimhef-0000000a", exit_reason="person_echo")
    write_run(results(data), "run-a", cases=[case("aimhef-0000000a")], sims=[rec])


@scenario("copied-turn")
def _copied_turn(data: Path) -> None:
    rec = transcript("aimhef-0000000a", n_turns=3)
    reply = "You don't have to rush. Just aim for **small, ordinary steps** and go gently."
    rec["turns"][1]["content"] = reply
    rec["turns"][4]["content"] = reply.replace("**", "")
    write_run(results(data), "run-a", cases=[case("aimhef-0000000a")], sims=[rec])


@scenario("malformed")
def _malformed(data: Path) -> None:
    odd = transcript("aimhef-0000000a", sim_failure={"reason": "x"})
    odd["sim_failure"] = {"reason": 429, "detail": "<b>x</b>"}
    empty = transcript("aimhef-0000000a", 2)
    empty["turns"][0]["content"] = None
    write_run(results(data), "run-a", cases=[case("aimhef-0000000a")], sims=[odd, empty])


@scenario("empty-turns")
def _empty_turns(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    rec["turns"] = []
    write_run(results(data), "run-a", cases=[case("aimhef-0000000a")], sims=[rec])


@scenario("driver")
def _driver(data: Path) -> None:
    run = write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a")],
    )
    write_jsonl(
        run / "batch-001" / "run-a-batch-001.trace.jsonl",
        [
            {
                "type": "llm_call",
                "role": "person_response_generator",
                "test_case_id": "aimhef-0000000a",
                "system": "You are roleplaying as this person.",
                "prompt": "stay in character",
            }
        ],
    )


@scenario("sent-markers")
def _sent_markers(data: Path) -> None:
    personas = data / "personas"
    write_persona(personas, elevated=("kl-900601",))
    write_run(
        results(data),
        "run-a",
        personas=personas,
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a")],
    )


@scenario("unreadable-and-persona")
def _unreadable_and_persona(data: Path) -> None:
    personas = data / "personas"
    write_persona(personas)
    write_run(
        results(data),
        "run-a",
        personas=personas,
        cases=[case("aimhef-0000000b", persona_ref="p-gone@1")],
        sims=[transcript("aimhef-0000000b"), '{"transcript_id": "x", "turns": ['],
    )


@scenario("scripted-l1")
def _scripted_l1(data: Path) -> None:
    rec = transcript("aimhef-0000000a", status=None)
    write_run(
        results(data),
        "run-a",
        level=1,
        cases=[case("aimhef-0000000a", level=1)],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()])],
    )


@scenario("scripted-l2")
def _scripted_l2(data: Path) -> None:
    rec = transcript("aimhef-0000000a", status=None)
    write_run(
        results(data),
        "run-a",
        level=2,
        cases=[case("aimhef-0000000a", level=2)],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()])],
    )


# ---------------------------------------------------------------------------
# Lists and the Suite page
# ---------------------------------------------------------------------------


@scenario("empty")
def _empty(data: Path) -> None:
    results(data).mkdir(parents=True)


@scenario("lists")
def _lists(data: Path) -> None:
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a")],
    )
    write_run(results(data), "run-b", cases=None, sims=[transcript("aimhef-0000000b")])
    unused = data / "test_case_records" / "unused.jsonl"
    unused.parent.mkdir(parents=True, exist_ok=True)
    unused.write_text(json.dumps(case("aimhef-0000000c")) + "\n")


@scenario("unjudged")
def _unjudged(data: Path) -> None:
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a")],
    )


@scenario("invalid-only")
def _invalid_only(data: Path) -> None:
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a", status="invalid")],
    )


@scenario("judged-one")
def _judged_one(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()])],
    )


@scenario("judged-six")
def _judged_six(data: Path) -> None:
    records = [transcript("aimhef-0000000a", n) for n in range(1, 7)]
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=records,
        judges=[
            judge(
                r,
                assessments=[
                    assessment(support="supported" if r["sample"] <= 3 else "not_supported")
                ],
            )
            for r in records
        ],
    )


@scenario("fact-modes")
def _fact_modes(data: Path) -> None:
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a")],
    )
    write_run(
        results(data),
        "run-b",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a")],
        fact_modes={"kl-900601": "stated"},
    )


@scenario("pairs")
def _pairs(data: Path) -> None:
    base_a, cf_a = transcript("aimhef-0000000a"), transcript("aimhef-000000cf")
    base_b = transcript("aimhef-0000000b")
    write_run(
        results(data),
        "run-a",
        cases=[
            case("aimhef-0000000a"),
            case("aimhef-000000cf", counterpart_of="aimhef-0000000a"),
            case("aimhef-0000000b"),
            case("aimhef-00000bcf", counterpart_of="aimhef-0000000b"),
        ],
        sims=[base_a, cf_a, base_b, transcript("aimhef-00000bcf", status="not_tested")],
        judges=[
            judge(base_a, assessments=[assessment()]),
            judge(cf_a, assessments=[assessment(support="not_supported")]),
            judge(base_b, assessments=[assessment()]),
        ],
    )


@scenario("two-systems")
def _two_systems(data: Path) -> None:
    records, judges = [], []
    for sample in (1, 2):
        rec = transcript("aimhef-0000000a", sample)
        records.append(rec)
        support = "supported" if sample == 1 else "not_supported"
        judges.append(judge(rec, assessments=[assessment(support=support)]))
    for sample in (1, 2):
        rec = transcript(
            "aimhef-0000000a",
            sample,
            sut_key="other-model",
            status="invalid" if sample == 2 else "valid",
        )
        records.append(rec)
        if sample == 1:
            judges.append(judge(rec, assessments=[assessment(support="not_supported")]))
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=records,
        judges=judges,
        suts=(SUT, "other/model"),
    )


@scenario("elicitation")
def _elicitation(data: Path) -> None:
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a"), case("aimhef-0000000b"), case("aimhef-0000000c")],
        sims=[
            transcript("aimhef-0000000a", status="invalid"),
            transcript("aimhef-0000000b", status=None),
            transcript("aimhef-0000000c", status="not_tested"),
        ],
    )


@scenario("onset")
def _onset(data: Path) -> None:
    rec = transcript("aimhef-0000000a", n_turns=6)
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment(turn=4)])],
    )


@scenario("gate-not-met")
def _gate_not_met(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment(gate_status="not_met")])],
    )


@scenario("gate-unknown")
def _gate_unknown(data: Path) -> None:
    fired, unknown = transcript("aimhef-0000000a", 1), transcript("aimhef-0000000a", 2)
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[fired, unknown],
        judges=[
            judge(fired, assessments=[assessment()]),
            judge(unknown, assessments=[assessment(gate_status="unknown")]),
        ],
    )


def _neighbor_kl(data: Path) -> Path:
    kl = data / "kl"
    shutil.copytree(KL_FIXTURE, kl)
    pathway = kl / "entries" / "pathways" / "kl-900102.yaml"
    pathway.write_text(
        pathway.read_text(encoding="utf-8")
        + "distinguish_from:\n- neighbor: kl-900101\n  boundary: Here the person is distressed.\n",
        encoding="utf-8",
    )
    return kl


@scenario("neighbor", kl_path=_neighbor_kl, records_use_kl_path=True)
def _neighbor(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a", pathway_id=TARGET)],
        sims=[rec],
        judges=[
            judge(
                rec,
                assessments=[
                    assessment(TARGET, support="not_supported"),
                    assessment(OTHER, gate_status="no_gates"),
                ],
            )
        ],
    )


@scenario("multi-system-unreadable")
def _multi_system_unreadable(data: Path) -> None:
    other = transcript("aimhef-0000000a", sut_key="other-model")
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a"), other, '{"transcript_id": "x", "turns": ['],
        suts=(SUT, "other/model"),
    )


def _bad_kl(data: Path) -> Path:
    bad = data / "bad-kl" / "entries" / "pathways"
    bad.mkdir(parents=True)
    (bad / "kl-999999.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    return data / "bad-kl"


@scenario("no-library", kl_path=_bad_kl)
def _no_library(data: Path) -> None:
    _judged_one(data)


# ---------------------------------------------------------------------------
# Run mode and the transcript view's lists
# ---------------------------------------------------------------------------


@scenario("run-mode")
def _run_mode(data: Path) -> None:
    good = transcript("aimhef-0000000a")
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a"), case("aimhef-0000000b")],
        sims=[good, transcript("aimhef-0000000b", status="invalid")],
        judges=[judge(good, assessments=[assessment()])],
    )


@scenario("long-run")
def _long_run(data: Path) -> None:
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a", n) for n in (1, 2, 3)],
    )


@scenario("prev-next")
def _prev_next(data: Path) -> None:
    records = [transcript(f"aimhef-0000000{c}", s) for c in "ab" for s in (1, 2)]
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a"), case("aimhef-0000000b")],
        sims=records,
        judges=[
            judge(
                r,
                assessments=[
                    assessment(
                        support="supported"
                        if r["sample"] == 1 and r["test_case_id"].endswith("b")
                        else "not_supported"
                    )
                ],
            )
            for r in records
        ],
    )


@scenario("strip")
def _strip(data: Path) -> None:
    short = transcript("aimhef-0000000a", n_turns=10)
    long = transcript("aimhef-0000000a", 2, n_turns=80)
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[short, long],
        judges=[
            judge(short, assessments=[assessment(turn=4)]),
            judge(long, assessments=[assessment(turn=63)]),
        ],
    )


@scenario("citations")
def _citations(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    record = judge(rec, assessments=[assessment()])
    record["coralmind_run"]["stage1"]["observations"][0]["evidence"] = [
        {"turn": 2, "role": "assistant", "quote": "reply 2", "verified": True},
        {"turn": 2, "role": "assistant", "quote": "never said", "verified": True},
        {"turn": 1, "role": "user", "quote": "hello 1", "verified": False},
    ]
    write_run(results(data), "run-a", cases=[case("aimhef-0000000a")], sims=[rec], judges=[record])


@scenario("steps")
def _steps(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    record = judge(rec, assessments=[assessment(turn=1)])
    record["coralmind_run"]["stage2"]["assessments"][0]["instances"][0]["steps"] = [
        {
            "order": 1,
            "turn": 1,
            "behavior_id": "kl-900001",
            "behavior_did": "AiBehavior-901",
            "role": "originator",
            "contribution": "The AI answers: 'reply 1'",
        },
        {
            "order": 2,
            "turn": 1,
            "behavior_id": "kl-900501",
            "behavior_did": "UserBehavior-901",
            "role": "initiator",
            "contribution": "The person opens: 'hello 1 and more words here'",
        },
    ]
    record["coralmind_run"]["stage1"]["observations"] = [
        {
            "behavior_id": "kl-900001",
            "kind": "ai_behavior",
            "turns": [1],
            "evidence": [{"turn": 1, "role": "assistant", "quote": "reply 1", "verified": True}],
        },
        {
            "behavior_id": "kl-900501",
            "kind": "user_behavior",
            "turns": [1],
            "evidence": [
                {
                    "turn": 1,
                    "role": "user",
                    "quote": "So: hello 1 and more words here, I said",
                    "verified": True,
                }
            ],
        },
    ]
    write_run(results(data), "run-a", cases=[case("aimhef-0000000a")], sims=[rec], judges=[record])


@scenario("condition")
def _condition(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    record = judge(rec, assessments=[assessment(gate_status="not_met")])
    gate = record["coralmind_run"]["stage2"]["assessments"][0]["gates"][0]
    gate.update(
        recorded="does_not_hold",
        inferred="holds",
        conflict=True,
        basis="test_case",
        cues=[{"turn": 1, "speaker": "user", "quote": "hello 1", "reading": "supports"}],
    )
    write_run(results(data), "run-a", cases=[case("aimhef-0000000a")], sims=[rec], judges=[record])


def _swap_lines(data: Path) -> None:
    a, b = transcript("aimhef-0000000a"), transcript("aimhef-0000000b")
    a["turns"][0]["content"] = "ALPHA"
    b["turns"][0]["content"] = "BRAVO"
    write_jsonl(results(data) / "run-a" / "batch-001" / "run-a-sim.jsonl", [b, a])


@scenario("rewritten", after=_swap_lines)
def _rewritten(data: Path) -> None:
    a, b = transcript("aimhef-0000000a"), transcript("aimhef-0000000b")
    a["turns"][0]["content"] = "ALPHA"
    b["turns"][0]["content"] = "BRAVO"
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a"), case("aimhef-0000000b")],
        sims=[a, b],
    )


@scenario("judge-prompts")
def _judge_prompts(data: Path) -> None:
    first = transcript("aimhef-0000000a", 1)
    second = transcript("aimhef-0000000a", 2)
    run = write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[first, second],
        judges=[
            judge(
                first,
                assessments=[assessment()],
                window=("2026-10-20T20:00:00+00:00", "2026-10-20T20:01:00+00:00"),
            ),
            judge(
                second,
                assessments=[assessment()],
                window=("2026-10-20T20:02:00+00:00", "2026-10-20T20:03:00+00:00"),
            ),
        ],
    )
    write_jsonl(
        run / "batch-001" / "run-a-judge.trace.jsonl",
        [
            {"type": "event", "seq": "1", "role": "action_load"},
            _call(2, "coralmind_s1_ai_behavior-g01", "2026-10-20T20:00:10+00:00", "first s1"),
            _call(3, "coralmind_s2_Pathway-902", "2026-10-20T20:00:40+00:00", "first s2"),
            _call(4, "coralmind_s1_ai_behavior-g01", "2026-10-20T20:02:10+00:00", "second s1"),
        ],
    )


def _rewrite_trace(data: Path) -> None:
    trace = results(data) / "run-a" / "batch-001" / "run-a-judge.trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    write_jsonl(
        trace,
        [_call(7, "coralmind_s1_ai_behavior-g01", "x", "other case", "aimhef-0000000z"), *rows],
    )


@scenario("judge-prompts-rewritten", after=_rewrite_trace)
def _judge_prompts_rewritten(data: Path) -> None:
    _judge_prompts(data)


@scenario("judge-setup-shown")
def _judge_setup_shown(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    run = write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[
            judge(
                rec,
                assessments=[assessment()],
                prompt_hash="old",
                window=("2026-10-20T20:00:00+00:00", "2026-10-20T20:01:00+00:00"),
            )
        ],
    )
    write_jsonl(
        run / "batch-001" / "run-a-judge.trace.jsonl",
        [_call(2, "coralmind_s2_Pathway-902", "2026-10-20T20:00:10+00:00", "old prompt")],
    )


@scenario("partial-judge")
def _partial_judge(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a"), case("aimhef-0000000b", pathway_id=OTHER)],
        sims=[rec],
        judges=[
            judge(
                rec,
                call_failures=1,
                stage2_calls=[{"pathway_ids": [TARGET], "status": "judge_error"}],
                stage1_calls=[
                    {
                        "group_id": "g2",
                        "kind": "ai_behavior",
                        "status": "judge_error",
                        "behavior_ids": ["kl-900001"],
                    }
                ],
                skipped=[OTHER],
            )
        ],
    )


@scenario("other-library")
def _other_library(data: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results(data),
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()], digest="sha256:other")],
    )
