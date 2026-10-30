"""facts.py: every transcript lands in one bucket, and every route gets an outcome."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from _results import (
    OTHER,
    SUT,
    TARGET,
    assessment,
    case,
    fidelity,
    judge,
    transcript,
    write_jsonl,
    write_persona,
    write_run,
)

from coralmind_viewer import facts as F


@pytest.fixture
def personas(tmp_path: Path) -> Path:
    directory = tmp_path / "personas"
    write_persona(directory, elevated=("kl-900601",))
    return directory


def build(root: Path) -> F.Snapshot:
    index = F.Index(root)
    snapshot = index.rebuild()
    assert snapshot is not None, index.status().first_error
    return snapshot


def only_run(snapshot: F.Snapshot) -> F.RunFacts:
    (name,) = snapshot.runs
    return F.run_facts(snapshot, name)


def route_rows(facts: F.RunFacts) -> list[F.Route]:
    return [r for t in facts.transcripts for r in facts.routes(t)]


def only_suite(snapshot: F.Snapshot) -> F.Suite:
    (suite,) = snapshot.suites.values()
    return suite


def test_every_transcript_lands_in_exactly_one_bucket(results: Path, personas: Path) -> None:
    judged = transcript("aimhef-0000000a", 1)
    unjudged = transcript("aimhef-0000000a", 2)
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        personas=personas,
        sims=[
            judged,
            unjudged,
            transcript("aimhef-0000000a", 3, status="invalid"),
            transcript("aimhef-0000000a", 4, status=None),
            transcript("aimhef-0000000a", 5, sim_failure={"reason": "timeout"}),
            transcript("aimhef-0000000a", 6, status="not_tested"),
            '{"transcript_id": "demo:x:aimhef-0000000a:7", "turns": [',
        ],
        judges=[judge(judged, assessments=[assessment()])],
    )
    facts = only_run(build(results))
    buckets = Counter(t.bucket for t in facts.transcripts)
    assert buckets == {
        "judged": 1,
        "not_judged": 1,
        "invalid": 1,
        "unchecked": 1,
        "sim_failure": 1,
        "not_tested": 1,
        "unreadable": 1,
    }
    assert sum(buckets.values()) == len(facts.transcripts) == 7


@pytest.mark.parametrize("level", [1, 2])
def test_scripted_levels_are_judged_without_a_fidelity_check(results: Path, level: int) -> None:
    rec = transcript("aimhef-0000000a", status=None)
    write_run(
        results,
        "run-a",
        level=level,
        cases=[case("aimhef-0000000a", level=level)],
        sims=[rec, transcript("aimhef-0000000a", 2, status=None)],
        judges=[judge(rec, assessments=[assessment()])],
    )
    buckets = [t.bucket for t in only_run(build(results)).transcripts]
    assert buckets == ["judged", "not_judged"]


@pytest.mark.parametrize(
    ("gate_status", "support", "outcome"),
    [
        ("met", "supported", "fires"),
        ("no_gates", "supported", "fires"),
        ("not_met", "supported", "not"),
        ("unknown", "supported", "gate_unknown"),
        ("met", "partially_supported", "partial"),
        ("met", "not_supported", "not"),
    ],
)
def test_fires_needs_support_and_a_gate_that_holds(
    results: Path, gate_status: str, support: str, outcome: str
) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment(support=support, gate_status=gate_status)])],
    )
    facts = only_run(build(results))
    (row,) = route_rows(facts)
    assert row.outcome == outcome


def test_a_partial_judge_run_marks_failed_routes_and_leaves_them_out(results: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
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
    facts = only_run(build(results))
    assert facts.transcripts[0].bucket == "judged"
    rows = {r.route: r for r in route_rows(facts)}
    assert rows[TARGET].outcome == "failed"


def test_judge_records_join_only_their_own_run(results: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()])],
    )
    write_run(results, "run-b", cases=[case("aimhef-0000000a")], sims=[rec])
    snapshot = build(results)
    assert snapshot.judges_by_key.keys() == {("run-a", rec["transcript_id"])}


def test_a_different_test_case_file_is_a_different_suite(results: Path) -> None:
    write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    write_run(
        results, "run-b", cases=[case("aimhef-0000000b")], sims=[transcript("aimhef-0000000b")]
    )
    write_run(results, "run-c", cases=None, sims=[transcript("aimhef-0000000c")])
    snapshot = build(results)
    assert len(snapshot.suites) == 2
    assert snapshot.runs["run-c"].suite_id is None


def test_a_record_judged_with_another_library_is_set_aside(results: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()], digest="sha256:other")],
    )
    facts = only_run(build(results))
    (row,) = route_rows(facts)
    assert row.outcome == "kl_mismatch"


def test_the_latest_record_is_used_and_orphans_are_counted(results: Path) -> None:
    rec = transcript("aimhef-0000000a")
    orphan = transcript("aimhef-0000000a", 9)
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[
            judge(
                rec,
                assessments=[assessment(support="not_supported")],
                evaluated_at="2026-10-20T00:00:00+00:00",
            ),
            judge(rec, assessments=[assessment()], evaluated_at="2026-10-21T00:00:00+00:00"),
            judge(orphan, assessments=[assessment()]),
        ],
    )
    facts = only_run(build(results))
    assert facts.orphans == 1
    assert [r.outcome for r in route_rows(facts)] == ["fires"]


def test_unreadable_judge_lines_are_counted_with_file_and_line(results: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()]), '{"transcript_id": ', '{"x": 1}'],
    )
    facts = only_run(build(results))
    assert [(p.line, p.error) for p in facts.judge_problems] == [
        (2, facts.judge_problems[0].error),
        (3, "missing transcript_id"),
    ]
    assert facts.transcripts[0].bucket == "judged"


def test_an_older_record_shape_stays_readable(results: Path) -> None:
    rec = transcript("aimhef-0000000a", future_field={"x": 1})
    del rec["exit_reason"]
    del rec["test_case_version"]
    write_run(results, "run-a", cases=[case("aimhef-0000000a")], sims=[rec])
    (row,) = only_run(build(results)).transcripts
    assert row.bucket == "not_judged"


def test_a_malformed_sim_failure_or_null_content_does_not_crash(results: Path) -> None:
    broken = transcript("aimhef-0000000a", sim_failure={"reason": "x"})
    broken["sim_failure"] = "boom"
    write_run(results, "run-a", cases=[case("aimhef-0000000a")], sims=[broken])
    (row,) = only_run(build(results)).transcripts
    assert row.bucket == "sim_failure"


def test_an_unloadable_library_is_reported_and_routes_still_count(
    results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()])],
    )
    bad = tmp_path / "bad-kl" / "entries" / "pathways"
    bad.mkdir(parents=True)
    (bad / "kl-999999.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv("CORALMIND_KL_PATH", str(tmp_path / "bad-kl"))
    snapshot = build(results)
    assert snapshot.library is None
    assert snapshot.library_error
    facts = only_run(snapshot)
    assert [r.outcome for r in route_rows(facts)] == ["fires"]


def test_a_missing_persona_is_named(results: Path, personas: Path) -> None:
    write_run(
        results,
        "run-a",
        personas=personas,
        cases=[case("aimhef-0000000a"), case("aimhef-0000000b", persona_ref="p-gone@1")],
        sims=[transcript("aimhef-0000000a"), transcript("aimhef-0000000b")],
    )
    cases = only_suite(build(results)).cases
    assert cases["aimhef-0000000b"].persona_error == "persona record not found: p-gone@1"
    assert cases["aimhef-0000000a"].persona is not None


def test_an_unchanged_file_is_not_read_again_and_an_appended_line_appears(results: Path) -> None:
    rec = transcript("aimhef-0000000a")
    run = write_run(results, "run-a", cases=[case("aimhef-0000000a")], sims=[rec])
    index = F.Index(results)
    first = index.rebuild()
    assert first is not None
    assert index.file_reads == 1
    index.rebuild()
    assert index.file_reads == 1
    sim = run / "batch-001" / "run-a-sim.jsonl"
    write_jsonl(sim, [rec, transcript("aimhef-0000000a", 2)])
    second = index.rebuild()
    assert second is not None
    assert len(second.sims) == 2
    assert index.file_reads == 2


def test_a_failed_rebuild_keeps_the_snapshot_and_says_it_is_stale(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    index = F.Index(results)
    good = index.rebuild()
    sim = run / "batch-001" / "run-a-sim.jsonl"
    write_jsonl(sim, [transcript("aimhef-0000000a"), transcript("aimhef-0000000a", 2)])
    real = F.read_sim_file

    def fail(*args: object) -> F.FileRows:
        raise OSError("disk went away")

    monkeypatch.setattr(F, "read_sim_file", fail)
    assert index.rebuild() is good
    state = index.status()
    assert state.snapshot is good
    assert state.stale_error is not None
    assert "disk went away" in state.stale_error
    assert str(sim) in state.stale_error
    monkeypatch.setattr(F, "read_sim_file", real)
    fresh = index.rebuild()
    assert fresh is not good
    assert index.status().stale_error is None


def test_a_failed_first_build_names_the_file_and_retries(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    real = F.read_sim_file

    def fail(*args: object) -> F.FileRows:
        raise OSError("permission denied")

    monkeypatch.setattr(F, "read_sim_file", fail)
    index = F.Index(results)
    assert index.rebuild() is None
    state = index.status()
    assert state.first_error is not None
    assert "permission denied" in state.first_error
    monkeypatch.setattr(F, "read_sim_file", real)
    assert index.rebuild() is not None
    assert index.status().first_error is None


def test_a_neighbor_that_fired_instead_is_counted(
    results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kl = tmp_path / "kl"
    shutil.copytree(Path(__file__).resolve().parents[1] / "fixtures" / "kl", kl)
    pathway = kl / "entries" / "pathways" / "kl-900102.yaml"
    pathway.write_text(
        pathway.read_text(encoding="utf-8")
        + "distinguish_from:\n- neighbor: kl-900101\n  boundary: Here the person is distressed.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CORALMIND_KL_PATH", str(kl))
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
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
                digest=None,
            )
        ],
    )
    snapshot = build(results)
    assert snapshot.library is not None, snapshot.library_error
    facts = only_run(snapshot)
    (row,) = (r for r in route_rows(facts) if r.route == TARGET)
    assert row.outcome == "not"
    assert row.neighbors_fired == (OTHER,)


def test_any_first_build_error_is_reported_not_a_loading_loop(results: Path) -> None:
    run = write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    (run / "inputs" / "aaa.yaml").write_bytes(b"\xff\xfe not utf-8")
    index = F.Index(results)
    assert index.rebuild() is None
    state = index.status()
    assert state.first_error is not None
    assert "UnicodeDecodeError" in state.first_error


def test_a_lone_unreadable_line_goes_to_the_runs_only_system(results: Path) -> None:
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a"), '{"transcript_id": "x", "turns": ['],
    )
    facts = only_run(build(results))
    assert not any(t.sut == F.UNKNOWN_SUT for t in facts.transcripts)
    assert Counter(t.sut for t in facts.transcripts) == {SUT: 2}


def test_unreadable_lines_from_a_multi_system_run_are_in_no_column(results: Path) -> None:
    other = transcript("aimhef-0000000a", sut_key="other-model")
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a"), other, '{"transcript_id": "x", "turns": ['],
        suts=(SUT, "other/model"),
    )
    facts = only_run(build(results))
    assert {t.sut for t in facts.transcripts} - {F.UNKNOWN_SUT} == {SUT, "other/model"}
    assert sum(t.sut == F.UNKNOWN_SUT for t in facts.transcripts) == 1


def test_manifest_labels_keep_one_model_at_two_settings_apart(results: Path) -> None:
    low = transcript("aimhef-0000000a", sut_key="mini-low")
    high = transcript("aimhef-0000000a", sut_key="mini-high")
    run = write_run(results, "run-a", cases=[case("aimhef-0000000a")], sims=[low, high])
    manifest = run / "inputs" / "suite.yaml"
    manifest.write_text(
        manifest.read_text().replace(
            f"- model_id: {SUT}",
            f"- model_id: {SUT}\n  label: mini-low\n- model_id: {SUT}\n  label: mini-high",
        )
    )
    assert sorted({t.sut for t in only_run(build(results)).transcripts}) == [
        "mini-high",
        "mini-low",
    ]
    (run / "suite.index.json").write_text(
        json.dumps(
            {
                "models": [
                    {"model_id": SUT, "output_key": "mini-low"},
                    {"model_id": SUT, "output_key": "mini-high"},
                ]
            }
        )
    )
    assert sorted({t.sut for t in only_run(build(results)).transcripts}) == [
        "mini-high",
        "mini-low",
    ]


def test_judge_setups_count_transcripts_per_run(results: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()])],
    )
    write_run(
        results,
        "run-b",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()])],
    )
    (only,) = only_suite(build(results)).passes
    assert only.transcripts == 2


def test_a_judge_record_that_never_ran_is_no_judge_setup(results: Path) -> None:
    rec = transcript("aimhef-0000000a")
    crashed = judge(rec)
    crashed["coralmind_run"] = None
    crashed["status"] = "judge_error"
    write_run(results, "run-a", cases=[case("aimhef-0000000a")], sims=[rec], judges=[crashed])
    snapshot = build(results)
    assert only_suite(snapshot).passes == ()
    assert [t.bucket for t in only_run(snapshot).transcripts] == ["not_judged"]


def test_a_persona_ref_cannot_name_a_path(tmp_path: Path) -> None:
    (tmp_path / "secret.json").write_text("{}")
    personas = tmp_path / "personas"
    personas.mkdir()
    assert F.load_persona(personas, "../secret@1") == "persona record not found: ../secret@1"


def test_the_checker_model_falls_back_and_survives_nulls() -> None:
    manifest = {
        "simulation": {
            "elicitation_fidelity": None,
            "person_response_model": {"model_id": "person-model"},
        }
    }
    assert F.checker_model(manifest) == "person-model"
    assert (
        F.checker_model({"simulation": {"elicitation_fidelity": {"model": {"model_id": "c"}}}})
        == "c"
    )
    assert F.checker_model({}) == ""


def test_a_person_turn_that_copies_the_ai_makes_the_transcript_invalid(results: Path) -> None:
    rec = transcript("aimhef-0000000a", n_turns=3)
    reply = "You don't have to rush. Just aim for small, ordinary steps and go gently."
    rec["turns"][1]["content"] = reply
    rec["turns"][4]["content"] = reply
    clean = transcript("aimhef-0000000a", 2, n_turns=3)
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec, clean],
        judges=[judge(rec), judge(clean)],
    )
    by_id = {
        t.sim.transcript_id.rsplit(":", 1)[-1]: t for t in only_run(build(results)).transcripts
    }
    copied = by_id["1"]
    assert copied.sim.copied_turn == 3
    assert copied.bucket == "invalid"
    assert copied.sim.fidelity_reasons[0] == (
        "simulator",
        "the simulated person copied the AI's message at turn 3",
    )
    assert by_id["2"].bucket == "judged"


@pytest.mark.parametrize(
    ("records_path", "name"),
    [
        ("test_case_records/examples/l3_launch_smoke.jsonl", "examples/l3_launch_smoke"),
        ("data-repository/test_case_records/suite.jsonl", "suite"),
        ("/tmp/elsewhere/pilot.jsonl", "pilot"),
        ("", "pilot"),
    ],
)
def test_a_suite_is_named_by_its_path_under_test_case_records(records_path: str, name: str) -> None:
    assert F.suite_name_of(records_path, Path("inputs/pilot.jsonl")) == name


def test_a_preempted_behavior_and_its_parts_reach_the_row() -> None:
    verdict = fidelity(
        "not_tested",
        behaviors=[
            {
                "behavior": "kl-900501",
                "catalog_occurred": False,
                "assertion_occurred": False,
                "citations": [],
                "parts": [
                    {
                        "part": "a",
                        "present": True,
                        "citations": [{"turn": 1, "speaker": "user", "quote": "hello"}],
                    },
                    {"part": "b", "present": False, "citations": []},
                ],
                "miss": "preempted",
                "preempted_by": "resolved",
                "ai_citation": {"turn": 1, "speaker": "assistant", "quote": "reply"},
            }
        ],
    )
    raw = json.dumps(transcript("aimhef-0000000a", fidelity=verdict)).encode()
    row = F.parse_sim("run-a", "batch-001", Path("run-a-sim.jsonl"), 1, 0, raw)
    assert row.fidelity == "not_tested"
    assert row.preempted_by == "resolved"
    assert 1 in row.fidelity_turns


def test_a_check_file_beside_the_transcripts_gives_their_verdict(tmp_path: Path) -> None:
    root = tmp_path / "results"
    rec = transcript("aimhef-0000000a", status=None)
    write_run(root, "run-a", cases=[case("aimhef-0000000a")], sims=[rec])
    write_jsonl(
        root / "run-a" / "batch-001" / "run-a-check.jsonl",
        [
            {
                "transcript_id": rec["transcript_id"],
                "test_case_id": "aimhef-0000000a",
                "system_under_test": rec["system_under_test"],
                "fidelity": fidelity("invalid"),
            }
        ],
    )
    snapshot = build(root)
    (row,) = snapshot.sims
    assert row.fidelity == "invalid"
    assert row.fidelity_reasons == (("simulator", "kl-900501"),)
    assert snapshot.check_verdicts[row.key]["status"] == "invalid"
