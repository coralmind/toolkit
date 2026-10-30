"""The ingest writes the page data: relative paths, atomic files, facts but not wording."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _results import (
    OTHER,
    assessment,
    case,
    judge,
    transcript,
    write_jsonl,
    write_persona,
    write_run,
)

from coralmind_viewer import export as E
from coralmind_viewer import export_models as M
from coralmind_viewer import facts as F
from coralmind_viewer.trace_loaders import check_call_from_rows, check_trace_path


def ingest(results: Path, out: Path) -> E.Writer:
    snapshot = F.Index(results).rebuild()
    assert snapshot is not None
    return E.export(snapshot, out)


def read(out: Path, *parts: str) -> dict:
    return json.loads((out.joinpath(*parts)).read_text(encoding="utf-8"))


def test_every_page_file_is_written_and_validates(results: Path, tmp_path: Path) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[assessment()])],
    )
    out = tmp_path / "gen"
    ingest(results, out)
    index = M.IndexData.model_validate(read(out, "index.json"))
    (suite,) = index.suites
    file = M.SuiteFileData.model_validate(read(out, "suite-files", f"{suite.key}.json"))
    assert len(file.versions) == 1
    M.RunData.model_validate(read(out, "runs", f"{E.key_of('run-a')}.json"))
    data = M.CaseData.model_validate(
        read(out, "cases", f"{E.key_of('run-a', rec['transcript_id'])}.json")
    )
    assert data.judges[0].target_route is not None
    assert data.judges[0].target_route.outcome == "fires"
    assert M.LibraryData.model_validate(read(out, "library.json")).roles["originator"]


def test_a_suite_is_a_test_case_file_across_its_versions(
    results: Path, tmp_path: Path, data_root: Path
) -> None:
    write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    records = data_root / "test_case_records"
    lines = [case("aimhef-0000000a"), case("aimhef-0000000b")]
    (records / "suite.jsonl").write_text("".join(json.dumps(c) + "\n" for c in lines))
    (records / "unrun.jsonl").write_text(json.dumps(case("aimhef-0000000c")) + "\n")
    out = tmp_path / "gen"
    ingest(results, out)
    index = M.IndexData.model_validate(read(out, "index.json"))
    assert [s.name for s in index.suites] == ["suite"]
    assert not (out / "suite-files" / f"{E.key_of('unrun')}.json").exists()
    assert index.runs[0].suite == "suite"
    file = M.SuiteFileData.model_validate(read(out, "suite-files", f"{E.key_of('suite')}.json"))
    assert file.in_records
    assert [c.test_case_id for c in file.cases] == ["aimhef-0000000a", "aimhef-0000000b"]
    (version,) = file.versions
    assert (version.current, version.cases, version.matches) == (False, 1, 1)
    assert [t.same_case for t in file.transcripts] == [True]


def test_no_generated_file_holds_an_absolute_path(results: Path, tmp_path: Path) -> None:
    write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    out = tmp_path / "gen"
    ingest(results, out)
    index = read(out, "index.json")
    assert index["root"] == "results"
    assert "site" not in index
    for path in out.rglob("*.json"):
        assert str(tmp_path) not in path.read_text(encoding="utf-8"), path.name
    run = read(out, "runs", f"{E.key_of('run-a')}.json")
    assert run["manifest"] == "../results/run-a/inputs/suite.yaml"


def test_the_results_folder_alone_chooses_what_is_ingested(results: Path, tmp_path: Path) -> None:
    from coralmind_viewer.__main__ import main

    write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    out = tmp_path / "gen"
    with pytest.raises(SystemExit) as done:
        main(["ingest", "--results", str(results), "--out", str(out)])
    assert done.value.code == 0
    assert [r["name"] for r in read(out, "index.json")["runs"]] == ["run-a"]
    with pytest.raises(SystemExit) as refused:
        main(["ingest", "--site", "public", "--results", str(results), "--out", str(out)])
    assert refused.value.code == 2


def test_the_records_a_transcript_page_links_to_are_copied_into_the_page_data(
    results: Path, tmp_path: Path
) -> None:
    rec = transcript("aimhef-0000000a")
    record = judge(
        rec,
        assessments=[assessment()],
        window=("2026-10-20T20:00:00+00:00", "2026-10-20T20:01:00+00:00"),
    )
    record["judge_prompt_snapshot"] = {"stage1": "the stage 1 template"}
    run = write_run(results, "run-a", cases=[case("aimhef-0000000a")], sims=[rec], judges=[record])

    def call(seq: int, test_case_id: str = "aimhef-0000000a") -> dict:
        return {
            "type": "llm_call",
            "seq": str(seq),
            "test_case_id": test_case_id,
            "role": "coralmind_s1_ai_behavior-g01" if seq < 3 else "coralmind_s2_Pathway-902",
            "timestamp": f"2026-10-20T20:00:{seq}0+00:00",
            "prompt": f"prompt {seq}",
            "response": f"response {seq}",
            "system": f"system {seq}",
        }

    write_jsonl(
        run / "batch-001" / "run-a-judge.trace.jsonl",
        [call(2), call(3), call(4, "aimhef-0000000z")],
    )
    out = tmp_path / "gen"
    ingest(results, out)
    data = M.RecordsData.model_validate(
        read(out, "records", f"{E.key_of('run-a', rec['transcript_id'])}.json")
    )
    assert data.transcript_id == rec["transcript_id"]
    assert data.transcript is not None
    assert data.transcript["turns"][0]["content"] == "hello 1"
    (ref,) = data.judges
    assert ref == "run-a-judge.jsonl:1"
    copied = data.judges[ref]
    assert "judge_prompt_snapshot" not in copied.record
    assert copied.record["judge_model_id"] == "judge-model"
    assert copied.prompts == {"stage1": "the stage 1 template"}
    assert [(c.seq, c.prompt, c.response, c.system) for c in copied.calls] == [
        (2, "prompt 2", "response 2", "system 2"),
        (3, "prompt 3", "response 3", "system 3"),
    ]


def test_writes_are_atomic_and_skip_unchanged_files(results: Path, tmp_path: Path) -> None:
    write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    out = tmp_path / "gen"
    first = ingest(results, out)
    assert first.changed > 0
    stamps = {p: p.stat().st_mtime_ns for p in out.rglob("*.json")}
    second = ingest(results, out)
    changed = [p.name for p in out.rglob("*.json") if p.stat().st_mtime_ns != stamps[p]]
    assert set(changed) <= {"index.json"}, "only the index, which carries the time, may change"
    assert second.changed == len(changed)
    assert not list(out.rglob("*.tmp"))


def test_a_deleted_run_is_pruned(results: Path, tmp_path: Path) -> None:
    write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    run_b = write_run(results, "run-b", cases=None, sims=[transcript("aimhef-0000000b")])
    out = tmp_path / "gen"
    ingest(results, out)
    assert (out / "runs" / f"{E.key_of('run-b')}.json").exists()
    for path in sorted(run_b.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    run_b.rmdir()
    ingest(results, out)
    assert not (out / "runs" / f"{E.key_of('run-b')}.json").exists()


def test_the_copied_turn_is_marked_in_the_case_data(results: Path, tmp_path: Path) -> None:
    rec = transcript("aimhef-0000000a", n_turns=3)
    reply = "You don't have to rush. Just aim for **small, ordinary steps** and go gently."
    rec["turns"][1]["content"] = reply
    rec["turns"][4]["content"] = reply.replace("**", "")
    write_run(results, "run-a", cases=[case("aimhef-0000000a")], sims=[rec])
    out = tmp_path / "gen"
    ingest(results, out)
    data = read(out, "cases", f"{E.key_of('run-a', rec['transcript_id'])}.json")
    flags = [
        (e["number"], m["role"], m["copied_from"]) for e in data["turns"] for m in e["messages"]
    ]
    assert [f for f in flags if f[2] is not None] == [(3, "user", 1)]


def test_codes_are_written_and_the_wording_is_left_to_the_pages(
    results: Path, tmp_path: Path
) -> None:
    rec = transcript(
        "aimhef-0000000a", sim_failure={"reason": "timeout", "detail": "took too long"}
    )
    write_run(results, "run-a", cases=[case("aimhef-0000000a")], sims=[rec])
    out = tmp_path / "gen"
    ingest(results, out)
    text = (out / "cases" / f"{E.key_of('run-a', rec['transcript_id'])}.json").read_text()
    assert '"sim_failure":"timeout"' in text
    assert "timed out" not in text
    assert "simulation failed" not in text


def test_a_runs_own_personas_and_its_own_time_are_used(
    results: Path, tmp_path: Path, data_root: Path
) -> None:
    write_persona(data_root / "persona_records")
    run = write_run(
        results,
        "run-a-20261023-165536",
        cases=[case("aimhef-0000000a")],
        sims=[transcript("aimhef-0000000a")],
    )
    own = run / "inputs" / "persona_records"
    write_persona(own)
    (own / "p-fix-01.json").write_text(
        (own / "p-fix-01.json").read_text().replace("Dana, 41", "Dana as the run saw her")
    )
    out = tmp_path / "gen"
    ingest(results, out)
    data = read(
        out, "cases", f"{E.key_of(run.name, 'demo:azure-gpt-5-4-mini:aimhef-0000000a:1')}.json"
    )
    assert data["case"]["persona"]["description"].startswith("Dana as the run saw her")
    (suite,) = read(out, "index.json")["suites"]
    assert suite["last_run"] == "2026-10-23T16:55:36"


def test_a_missing_file_is_reported_and_keeps_the_old_data(
    results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from coralmind_viewer import ingest as I

    write_run(
        results, "run-a", cases=[case("aimhef-0000000a")], sims=[transcript("aimhef-0000000a")]
    )
    out = tmp_path / "gen"
    assert I.run_ingest(results, out) == 0
    before = read(out, "index.json")

    def fail(*args: object) -> F.FileRows:
        raise OSError("disk went away")

    monkeypatch.setattr(F, "read_sim_file", fail)
    write_run(results, "run-b", cases=None, sims=[transcript("aimhef-0000000b")])
    assert I.run_ingest(results, out) == 1
    assert read(out, "index.json") == before
    assert os.path.exists(out / "status.json")


def test_an_occurrence_carries_the_judges_reasoning_and_not_the_placeholder(
    results: Path, tmp_path: Path
) -> None:
    rec = transcript("aimhef-0000000a")
    route = assessment()
    route["rationale"] = "1 instance(s) found; strongest is Pathway-902#1 (supported)."
    route["instances"][0].update(
        rationale="The answer at turn 2 settles the doubt.",
        evidence_gaps=["No second return after turn 2."],
        validation_notes=["turn_window [1, 3] replaced by the steps' span [1, 2]"],
    )
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a")],
        sims=[rec],
        judges=[judge(rec, assessments=[route])],
    )
    out = tmp_path / "gen"
    ingest(results, out)
    data = M.CaseData.model_validate(
        read(out, "cases", f"{E.key_of('run-a', rec['transcript_id'])}.json")
    )
    (rail,) = data.judges[0].routes
    assert rail.rationale is None
    (inst,) = rail.instances
    assert inst.rationale == "The answer at turn 2 settles the doubt."
    assert inst.evidence_gaps == ["No second return after turn 2."]
    assert inst.uncertainty == 0.1
    assert inst.adjustments == ["turn_window [1, 3] replaced by the steps' span [1, 2]"]


def test_a_fidelity_part_carries_its_clause_from_the_library() -> None:
    library = F.Library(
        version="0",
        digest="d",
        routes={},
        names={},
        criteria_parts={"kl-000001": {"a": "returns to the theme", "b": "adds nothing new"}},
        parts_needed={"kl-000001": 2},
    )
    verdict = {
        "status": "invalid",
        "behaviors": [
            {
                "behavior": "kl-000001",
                "catalog_occurred": False,
                "assertion_occurred": True,
                "miss": "not_attempted",
                "parts": [
                    {"part": "a", "present": True, "citations": [], "reason": "comes back"},
                    {
                        "part": "b",
                        "present": False,
                        "citations": [],
                        "ruled_out_by": {"turn": 3, "speaker": "user", "quote": "and my dog"},
                        "reason": "adds a new fact: the dog",
                    },
                ],
            }
        ],
    }
    view = E._fidelity_view({}, None, lambda turn, role: "", verdict, library)
    assert view is not None
    assert [(p.part, p.text) for p in view.behaviors[0].parts] == [
        ("a", "returns to the theme"),
        ("b", "adds nothing new"),
    ]
    assert view.behaviors[0].parts_needed == 2
    a, b = view.behaviors[0].parts
    assert (a.ruled_out, a.reason) == (None, "comes back")
    assert b.ruled_out is not None
    assert (b.ruled_out.turn, b.reason) == (3, "adds a new fact: the dog")
    without = E._fidelity_view({}, None, lambda turn, role: "", verdict)
    assert without is not None
    assert without.behaviors[0].parts[1].text == ""
    assert without.behaviors[0].parts_needed == 0


def test_the_fidelity_check_call_is_read_from_the_check_trace_beside_the_transcripts(
    tmp_path: Path,
) -> None:
    sim_file = tmp_path / "run-a__azure-m-sim.jsonl"
    assert check_trace_path(sim_file) == tmp_path / "run-a__azure-m-check.trace.jsonl"
    rows = [
        {"role": "elicitation_fidelity", "test_case_id": "aimhef-0000000a", "prompt": "I said A"},
        {"role": "elicitation_fidelity", "test_case_id": "aimhef-0000000a", "prompt": "I said B"},
        {"role": "elicitation_fidelity", "test_case_id": "aimhef-0000000b", "prompt": "I said B"},
        {"role": "judge", "test_case_id": "aimhef-0000000a", "prompt": "I said B"},
    ]
    row = check_call_from_rows(rows, "aimhef-0000000a", "said B")
    assert row is rows[1]
    assert check_call_from_rows(rows, "aimhef-0000000c", "said B") is None


def test_a_skipped_target_lists_its_behaviors_and_whether_stage_1_saw_them(
    results: Path, tmp_path: Path
) -> None:
    rec = transcript("aimhef-0000000a")
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a", pathway_id=OTHER)],
        sims=[rec],
        judges=[judge(rec, skipped=[OTHER])],
    )
    out = tmp_path / "gen"
    ingest(results, out)
    data = M.CaseData.model_validate(
        read(out, "cases", f"{E.key_of('run-a', rec['transcript_id'])}.json")
    )
    view = data.judges[0]
    assert view.target_missing == "no participant"
    assert [(p.id, p.seen, p.failed) for p in view.target_parts] == [("kl-900001", True, False)]


def test_a_target_behavior_the_judge_ruled_out_carries_its_reason(
    results: Path, tmp_path: Path
) -> None:
    rec = transcript("aimhef-0000000a")
    record = judge(rec, skipped=[OTHER])
    stage1 = record["coralmind_run"]["stage1"]
    stage1["observations"] = []
    stage1["negatives"] = [{"behavior_id": "kl-900001", "reason": "the number is real"}]
    write_run(
        results,
        "run-a",
        cases=[case("aimhef-0000000a", pathway_id=OTHER)],
        sims=[rec],
        judges=[record],
    )
    out = tmp_path / "gen"
    ingest(results, out)
    data = M.CaseData.model_validate(
        read(out, "cases", f"{E.key_of('run-a', rec['transcript_id'])}.json")
    )
    assert [(p.id, p.seen, p.reason) for p in data.judges[0].target_parts] == [
        ("kl-900001", False, "the number is real")
    ]
