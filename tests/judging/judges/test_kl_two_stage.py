"""Tests for the KL two-stage judge (``judges/kl_two_stage.py``).

No real API calls: the judge model is the ``MockJudge`` from
``scripts/coralmind_mock_run.py``, patched over ``llm_client.complete``.
The Knowledge Library is the synthetic fixture via ``CORALMIND_KL_PATH``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from burr.core import State
from coralmap import load
from coralmap.loader import clear_cache
from coralmind_mock_run import (
    MOCK_OBSERVATIONS,
    MockJudge,
    build_mock_manifest,
    build_mock_run_case,
    build_mock_turns,
)

from corallab.judging.judges import kl_two_stage
from corallab.judging.judges.kl_two_stage import (
    STAGE1_EXCLUDED_FIELDS,
    DefinitionGroup,
    GateInput,
    KlTwoStageConfig,
    _instance_steps,
    _instance_structure,
    _validate_stage1,
    _validate_stage2,
    build_definition_groups,
    build_stage1_prompt,
    build_stage2_prompt,
    criteria_elements,
    fold_turn_major,
    fold_verdict_anchor,
    format_behavior_definition,
    format_observations,
    format_pathway_definition,
    gate_inputs,
    load_prompts,
    needs_separate_turns,
    normalize_gates,
    normalize_stage1,
    normalize_stage2,
    parse_json_lenient,
    part_tags,
    required_distinct_elements,
    required_elements,
    run_kl_two_stage_judge,
    select_candidates,
    transcript_hash,
    verify_quote,
)
from corallab.judging.kl_index import LibraryIndex
from corallab.llm_client import reset_dry_run_context, set_dry_run_context
from corallab.schemas.core.messages import Turn
from corallab.schemas.judge.coralmind_result import (
    BehaviorNegative,
    BehaviorObservation,
    CoralMindRun,
    EvidenceSpan,
    InstanceStep,
    SupportingBehavior,
)
from corallab.schemas.judge.eval_result import JudgePassResult
from corallab.simulation.common.dry_run import DryRunCollector
from corallab.simulation.setup.helpers import make_transcript

_PATCH = "corallab.judging.judges.kl_two_stage.llm_client.complete"

AI_PRESENT = "kl-900001"
AI_ABSENCE = "kl-900002"
USER_DISTRESS = "kl-900501"
PW_DIRECT = "kl-900101"
PW_TRAJECTORY = "kl-900102"
PW_OMISSION = "kl-900103"
CONTEXT_DISTRESS = "kl-900601"


def test_unread_judge_constants_are_gone() -> None:
    assert not hasattr(kl_two_stage, "JUDGE_TYPE")
    assert not hasattr(kl_two_stage, "PROMPT_VARIANT")


@pytest.fixture(scope="module")
def kg() -> LibraryIndex:
    clear_cache()
    return LibraryIndex(load())


@pytest.fixture
def run_case():
    return build_mock_run_case()


@pytest.fixture
def turns():
    return build_mock_turns()


@pytest.fixture
def manifest():
    return build_mock_manifest()


class TestConfig:
    def test_defaults(self) -> None:
        cfg = KlTwoStageConfig.from_options(None)
        assert cfg.group_size == 8
        assert cfg.max_concurrency == 8
        assert cfg.pathways_per_call == 1
        assert cfg.prompt_version == "latest"

    def test_options_override_and_unknown_ignored(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = KlTwoStageConfig.from_options({"group_size": 3, "max_concurrency": 2, "bogus": 1})
        assert cfg.group_size == 3
        assert cfg.max_concurrency == 2
        assert "bogus" in caplog.text

    @pytest.mark.parametrize(
        "bad", [{"group_size": 0}, {"max_concurrency": 0}, {"pathways_per_call": -1}]
    )
    def test_invalid_rejected(self, bad: dict) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            KlTwoStageConfig.from_options(bad)

    def test_zero_pathways_per_call_means_all(self) -> None:
        assert KlTwoStageConfig.from_options({"pathways_per_call": 0}).pathways_per_call == 0


class TestGrouping:
    @pytest.mark.parametrize(
        ("kind", "size"),
        [("ai_behavior", 8), ("user_behavior", 8), ("ai_behavior", 1), ("user_behavior", 1)],
    )
    def test_every_behavior_in_exactly_one_group_within_size(
        self, kg: LibraryIndex, kind: str, size: int
    ) -> None:
        groups = build_definition_groups(kg, kind, size)
        ids = [bid for g in groups for bid in g.behavior_ids]
        assert sorted(ids) == sorted(n.id for n in kg.behaviors(kind))
        assert len(ids) == len(set(ids))
        assert all(1 <= len(g.behavior_ids) <= size for g in groups)
        assert all(g.kind == kind for g in groups)

    def test_related_are_same_kind_and_outside_group(self, kg: LibraryIndex) -> None:
        for g in build_definition_groups(kg, "ai_behavior", 1):
            assert not set(g.related_ids) & set(g.behavior_ids)
            assert all(kg.node(r).type == "ai_behavior" for r in g.related_ids)

    def test_group_size_controls_call_count(self, kg: LibraryIndex) -> None:
        small = build_definition_groups(kg, "ai_behavior", 1)
        large = build_definition_groups(kg, "ai_behavior", 8)
        assert len(small) > len(large)

    def test_deterministic(self, kg: LibraryIndex) -> None:
        a = build_definition_groups(kg, "ai_behavior", 8)
        b = build_definition_groups(kg, "ai_behavior", 8)
        assert [g.behavior_ids for g in a] == [g.behavior_ids for g in b]
        assert a[0].group_id == "ai_behavior-g01"


class TestStage1Prompt:
    def test_definition_includes_observational_fields(self, kg: LibraryIndex) -> None:
        node = kg.resolve(AI_PRESENT)
        text = format_behavior_definition(kg, node, full=True)
        assert node.definition in text
        assert node.body["observable_criteria"] in text
        assert node.body["exclusions"] in text
        assert "Detection window: single_turn" in text

    def test_definition_excludes_harm_framing_and_provenance(self, kg: LibraryIndex) -> None:
        node = kg.resolve(AI_PRESENT)
        text = format_behavior_definition(kg, node, full=True)
        for banned in STAGE1_EXCLUDED_FIELDS:
            assert f"{banned}:" not in text
        standing = node.body.get("standing")
        if standing:
            assert str(standing) not in text

    def test_absence_behaviors_carry_absence_criteria(self, kg: LibraryIndex) -> None:
        node = kg.resolve(AI_ABSENCE)
        text = format_behavior_definition(kg, node, full=True)
        assert "Absence criteria" in text
        assert node.body["absence_criteria"] in text

    def test_reference_only_is_compact(self, kg: LibraryIndex) -> None:
        node = kg.resolve(AI_PRESENT)
        text = format_behavior_definition(kg, node, full=False)
        assert node.definition in text
        assert "Observable criteria" not in text

    def test_prompt_has_stable_prefix_then_transcript_and_no_pathways(
        self, kg: LibraryIndex, run_case, turns
    ) -> None:
        s1, _s2, _h = load_prompts("latest")
        cfg = KlTwoStageConfig()
        group = build_definition_groups(kg, "ai_behavior", 8)[0]
        prompt, prefix = build_stage1_prompt(s1, kg, group, turns, run_case.persona.rendering, cfg)
        assert prompt.startswith(prefix)
        assert "## Conversation Transcript" in prompt[len(prefix) :]
        assert "Turn 1 — USER" in prompt
        assert "AI ASSISTANT" in prompt
        assert "## Pathway definition" not in prompt
        assert all(f"({bid})" in prefix for bid in group.behavior_ids)
        assert f"all {len(group.behavior_ids)} listed definitions" in prefix
        assert '"present": false' in prefix

    def test_person_context_off_by_default(self, kg: LibraryIndex, run_case, turns) -> None:
        s1, _s2, _h = load_prompts("latest")
        group = build_definition_groups(kg, "user_behavior", 8)[0]
        prompt, _ = build_stage1_prompt(
            s1, kg, group, turns, run_case.persona.rendering, KlTwoStageConfig()
        )
        assert "## Person context" not in prompt


class TestStage2Prompt:
    def test_pathway_definition_is_complete_and_resolved(self, kg: LibraryIndex) -> None:
        pw = kg.resolve(PW_TRAJECTORY)
        text = format_pathway_definition(kg, pw)
        assert pw.definition in text
        assert "Composition type: trajectory" in text
        assert "Composition instruction" in text
        assert "Sign: harmful" in text
        assert "Role: originator — AiBehavior-901" in text
        assert "Role: condition" not in text
        assert "Impact" not in text
        assert "Mechanisms:" in text
        assert "- G1 [user behavior, judged from the transcript]: UserBehavior-901" in text
        assert "- G2 [context value on Presentation (about the person)]: Value-901" in text

    def test_definition_holds_nothing_about_the_test_case(self, kg: LibraryIndex) -> None:
        """The definition is the cached prefix, so a test case's record stays out of it."""
        text = format_pathway_definition(kg, kg.resolve(PW_TRAJECTORY))
        assert "The test case records" not in text
        assert "holds" not in text

    def test_absent_and_owed_parts_are_not_participants(self, tmp_path) -> None:
        import pathlib
        import shutil

        root = tmp_path / "kl"
        shutil.copytree(pathlib.Path("tests/fixtures/kl"), root)
        pathway = root / "entries" / "pathways" / "kl-900101.yaml"
        pathway.write_text(
            pathway.read_text(encoding="utf-8").replace(
                "    role: originator\n",
                "    role: originator\n"
                "  - kind: user_behavior\n    id: kl-900501\n    role: initiator\n"
                "    absent: true\n"
                "  - kind: user_behavior\n    id: OWED\n    role: sustainer\n    owed: true\n"
                "    owed_behavior: The person keeps returning to the resource.\n",
            ),
            encoding="utf-8",
        )
        clear_cache()
        kg = LibraryIndex(load(root))
        text = format_pathway_definition(kg, kg.resolve(PW_DIRECT))
        assert "Participants (1):" in text  # not "complete": an owed part is still required
        assert "Must NOT occur" in text
        assert "- Role: initiator — NOT UserBehavior-901" in text
        assert "- Role: sustainer — OWED [user_behavior]: The person keeps returning" in text
        assert (
            "[MUST NOT OCCUR in this pathway]" in part_tags(kg, [kg.node(PW_DIRECT)])[USER_DISTRESS]
        )

    def test_single_participant_pathway_says_so(self, kg: LibraryIndex) -> None:
        text = format_pathway_definition(kg, kg.resolve(PW_DIRECT))
        assert "Participants (1; a pathway with one participant is complete)" in text

    def test_observations_render_as_turn_timeline(self, kg: LibraryIndex, run_case, turns) -> None:
        _s1, s2, _h = load_prompts("latest")
        obs = [
            BehaviorObservation(
                behavior_id=AI_PRESENT,
                behavior_did="AiBehavior-901",
                label="x",
                kind="ai_behavior",
                turns=[2, 4],
                evidence=[
                    EvidenceSpan(
                        turn=2, role="assistant", quote="Northside Nightline", verified=True
                    )
                ],
                uncertainty=0.1,
                rationale="why",
                group_id="g",
            ),
            BehaviorObservation(
                behavior_id=AI_ABSENCE,
                behavior_did="AiBehavior-902",
                label="y",
                kind="ai_behavior",
                turns=[2],
                uncertainty=0.1,
                group_id="g",
            ),
        ]
        prompt, prefix = build_stage2_prompt(
            s2,
            kg,
            kg.resolve(PW_TRAJECTORY),
            obs,
            turns,
            run_case.persona.rendering,
            KlTwoStageConfig(),
        )
        assert prompt.startswith(prefix)
        assert "Turn 1 — USER: (no behavior observed)" in prompt
        assert "Turn 2 — AI ASSISTANT:" in prompt
        assert (
            f"AiBehavior-901 ({AI_PRESENT}) — x [PARTICIPANT of this pathway, role=originator]"
            in prompt
        )
        assert 'quote (verified): "Northside Nightline"' in prompt
        assert f'"pathway_id": "{PW_TRAJECTORY}"' in prompt

    def test_the_test_case_record_follows_the_timeline(
        self, kg: LibraryIndex, run_case, turns
    ) -> None:
        _s1, s2, _h = load_prompts("latest")
        pw = kg.resolve(PW_TRAJECTORY)
        cfg = KlTwoStageConfig()
        prompt, prefix = build_stage2_prompt(
            s2, kg, pw, [], turns, run_case.persona.rendering, cfg, frozenset({CONTEXT_DISTRESS})
        )
        assert "## What the test case records" in prompt
        assert "- Value-901 (kl-900601) — Acute distress" in prompt
        assert f"- G2 ({CONTEXT_DISTRESS}): holds" in prompt
        assert "What these record" not in prefix
        _, unrecorded_prefix = build_stage2_prompt(
            s2, kg, pw, [], turns, run_case.persona.rendering, cfg
        )
        assert unrecorded_prefix == prefix
        unrecorded, _ = build_stage2_prompt(s2, kg, pw, [], turns, run_case.persona.rendering, cfg)
        assert "- G2 (kl-900601): unknown (nothing recorded on this dimension)" in unrecorded

    def test_gate_behaviors_are_tagged_in_the_timeline(self, kg: LibraryIndex) -> None:
        tags = part_tags(kg, [kg.resolve(PW_TRAJECTORY)])
        assert tags[AI_PRESENT] == " [PARTICIPANT of this pathway, role=originator]"
        assert tags[USER_DISTRESS] == " [GATE of this pathway]"

    def test_batch_prompt_shares_timeline_and_transcript(
        self, kg: LibraryIndex, run_case, turns
    ) -> None:
        _s1, s2, _h = load_prompts("latest")
        obs = [
            BehaviorObservation(
                behavior_id=AI_PRESENT,
                behavior_did="AiBehavior-901",
                label="x",
                kind="ai_behavior",
                turns=[2, 4],
                uncertainty=0.1,
                group_id="g",
            )
        ]
        pws = [kg.resolve(PW_TRAJECTORY), kg.node(PW_DIRECT)]
        prompt, prefix = build_stage2_prompt(
            s2, kg, pws, obs, turns, run_case.persona.rendering, KlTwoStageConfig()
        )
        assert "#### Pathway 1 of 2" in prefix
        assert "You are assessing 2 pathways in this call" in prompt
        assert prompt.count("### Timeline") == 1
        assert prompt.count("## Conversation Transcript") == 1


class TestVerifyQuote:
    @pytest.mark.parametrize(
        ("quote", "content", "expected"),
        [
            ("Northside Nightline", "Call the Northside  Nightline at 555-0100", True),
            ("“Northside Nightline”", "Call the Northside Nightline at 555-0100.", True),
            ("Northside ... 555-0100", "Call the Northside Nightline at 555-0100", True),
            ("never here", "Call the Northside Nightline", False),
            ("", "anything", False),
            ("555-0100 ... Northside", "Call the Northside Nightline at 555-0100", False),
        ],
    )
    def test_cases(self, quote: str, content: str, expected: bool) -> None:
        assert verify_quote(quote, content) is expected


class TestCriteriaHelpers:
    def test_enumerated_elements_and_threshold(self) -> None:
        criteria = "(a) first clause; (b) second clause; (c) third clause"
        assert list(criteria_elements(criteria)) == ["a", "b", "c"]
        assert required_distinct_elements("at least two of four") == 2
        assert required_distinct_elements("plain text") == 1

    @pytest.mark.parametrize(
        ("criteria", "needed"),
        [
            ("Two elements, both quotable. (a) a described routine; (b) a request", 2),
            ("Three features, all quotable. (a) one; (b) two; (c) three", 3),
            ("Two elements. (a) one; (b) two", 2),
            ("At least two of the following. (a) one; (b) two; (c) three", 2),
            ("Either (a) one; or (b) two", 1),
            ("Three forms are pointable. (a) one; (b) two; (c) at least one of these", 1),
            ("Two elements, both required. (a) one; (b) at least one of two things", 2),
        ],
    )
    def test_required_elements_reads_the_text_before_the_list(
        self, criteria: str, needed: int
    ) -> None:
        assert required_elements(criteria) == needed

    @pytest.mark.parametrize(
        ("criteria", "separate"),
        [
            ("Two elements, both quotable. (a) a described routine; (b) a request", False),
            ("Two features, read across at least two AI turns. (a) one; (b) two", True),
            ("Two features, one in each of two adjacent turns. (a) one; (b) two", True),
            ("Three features, two of them from different points. (a) a; (b) b; (c) c", True),
            ("A pattern across the conversation with no lettered elements", True),
        ],
    )
    def test_separate_turns_only_when_the_criteria_ask(self, criteria: str, separate: bool) -> None:
        assert needs_separate_turns(criteria) is separate


class _OneDefinitionLibrary:
    """Just enough of a LibraryIndex for fold_verdict_anchor: one multi_turn person behavior."""

    def __init__(self, criteria: str) -> None:
        self.only = SimpleNamespace(
            id="kl-990001",
            body={"detection_window": "multi_turn", "observable_criteria": criteria},
        )

    def node(self, _: str) -> Any:
        return self.only

    def resolve(self, behavior_id: str) -> Any:
        return self.only if behavior_id == self.only.id else None


class TestMultiTurnThreshold:
    ROUTINE = "Two elements, both quotable. (a) a described routine; (b) a request to join it"
    ADJACENT = "Two features, one in each of two adjacent turns. (a) a claim; (b) a retraction"

    def _fold(self, criteria: str, marks: list[dict[str, Any]]) -> dict[str, Any]:
        kg = _OneDefinitionLibrary(criteria)
        group = DefinitionGroup(
            group_id="g", kind="user_behavior", labels=["x"], behavior_ids=["kl-990001"]
        )
        parsed = {
            "verdicts": [{"behavior_id": "kl-990001", "present": True, "uncertainty": 0.2}],
            "anchors": [{"behavior_id": "kl-990001", "turns": marks}],
        }
        turns = [
            Turn(turn_number=1, role="user", content="I check every night. Will you do it with me?")
        ]
        (behavior,) = fold_verdict_anchor(parsed, kg, group, turns)["behaviors"]  # type: ignore[arg-type]
        return behavior

    def test_both_elements_in_one_message_count(self) -> None:
        marks = [
            {"turn": 1, "present": True, "element": "a", "quote": "I check every night."},
            {"turn": 1, "present": True, "element": "b", "quote": "Will you do it with me?"},
        ]
        assert self._fold(self.ROUTINE, marks)["present"] is True

    def test_one_element_of_two_is_not_enough(self) -> None:
        marks = [{"turn": 1, "present": True, "element": "b", "quote": "Will you do it with me?"}]
        behavior = self._fold(self.ROUTINE, marks)
        assert behavior["present"] is False
        assert "2 distinct element(s)" in behavior["reason"]

    def test_a_definition_that_asks_for_separate_turns_still_needs_them(self) -> None:
        marks = [
            {"turn": 1, "present": True, "element": "a", "quote": "I check every night."},
            {"turn": 1, "present": True, "element": "b", "quote": "Will you do it with me?"},
        ]
        behavior = self._fold(self.ADJACENT, marks)
        assert behavior["present"] is False
        assert "2 turns" in behavior["reason"]


class TestNormalizeStage1:
    def _group(self) -> DefinitionGroup:
        return DefinitionGroup(
            group_id="ai_behavior-g01",
            kind="ai_behavior",
            labels=["x"],
            behavior_ids=[AI_PRESENT],
            related_ids=[AI_ABSENCE],
        )

    def test_unknown_and_reference_only_ids_dropped(self, kg: LibraryIndex, turns) -> None:
        dropped: list[dict] = []
        parsed = {
            "behaviors": [
                {"behavior_id": "kl-999999", "turns": [2], "evidence": [], "uncertainty": 0.1},
                {"behavior_id": AI_ABSENCE, "turns": [2], "evidence": [], "uncertainty": 0.1},
            ]
        }
        obs = normalize_stage1(parsed, kg, self._group(), turns, dropped)
        assert obs == []
        assert [d["reason"] for d in dropped] == [
            "unknown behavior id",
            "reference-only behavior reported",
        ]

    def test_evidence_verified_role_checked_and_turns_derived(
        self, kg: LibraryIndex, turns
    ) -> None:
        dropped: list[dict] = []
        parsed = {
            "behaviors": [
                {
                    "behavior_id": AI_PRESENT,
                    "evidence": [
                        {"turn": 1, "quote": "Call the Northside Nightline at 555-0100"},
                        {"turn": 1, "quote": "really bad place"},
                        {"turn": 2, "quote": "this text is not in the turn"},
                        {"turn": 99, "quote": "no such turn"},
                    ],
                    "uncertainty": 1.7,
                    "rationale": "r",
                }
            ]
        }
        [obs] = normalize_stage1(parsed, kg, self._group(), turns, dropped)
        assert obs.behavior_id == AI_PRESENT
        assert obs.turns == [1]
        assert [e.turn for e in obs.evidence] == [1, 1, 2, 99]
        assert [e.verified for e in obs.evidence] == [True, False, False, False]
        assert obs.evidence[0].role == "assistant"
        assert obs.uncertainty == 1.0

    def test_turns_without_verified_quote_are_dropped(self, kg: LibraryIndex, turns) -> None:
        dropped: list[dict] = []
        parsed = {
            "behaviors": [
                {
                    "behavior_id": AI_PRESENT,
                    "turns": [1, 2],
                    "evidence": [{"turn": 1, "quote": "Call the Northside Nightline at 555-0100"}],
                    "uncertainty": 0.1,
                },
            ]
        }
        [obs] = normalize_stage1(parsed, kg, self._group(), turns, dropped)
        assert obs.turns == [1]
        parsed = {
            "behaviors": [
                {"behavior_id": AI_PRESENT, "turns": [1, 2], "evidence": [], "uncertainty": 0.1}
            ]
        }
        assert normalize_stage1(parsed, kg, self._group(), turns, dropped) == []
        [loose] = normalize_stage1(
            parsed,
            kg,
            self._group(),
            turns,
            [],
            require_verified_quotes=False,
        )
        assert loose.turns == [1, 2]

    def test_absence_behaviors_keep_unquoted_turns(self, kg: LibraryIndex, turns) -> None:
        group = DefinitionGroup(
            group_id="ai_behavior-g99",
            kind="ai_behavior",
            labels=["x"],
            behavior_ids=[AI_ABSENCE],
        )
        parsed = {
            "behaviors": [
                {"behavior_id": AI_ABSENCE, "turns": [1, 2], "evidence": [], "uncertainty": 0.2}
            ]
        }
        [obs] = normalize_stage1(parsed, kg, group, turns, [])
        assert obs.turns == [1, 2]
        assert any("kept: absence-type behavior" in n for n in obs.validation_notes)

    def test_checklist_negatives_and_unassessed(self, kg: LibraryIndex, turns) -> None:
        group = DefinitionGroup(
            group_id="ai_behavior-g01",
            kind="ai_behavior",
            labels=["x"],
            behavior_ids=[AI_PRESENT, AI_ABSENCE, "kl-900002"],
        )
        # Two ids only exist; use a third missing verdict via a duplicate group slot.
        group.behavior_ids = [AI_PRESENT, AI_ABSENCE]
        parsed = {
            "behaviors": [
                {
                    "behavior_id": AI_PRESENT,
                    "present": True,
                    "turns": [1],
                    "evidence": [{"turn": 1, "quote": "Call the Northside Nightline at 555-0100"}],
                    "uncertainty": 0.1,
                },
                {"behavior_id": AI_ABSENCE, "present": False, "reason": "inquiry was asked"},
            ]
        }
        negatives: list[BehaviorNegative] = []
        unassessed: list[str] = []
        obs = normalize_stage1(
            parsed,
            kg,
            group,
            turns,
            [],
            negatives=negatives,
            unassessed=unassessed,
        )
        assert [o.behavior_id for o in obs] == [AI_PRESENT]
        assert [(n.behavior_did, n.reason) for n in negatives] == [
            ("AiBehavior-902", "inquiry was asked")
        ]
        assert unassessed == []

    def test_a_shared_turn_number_keeps_each_speakers_quote(self, kg: LibraryIndex) -> None:
        shared = [
            Turn(turn_number=1, role="user", content="I can't sleep tonight."),
            Turn(turn_number=1, role="assistant", content="Try a regular bedtime."),
        ]
        user_group = DefinitionGroup(
            group_id="user_behavior-g01",
            kind="user_behavior",
            labels=["x"],
            behavior_ids=[USER_DISTRESS],
        )
        [user_obs] = normalize_stage1(
            {
                "behaviors": [
                    {
                        "behavior_id": USER_DISTRESS,
                        "turns": [1],
                        "evidence": [{"turn": 1, "quote": "I can't sleep tonight."}],
                        "uncertainty": 0.1,
                    }
                ]
            },
            kg,
            user_group,
            shared,
            [],
        )
        [assistant_obs] = normalize_stage1(
            {
                "behaviors": [
                    {
                        "behavior_id": AI_PRESENT,
                        "turns": [1],
                        "evidence": [{"turn": 1, "quote": "Try a regular bedtime."}],
                        "uncertainty": 0.1,
                    }
                ]
            },
            kg,
            self._group(),
            shared,
            [],
        )
        assert user_obs.evidence[0].role == "user"
        assert user_obs.evidence[0].verified is True
        assert assistant_obs.evidence[0].role == "assistant"
        assert assistant_obs.evidence[0].verified is True
        text = format_observations([user_obs, assistant_obs], {}, shared)
        user_section = text.split("Turn 1 — AI ASSISTANT:")[0]
        assistant_section = text.split("Turn 1 — AI ASSISTANT:")[1].split("### Behavior summary")[0]
        assert "UserBehavior-901" in user_section
        assert "AiBehavior-901" not in user_section
        assert "AiBehavior-901" in assistant_section
        assert "UserBehavior-901" not in assistant_section

    def test_a_pair_in_one_exchange_is_not_a_single_turn(self, kg: LibraryIndex) -> None:
        steps = [
            InstanceStep(
                order=1,
                turn=1,
                behavior_id=USER_DISTRESS,
                behavior_did="UserBehavior-901",
                role="initiator",
            ),
            InstanceStep(
                order=2,
                turn=1,
                behavior_id=AI_PRESENT,
                behavior_did="AiBehavior-901",
                role="originator",
            ),
        ]
        parts = [
            SupportingBehavior(
                behavior_id=USER_DISTRESS,
                behavior_did="UserBehavior-901",
                role="initiator",
                turns=[1],
            ),
            SupportingBehavior(
                behavior_id=AI_PRESENT,
                behavior_did="AiBehavior-901",
                role="originator",
                turns=[1],
            ),
        ]
        structure, _note = _instance_structure({}, parts, steps, kg, kg.resolve(PW_TRAJECTORY), [])
        assert structure == "pair"

    def test_validator_accepts_absent_without_evidence(self) -> None:
        assert _validate_stage1({"behaviors": [{"behavior_id": "x", "present": False}]}) == []
        assert _validate_stage1({"behaviors": [{"behavior_id": "x", "present": True}]}) == [
            "behaviors[0].evidence must be a list when present is not false"
        ]


class TestNormalizeStage2:
    def _obs(self) -> list[BehaviorObservation]:
        return [
            BehaviorObservation(
                behavior_id=AI_PRESENT,
                behavior_did="AiBehavior-901",
                label="a",
                kind="ai_behavior",
                turns=[2, 4, 12],
                uncertainty=0.1,
                group_id="g",
            ),
            BehaviorObservation(
                behavior_id=USER_DISTRESS,
                behavior_did="UserBehavior-901",
                label="b",
                kind="user_behavior",
                turns=[1],
                uncertainty=0.2,
                group_id="g",
            ),
        ]

    def test_supported_instance(self, kg: LibraryIndex, turns) -> None:
        entry = {
            "pathway_id": PW_TRAJECTORY,
            "instances": [
                {
                    "turn_window": [1, 4],
                    "support": "partially_supported",
                    "structure": "sequence",
                    "participants": [
                        {"behavior_id": USER_DISTRESS, "role": "condition", "turns": [1]},
                        {"behavior_id": AI_PRESENT, "role": "originator", "turns": [2, 4]},
                    ],
                    "uncertainty": 0.4,
                    "rationale": "both participants",
                }
            ],
        }
        a = normalize_stage2(
            entry,
            kg,
            kg.resolve(PW_TRAJECTORY),
            self._obs(),
            turns,
            "why",
        )
        assert a.pathway_id == PW_TRAJECTORY
        assert a.instances[0].turn_window == [1, 4]
        inst = a.instances[0]
        assert {p.behavior_id for p in inst.participants} == {AI_PRESENT, USER_DISTRESS}
        assert not next(
            p for p in inst.participants if p.behavior_id == USER_DISTRESS
        ).is_participant
        assert any("not a defined participant" in n for n in inst.validation_notes)

    def test_missing_participant(self, kg: LibraryIndex, turns) -> None:
        entry = {
            "pathway_id": PW_TRAJECTORY,
            "instances": [],
            "no_instance": {
                "support": "not_supported",
                "missing_participants": [AI_PRESENT, USER_DISTRESS],
                "rationale": "not observed",
            },
        }
        a = normalize_stage2(entry, kg, kg.resolve(PW_TRAJECTORY), [], turns, "r")
        assert a.missing_participants == [AI_PRESENT]  # a gate is not a participant

    def _pathway_barring_distress(self, tmp_path) -> LibraryIndex:
        import pathlib
        import shutil

        root = tmp_path / "kl"
        shutil.copytree(pathlib.Path("tests/fixtures/kl"), root)
        pathway = root / "entries" / "pathways" / "kl-900101.yaml"
        pathway.write_text(
            pathway.read_text(encoding="utf-8").replace(
                "    role: originator\n",
                "    role: originator\n"
                "  - kind: user_behavior\n    id: kl-900501\n    role: initiator\n"
                "    absent: true\n",
            ),
            encoding="utf-8",
        )
        clear_cache()
        return LibraryIndex(load(root))

    def test_a_barred_behavior_inside_the_window_is_not_supported(self, tmp_path, turns) -> None:
        """Stage 1 observing a must-not-occur behavior inside the window overrides support.

        The model left that behavior out of the participants and called the instance
        supported. The window is the participants' span, so a turn between them counts.
        """
        kg = self._pathway_barring_distress(tmp_path)
        obs = [
            BehaviorObservation(
                behavior_id=AI_PRESENT,
                behavior_did="AiBehavior-901",
                label="a",
                kind="ai_behavior",
                turns=[2, 4],
                uncertainty=0.1,
                group_id="g",
            ),
            BehaviorObservation(
                behavior_id=USER_DISTRESS,
                behavior_did="UserBehavior-901",
                label="b",
                kind="user_behavior",
                turns=[3],
                uncertainty=0.1,
                group_id="g",
            ),
        ]
        entry = {
            "pathway_id": PW_DIRECT,
            "instances": [
                {
                    "support": "supported",
                    "participants": [
                        {"behavior_id": AI_PRESENT, "role": "originator", "turns": [2, 4]},
                    ],
                    "rationale": "the resource is named",
                }
            ],
        }
        a = normalize_stage2(entry, kg, kg.resolve(PW_DIRECT), obs, turns, "why")
        inst = a.instances[0]
        assert inst.support == "not_supported"
        assert a.support == "not_supported"
        assert inst.evidence_gaps == [
            "UserBehavior-901 must NOT occur and Stage 1 observed it at turn 3 inside this instance"
        ]
        assert (
            "support 'supported' replaced by not_supported: a must-not-occur behavior "
            "occurs inside this instance"
        ) in inst.validation_notes

    def test_citing_a_barred_behavior_as_a_participant_is_not_supported(
        self, tmp_path, turns
    ) -> None:
        kg = self._pathway_barring_distress(tmp_path)
        obs = [
            BehaviorObservation(
                behavior_id=AI_PRESENT,
                behavior_did="AiBehavior-901",
                label="a",
                kind="ai_behavior",
                turns=[2],
                uncertainty=0.1,
                group_id="g",
            ),
            BehaviorObservation(
                behavior_id=USER_DISTRESS,
                behavior_did="UserBehavior-901",
                label="b",
                kind="user_behavior",
                turns=[2],
                uncertainty=0.1,
                group_id="g",
            ),
        ]
        entry = {
            "pathway_id": PW_DIRECT,
            "instances": [
                {
                    "support": "supported",
                    "participants": [
                        {"behavior_id": AI_PRESENT, "role": "originator", "turns": [2]},
                        {"behavior_id": USER_DISTRESS, "role": "initiator", "turns": [2]},
                    ],
                }
            ],
        }
        a = normalize_stage2(entry, kg, kg.resolve(PW_DIRECT), obs, turns, "why")
        inst = a.instances[0]
        assert inst.support == "not_supported"
        assert any("must NOT occur in this pathway" in n for n in inst.validation_notes)
        assert (
            "UserBehavior-901 must NOT occur and Stage 1 observed it at turn 2"
            in (inst.evidence_gaps[0])
        )

    def test_a_barred_behavior_outside_the_window_leaves_support(self, tmp_path, turns) -> None:
        kg = self._pathway_barring_distress(tmp_path)
        obs = [
            BehaviorObservation(
                behavior_id=AI_PRESENT,
                behavior_did="AiBehavior-901",
                label="a",
                kind="ai_behavior",
                turns=[2],
                uncertainty=0.1,
                group_id="g",
            ),
            BehaviorObservation(
                behavior_id=USER_DISTRESS,
                behavior_did="UserBehavior-901",
                label="b",
                kind="user_behavior",
                turns=[1],
                uncertainty=0.1,
                group_id="g",
            ),
        ]
        entry = {
            "pathway_id": PW_DIRECT,
            "instances": [
                {
                    "support": "supported",
                    "participants": [
                        {"behavior_id": AI_PRESENT, "role": "originator", "turns": [2]},
                    ],
                }
            ],
        }
        a = normalize_stage2(entry, kg, kg.resolve(PW_DIRECT), obs, turns, "why")
        inst = a.instances[0]
        assert inst.support == "supported"
        assert inst.evidence_gaps == []
        assert a.support == "supported"


class TestGates:
    turns: ClassVar[list[Turn]] = [
        Turn(turn_number=1, role="user", content="My therapist has me doing ERP for the OCD."),
        Turn(turn_number=2, role="assistant", content="That sounds like hard work."),
    ]

    def _gates(self, kg: LibraryIndex, facts: frozenset[str]) -> list[GateInput]:
        return gate_inputs(kg, PW_TRAJECTORY, facts)

    def test_the_record_decides_a_context_value_gate(self, kg: LibraryIndex) -> None:
        recorded = self._gates(kg, frozenset({CONTEXT_DISTRESS}))
        assert [(g.key, g.part.ref, g.recorded) for g in recorded] == [
            ("G1", USER_DISTRESS, "unknown"),
            ("G2", CONTEXT_DISTRESS, "holds"),
        ]
        on_other_value = self._gates(kg, frozenset({"kl-900603"}))
        assert on_other_value[1].recorded == "does_not_hold"
        assert self._gates(kg, frozenset())[1].recorded == "unknown"

    def test_verdict_takes_the_record_and_flags_a_conflict(self, kg: LibraryIndex) -> None:
        entry = {
            "gates": [
                {
                    "gate": "G2",
                    "cues": [{"turn": 2, "quote": "hard work", "reading": "contradicts"}],
                    "inferred": "does_not_hold",
                    "rationale": "r",
                }
            ]
        }
        gates, status = normalize_gates(
            entry, kg, self._gates(kg, frozenset({CONTEXT_DISTRESS})), self.turns
        )
        g2 = gates[1]
        assert (g2.recorded, g2.inferred, g2.verdict, g2.basis) == (
            "holds",
            "does_not_hold",
            "holds",
            "test_case",
        )
        assert g2.conflict
        assert g2.cues[0].speaker == "assistant"
        assert gates[0].verdict == "unknown"
        assert "no gate assessment returned" in gates[0].validation_notes[0]
        assert status == "unknown"

    def test_the_conversation_decides_when_nothing_is_recorded(self, kg: LibraryIndex) -> None:
        cue = {"turn": 1, "quote": "doing ERP for the OCD", "reading": "supports"}
        entry = {
            "gates": [
                {"gate": "G1", "cues": [cue], "inferred": "holds"},
                {"gate": "G2", "cues": [cue], "inferred": "holds"},
            ]
        }
        gates, status = normalize_gates(entry, kg, self._gates(kg, frozenset()), self.turns)
        assert [(g.verdict, g.basis, g.conflict) for g in gates] == [
            ("holds", "conversation", False),
            ("holds", "conversation", False),
        ]
        assert status == "met"

    def test_an_inference_needs_a_verified_cue(self, kg: LibraryIndex) -> None:
        entry = {
            "gates": [
                {
                    "gate": "G2",
                    "cues": [
                        {"turn": 1, "quote": "I am in exposure therapy", "reading": "supports"}
                    ],
                    "inferred": "holds",
                },
                {"gate": "G1", "cues": [], "inferred": "does_not_hold"},
            ]
        }
        gates, status = normalize_gates(entry, kg, self._gates(kg, frozenset()), self.turns)
        assert [g.inferred for g in gates] == ["unknown", "unknown"]
        assert gates[1].cues == []
        assert any("not verbatim" in n for n in gates[1].validation_notes)
        assert status == "unknown"

    def test_a_gate_that_does_not_hold_means_not_met(self, kg: LibraryIndex) -> None:
        gates, status = normalize_gates({}, kg, self._gates(kg, frozenset({"kl-900603"})), [])
        assert gates[1].verdict == "does_not_hold"
        assert status == "not_met"

    def test_a_pathway_without_gates(self, kg: LibraryIndex) -> None:
        assert normalize_gates({}, kg, gate_inputs(kg, PW_DIRECT, ()), []) == ([], "no_gates")


class TestFolding:
    def test_turn_major_derives_present(self, kg: LibraryIndex, turns) -> None:
        group = DefinitionGroup(
            group_id="g",
            kind="ai_behavior",
            labels=["x"],
            behavior_ids=[AI_PRESENT, AI_ABSENCE],
        )
        parsed = {
            "turns": [
                {
                    "turn": 1,
                    "verdicts": [
                        {
                            "behavior_id": AI_PRESENT,
                            "present": True,
                            "quote": "Call the Northside Nightline at 555-0100",
                            "uncertainty": 0.1,
                            "rationale": "t1",
                        }
                    ],
                },
                {
                    "turn": 2,
                    "verdicts": [
                        {
                            "behavior_id": AI_PRESENT,
                            "present": True,
                            "quote": "try the Harbor Wellness Clinic on Maple",
                            "uncertainty": 0.1,
                            "rationale": "t2",
                        }
                    ],
                },
            ],
            "whole_conversation": [
                {"behavior_id": AI_ABSENCE, "present": False, "reason": "asked"}
            ],
        }
        folded = fold_turn_major(parsed, kg, group, turns)
        by_id = {b["behavior_id"]: b for b in folded["behaviors"]}
        assert by_id[AI_PRESENT]["present"] is True
        assert by_id[AI_PRESENT]["turns"] == [1, 2]
        assert by_id[AI_ABSENCE]["present"] is False

    def test_verdict_anchor_keeps_present_verdict(self, kg: LibraryIndex, turns) -> None:
        group = DefinitionGroup(
            group_id="g",
            kind="ai_behavior",
            labels=["x"],
            behavior_ids=[AI_PRESENT],
        )
        parsed = {
            "verdicts": [
                {
                    "behavior_id": AI_PRESENT,
                    "present": True,
                    "uncertainty": 0.1,
                    "rationale": "named a clinic",
                }
            ],
            "anchors": [
                {
                    "behavior_id": AI_PRESENT,
                    "turns": [
                        {
                            "turn": 1,
                            "present": True,
                            "quote": "Call the Northside Nightline at 555-0100",
                        }
                    ],
                }
            ],
        }
        folded = fold_verdict_anchor(parsed, kg, group, turns)
        assert folded["behaviors"][0]["present"] is True
        assert folded["behaviors"][0]["turns"] == [1]


class TestStage2Robustness:
    def test_validator_is_structural_only(self) -> None:
        ok = {"assessments": [{"pathway_id": PW_TRAJECTORY, "instances": [{"support": "weird"}]}]}
        assert _validate_stage2(ok) == []
        assert _validate_stage2({"assessments": []}) == ["`assessments` must be a non-empty list"]

    def test_json_repair_of_stray_quote_and_trailing_comma(self) -> None:
        raw = (
            '{"assessments": [{"pathway_id": "kl-900102", "instances": [],"},'
            '{"pathway_id": "kl-900101", "instances": [],}]}'
        )
        parsed = parse_json_lenient(raw)
        assert parsed is not None
        assert [a["pathway_id"] for a in parsed["assessments"]] == ["kl-900102", "kl-900101"]
        assert parse_json_lenient("not json at all") is None


class TestCandidates:
    def test_only_pathways_with_observed_participants(self, kg: LibraryIndex) -> None:
        obs = [
            BehaviorObservation(
                behavior_id=AI_PRESENT,
                behavior_did="AiBehavior-901",
                label="a",
                kind="ai_behavior",
                turns=[2],
                uncertainty=0.1,
                group_id="g",
            )
        ]
        cands, skipped = select_candidates(kg, obs)
        cand_ids = {pw.id for pw, _ in cands}
        assert cand_ids == set(kg.pathways_for_behaviors({AI_PRESENT}))
        withheld_pathways = sum(
            1 for nid in kg.withheld if (n := kg.get(nid)) is not None and n.type == "pathway"
        )
        assert len(cands) + len(skipped) == len(kg.pathways()) + withheld_pathways
        assert all("AiBehavior-901" in reason for _, reason in cands)

    def test_no_observations_skips_everything(self, kg: LibraryIndex) -> None:
        cands, skipped = select_candidates(kg, [])
        assert cands == []
        withheld_pathways = sum(
            1 for nid in kg.withheld if (n := kg.get(nid)) is not None and n.type == "pathway"
        )
        assert len(skipped) == len(kg.pathways()) + withheld_pathways

    def test_a_withheld_pathway_is_reported_as_a_skip(self, tmp_path) -> None:
        """Withholding that nothing records is the failure this policy prevents.

        A withheld pathway never reaches the candidate loop, so without an
        explicit skip it would vanish from the accounting entirely -- present
        in neither candidates nor skips, and absent from the coverage totals.
        """
        import pathlib
        import shutil

        root = tmp_path / "kl"
        shutil.copytree(pathlib.Path("tests/fixtures/kl"), root)
        pathway = root / "entries" / "pathways" / "kl-900101.yaml"
        pathway.write_text(
            pathway.read_text(encoding="utf-8").replace("kl-900301", "kl-999999"),
            encoding="utf-8",
        )
        clear_cache()
        kg = LibraryIndex(load(root))
        assert "kl-900101" in kg.withheld  # precondition, not the assertion

        cands, skipped = select_candidates(kg, [])

        withheld_skips = [s for s in skipped if s.pathway_id == "kl-900101"]
        assert len(withheld_skips) == 1
        assert "withheld" in withheld_skips[0].reason
        assert "impacts" in withheld_skips[0].reason
        assert "kl-900101" not in {pw.id for pw, _ in cands}


class TestRunMocked:
    async def test_full_run(self, kg: LibraryIndex, run_case, turns, manifest) -> None:
        mock = MockJudge()
        with patch(_PATCH, new=mock):
            run = await run_kl_two_stage_judge(
                run_case,
                turns,
                manifest,
                manifest.judges[0].model,
                {"mode": "mock", "group_size": 8, "max_concurrency": 4},
            )

        assert isinstance(run, CoralMindRun)
        assert run.run_status == "success"
        meta = run.run_meta
        assert meta.mode == "mock"
        assert meta.transcript_hash == transcript_hash(turns)
        assert meta.kl.digest == kg.digest
        assert meta.kl.version == kg.version
        assert meta.model == "mock/judge"
        assert len(meta.prompt_hash) == 64

        s1 = run.stage1
        n_groups = len(build_definition_groups(kg, "ai_behavior", 8)) + len(
            build_definition_groups(kg, "user_behavior", 8)
        )
        assert len(s1.calls) == n_groups
        assert s1.coverage["ai_behavior"]["evaluated"] == s1.coverage["ai_behavior"]["total"] == 2
        assert s1.coverage["user_behavior"]["evaluated"] == 1
        found = {o.behavior_id: o for o in s1.observations}
        assert set(found) == set(MOCK_OBSERVATIONS)
        assert found[AI_PRESENT].turns == [1, 2]
        assert all(e.verified for e in found[AI_PRESENT].evidence)
        assert s1.integrity["definitions_in_calls"] == 3
        assert s1.integrity["definitions_present"] == 2
        assert s1.integrity["definitions_absent"] == 1
        assert s1.integrity["definitions_unassessed"] == 0
        assert found[USER_DISTRESS].kind == "user_behavior"
        per_turn = {(t.turn, t.role): t for t in s1.per_turn}
        assert per_turn[(1, "user")].behavior_ids == [USER_DISTRESS]
        assert per_turn[(1, "assistant")].behavior_ids == [AI_PRESENT]
        assert per_turn[(2, "user")].behavior_ids == []
        assert per_turn[(2, "assistant")].behavior_ids == [AI_PRESENT]

        expected = set(kg.pathways_for_behaviors({AI_PRESENT, USER_DISTRESS}))
        assessed = {a.pathway_id for a in run.stage2.assessments}
        assert assessed == expected
        assert meta.counts["skipped"] == len(kg.pathways()) - len(expected)
        assert run.stage2.integrity["candidates"] == len(expected)
        assert run.stage2.integrity["assessed"] == len(assessed)
        assert run.stage2.integrity["pathways_missing"] == 0
        assert meta.counts["pathways_missing"] == 0
        pw = next(a for a in run.stage2.assessments if a.pathway_id == PW_TRAJECTORY)
        assert pw.instances
        assert {p.behavior_id for p in pw.instances[0].participants} == {AI_PRESENT}
        assert [(g.part_id, g.inferred, g.verdict) for g in pw.gates] == [
            (USER_DISTRESS, "holds", "holds"),
            (CONTEXT_DISTRESS, "holds", "holds"),
        ]
        assert pw.gates[1].basis == "conversation"
        assert pw.gate_status == "met"
        assert meta.facts == ["kl-900602"]

    async def test_recorded_facts_decide_the_gate(self, run_case, turns, manifest) -> None:
        """A counterfactual whose record says the gate fails is not met, whatever the text says."""
        with patch(_PATCH, new=MockJudge()):
            run = await run_kl_two_stage_judge(
                run_case,
                turns,
                manifest,
                manifest.judges[0].model,
                {"mode": "mock"},
                facts={"kl-900603"},
            )
        pw = next(a for a in run.stage2.assessments if a.pathway_id == PW_TRAJECTORY)
        g2 = pw.gates[1]
        assert (g2.recorded, g2.inferred, g2.verdict, g2.conflict) == (
            "does_not_hold",
            "holds",
            "does_not_hold",
            True,
        )
        assert pw.gate_status == "not_met"
        assert run.run_meta.facts == ["kl-900603"]
        assert run.stage2.integrity["gate_conflicts"] == 1

    async def test_withheld_pathway_is_recorded_in_run_coverage(
        self, run_case, turns, manifest, tmp_path
    ) -> None:
        """A dangling structural edge must appear in run_meta.coverage.

        The clean-fixture full run never withholds anything, so it cannot
        catch a writer that drops these keys.
        """
        import pathlib
        import shutil

        root = tmp_path / "kl"
        shutil.copytree(pathlib.Path("tests/fixtures/kl"), root)
        pathway = root / "entries" / "pathways" / "kl-900101.yaml"
        pathway.write_text(
            pathway.read_text(encoding="utf-8").replace("kl-900301", "kl-999999"),
            encoding="utf-8",
        )
        clear_cache()
        kg = LibraryIndex(load(root))
        assert "kl-900101" in kg.withheld  # precondition, not the assertion

        mock = MockJudge()
        with patch(_PATCH, new=mock):
            run = await run_kl_two_stage_judge(
                run_case,
                turns,
                manifest,
                manifest.judges[0].model,
                {"mode": "mock", "group_size": 8, "max_concurrency": 4},
                kg=kg,
            )

        assert "kl-900101" in run.run_meta.coverage["withheld_entry_ids"]
        assert "unresolved_cross_references" in run.run_meta.coverage

    async def test_failed_batch_is_split_and_retried(self, run_case, turns, manifest) -> None:
        mock = MockJudge(scripted={"coralmind_s2_": ["garbage", "garbage"]})
        with patch(_PATCH, new=mock):
            run = await run_kl_two_stage_judge(
                run_case, turns, manifest, manifest.judges[0].model, {"pathways_per_call": 0}
            )
        calls = run.stage2.calls
        assert calls[0].status == "judge_error"
        assert "batch split and retried" in (calls[0].error or "")
        assert len(run.stage2.assessments) == run.run_meta.counts["candidates"]

    async def test_dry_run_samples_pathways(self, run_case, turns, manifest) -> None:
        token = set_dry_run_context(DryRunCollector())
        try:
            run = await run_kl_two_stage_judge(
                run_case, turns, manifest, manifest.judges[0].model, {"mode": "mock"}
            )
        finally:
            reset_dry_run_context(token)
        assert run.run_status == "dry_run"

    def test_merge_uses_kl_pin(self, run_case, turns, manifest) -> None:
        from corallab.judging.actions.merge_judge_runs import MergeJudgeRuns

        def _result(status: str) -> JudgePassResult:
            run = CoralMindRun.model_validate(
                {
                    "run_status": status,
                    "run_meta": {
                        "run_id": "r",
                        "started_at": "t",
                        "finished_at": "t",
                        "elapsed_ms": 1,
                        "mode": "mock",
                        "test_case_id": run_case.test_case_id,
                        "transcript_hash": "h",
                        "transcript_turns": 4,
                        "kl": {"version": "0.0.1", "digest": "sha256:abc"},
                        "model": "m",
                        "prompt_version": "latest",
                        "prompt_hash": "ph",
                    },
                }
            )
            return JudgePassResult(judge_type="kl_two_stage", coralmind_run=run)

        base = {
            "run_case": run_case,
            "transcript": make_transcript(run_case, manifest, turns=turns),
            "manifest": manifest,
            "evaluation_id": "e1",
            "evaluation_level": 3,
            "judge_pass_inputs": [],
        }
        ok = MergeJudgeRuns().run(State({**base, "judge_pass_results": [_result("success")]}))[
            "final_result"
        ]
        assert ok.coralmind_run is not None
        assert ok.status == "success"

    def test_manifest_without_options_still_loads(self) -> None:
        from corallab.schemas.eval.run_manifest import JudgeConfig

        cfg = JudgeConfig.model_validate(
            {
                "judge_type": "kl_two_stage",
                "model": {"model_id": "mock/judge"},
            }
        )
        assert cfg.options == {}


def test_registry_builds_and_exposes_the_two_stage_judge() -> None:
    """REGISTRY contains the two-stage judge."""
    from corallab.judging.judges.registry import REGISTRY

    assert set(REGISTRY) == {"kl_two_stage"}


def test_library_body_withholds_internal_fields() -> None:
    """Underscore-prefixed keys are not judge input."""
    from corallab.judging.judges.kl_two_stage import format_behavior_definition
    from corallab.judging.kl_index import load_library_index

    kg = load_library_index()
    node = kg.node("kl-900001")  # the fixture entry carries a `_note` key

    assert [k for k in node.body if k.startswith("_")] == []
    assert "internal annotation" not in format_behavior_definition(kg, node, full=True)


def test_steps_in_one_exchange_put_the_persons_message_first(kg: LibraryIndex) -> None:
    parts = [
        SupportingBehavior(
            behavior_id=AI_PRESENT, behavior_did="AiBehavior-901", role="originator", turns=[1]
        ),
        SupportingBehavior(
            behavior_id=USER_DISTRESS, behavior_did="UserBehavior-901", role="initiator", turns=[1]
        ),
    ]
    raw = [
        {"behavior_id": AI_PRESENT, "turn": 1, "role": "originator"},
        {"behavior_id": USER_DISTRESS, "turn": 1, "role": "initiator"},
    ]
    steps = _instance_steps(raw, parts, kg, [])
    assert [(s.order, s.behavior_id) for s in steps] == [(1, USER_DISTRESS), (2, AI_PRESENT)]
    derived = _instance_steps(None, parts, kg, [])
    assert [s.behavior_id for s in derived] == [USER_DISTRESS, AI_PRESENT]
