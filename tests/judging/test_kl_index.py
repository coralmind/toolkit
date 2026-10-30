"""LibraryIndex over the synthetic Knowledge Library fixture."""

from __future__ import annotations

import pathlib
import shutil

import pytest
from coralmap import load
from coralmap.loader import clear_cache

from corallab.judging.kl_index import BEHAVIOR_TYPES, LibraryIndex, load_library_index


@pytest.fixture(scope="module")
def kg() -> LibraryIndex:
    clear_cache()
    return LibraryIndex(load())


def test_context_value_by_label_is_gone() -> None:
    assert not hasattr(LibraryIndex, "context_value_by_label")


def test_load_library_index_honours_env(kg: LibraryIndex) -> None:
    again = load_library_index()
    assert again.digest == kg.digest
    assert again.version == kg.version


def test_resolve_by_id_and_display_id(kg: LibraryIndex) -> None:
    node = kg.resolve("kl-900001")
    assert node is not None
    assert node.did == "AiBehavior-901"
    assert kg.resolve("AiBehavior-901") is node
    assert kg.resolve("does-not-exist") is None


def test_behaviors_and_pathways_match_fixture(kg: LibraryIndex) -> None:
    assert {n.id for n in kg.behaviors("ai_behavior")} == {"kl-900001", "kl-900002"}
    assert {n.id for n in kg.behaviors("user_behavior")} == {"kl-900501"}
    assert {n.id for n in kg.pathways()} == {"kl-900101", "kl-900102", "kl-900103"}


def test_behaviors_rejects_non_behavior_kind(kg: LibraryIndex) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        kg.behaviors("pathway")
    assert BEHAVIOR_TYPES == ("ai_behavior", "user_behavior")


def test_participants_are_the_composing_behaviors(kg: LibraryIndex) -> None:
    parts = kg.participants("kl-900102")
    assert [(n.id, role) for n, role in parts] == [("kl-900001", "originator")]


def test_gates_are_the_condition_parts_in_authored_order(kg: LibraryIndex) -> None:
    gates = kg.gates("kl-900102")
    assert [(g.ref, g.kind, g.absent) for g in gates] == [
        ("kl-900501", "user_behavior", False),
        ("kl-900601", "context_value", False),
    ]
    assert all(g.is_gate for g in gates)


def test_parts_keep_absent_and_owed_parts(tmp_path: pathlib.Path) -> None:
    """A part that must not occur, and an owed part, are parts but not participants."""
    root = tmp_path / "kl"
    shutil.copytree(pathlib.Path("tests/fixtures/kl"), root)
    pathway = root / "entries" / "pathways" / "kl-900101.yaml"
    pathway.write_text(
        pathway.read_text(encoding="utf-8").replace(
            "    role: originator\n",
            "    role: originator\n"
            "  - kind: user_behavior\n    id: kl-900501\n    role: initiator\n    absent: true\n"
            "  - kind: user_behavior\n    id: OWED\n    role: sustainer\n    owed: true\n"
            "    owed_behavior: The person keeps returning to the resource.\n",
        ),
        encoding="utf-8",
    )
    clear_cache()
    kg = LibraryIndex(load(root))

    parts = kg.parts("kl-900101")
    assert [(p.ref, p.role, p.absent, p.owed) for p in parts] == [
        ("kl-900001", "originator", False, None),
        ("kl-900501", "initiator", True, None),
        ("OWED", "sustainer", False, "The person keeps returning to the resource."),
    ]
    assert [n.id for n, _ in kg.participants("kl-900101")] == ["kl-900001"]
    assert "kl-900101" not in kg.pathways_for_behaviors({"kl-900501"})


def test_single_participant_pathway(kg: LibraryIndex) -> None:
    parts = kg.participants("kl-900101")
    assert len(parts) == 1
    assert parts[0][1] == "originator"


def test_pathways_for_behaviors(kg: LibraryIndex) -> None:
    hits = kg.pathways_for_behaviors({"kl-900001"})
    assert set(hits) == {"kl-900101", "kl-900102"}
    assert hits["kl-900101"] == {"kl-900001"}


def test_auxiliary_lookups_resolve(kg: LibraryIndex) -> None:
    impacts = kg.impacts("kl-900102")
    assert {n.type for n, _ in impacts} == {"impact"}
    mechs = kg.mechanisms("kl-900102")
    assert [n.type for n, _ in mechs] == ["mechanism"]
    mods = kg.moderators("kl-900102")
    assert [n.id for n, _ in mods] == ["kl-900601"]
    assert [attrs["absent"] for _, attrs in mods] == [False]


def test_participants_leave_out_context_values(kg: LibraryIndex) -> None:
    """A context value gates a pathway and is not a behavior taking part in it."""
    assert "kl-900601" not in {n.id for n, _ in kg.participants("kl-900102")}
    assert "kl-900601" not in kg.pathways_for_behaviors({"kl-900601"}).get("kl-900102", set())


def test_node_body_is_the_library_published_view() -> None:
    """The strip is the library's job, not ours.

    Pins that ``LibraryNode.body`` equals what ``public_fields(mode="json")``
    returns, so reintroducing a toolkit-side strip -- which would drift from
    the library's convention the moment it gains a field -- shows up here.
    """
    from coralmap import load

    from corallab.judging.kl_index import LibraryNode

    entry = load().entry("kl-900001")

    assert LibraryNode.from_entry(entry).body == entry.body.public_fields(mode="json")


def test_internal_keys_never_reach_a_node_body() -> None:
    """kl-900001 carries ``_note`` at entry level for exactly this check."""
    from coralmap import load

    from corallab.judging.kl_index import LibraryNode

    entry = load().entry("kl-900001")

    assert "_note" in entry.model_dump()
    assert not [k for k in LibraryNode.from_entry(entry).body if k.startswith("_")]


def test_an_entry_with_a_dangling_definitional_edge_is_withheld(
    tmp_path: pathlib.Path,
) -> None:
    """A pathway whose impact edge does not resolve is not judged.

    Its harm claim is incomplete, and judging it anyway is how a pathway
    silently drops out of a downstream count -- which is the failure the
    library's own loader docstring warns about.
    """
    root = tmp_path / "kl"
    shutil.copytree(pathlib.Path("tests/fixtures/kl"), root)
    pathway = root / "entries" / "pathways" / "kl-900101.yaml"
    pathway.write_text(
        pathway.read_text(encoding="utf-8").replace("kl-900301", "kl-999999"),
        encoding="utf-8",
    )
    clear_cache()

    kg = LibraryIndex(load(root))

    assert "kl-900101" in kg.withheld
    assert "impacts" in kg.withheld["kl-900101"]
    assert "kl-900101" not in {p.id for p in kg.pathways()}
    # Still resolvable: another entry may point at it, and that reference has
    # to render even though the pathway is not judged.
    assert kg.resolve("kl-900101") is not None
    assert kg.node("kl-900101").did == "Pathway-901"


def test_a_coherent_library_withholds_nothing(kg: LibraryIndex) -> None:
    """The fixture is clean, so the policy is inert on it.

    Guards against a filter that withholds on some condition other than an
    unresolved edge -- which would quietly shrink judge input everywhere.
    """
    assert kg.withheld == {}
    assert kg.unresolved_cross_references == ()


def test_a_dangling_prose_cross_reference_is_logged_not_withheld(
    tmp_path: pathlib.Path,
) -> None:
    """distinguish_from does not change what an entry asserts."""
    root = tmp_path / "kl"
    shutil.copytree(pathlib.Path("tests/fixtures/kl"), root)
    behavior = root / "entries" / "ai_behaviors" / "kl-900001.yaml"
    # kl-900001 has no distinguish_from at all, so the block is added rather
    # than edited. Verified to produce:
    #   AiBehavior-901 (kl-900001): distinguish_from names 'kl-999999', ...
    behavior.write_text(
        behavior.read_text(encoding="utf-8").replace(
            "type: ai_behavior\n",
            "type: ai_behavior\ndistinguish_from:\n- neighbor: kl-999999\n  boundary: probe\n",
        ),
        encoding="utf-8",
    )
    clear_cache()

    kg = LibraryIndex(load(root))

    assert kg.withheld == {}, "a prose cross-reference is not a structural edge"
    assert any("kl-900001" in ref for ref in kg.unresolved_cross_references)
