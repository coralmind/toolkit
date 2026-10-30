"""In-memory lookups over a loaded Knowledge Library for the two-stage judge.

Walks entry fields and resolves ids through ``KnowledgeLibrary.entry`` rather
than the library's query surface (``leading_to``, ``involving``, ``PathwayView``),
so the judge can resolve display ids and aliases and withhold an entry whose
structural edges do not resolve.

The published body view comes from the library's ``public_fields(mode="json")``,
not from a local reconstruction of the underscore-is-internal convention.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coralmap import KnowledgeLibrary
from coralmap.models import Entry, observations

from corallab.kl import load_library

logger = logging.getLogger(__name__)

BEHAVIOR_TYPES: tuple[str, ...] = ("ai_behavior", "user_behavior")

#: Edge fields whose target must resolve before an entry is judged.
#: ``distinguish_from`` is omitted: an unresolved neighbour note does not
#: change what the entry asserts.
STRUCTURAL_EDGE_FIELDS: frozenset[str] = frozenset(
    {
        "composition_parts",
        "impacts",
        "composed_of",
        "cluster_candidate",
        "condition_tags",
        "experience_clusters",
        "dimension",
    }
)

#: ``validate()`` writes "Display-1 (kl-000001): field names 'kl-000002', which
#: is not ...". Parsed rather than re-derived from the models, because
#: ``validate()`` is the library's own account of its graph and duplicating its
#: traversal here would be a second implementation to keep in step.
_PROBLEM = re.compile(r"^\S+ \((?P<carrier>kl-\d{6})\): (?P<field>[a-z_]+) names ")


def _composition_parts(entry: Entry) -> tuple[Any, ...]:
    body = getattr(entry, "body", None)
    return tuple(getattr(body, "composition_parts", ()) or ())


def _composing_behaviors(entry: Entry) -> tuple[Any, ...]:
    """The behavior parts that make up a pathway's occurrence, in authored order.

    Excludes gates (``role: condition``), parts that must not occur (``absent``),
    and owed parts, which name no entry.
    """
    body = getattr(entry, "body", None)
    if body is None or not hasattr(body, "composition_parts"):
        return ()
    return tuple(
        part
        for part in observations(body)
        if part.kind != "context_value" and not part.absent and part.id != "OWED"
    )


def _context_parts(entry: Entry) -> tuple[Any, ...]:
    """A pathway's parts that name a context value: the facts a test case must hold."""
    return tuple(part for part in _composition_parts(entry) if part.kind == "context_value")


def _as_id(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("kl-"):
        return value
    if isinstance(value, dict):
        found = value.get("id") or value.get("replaces")
        return found if isinstance(found, str) else None
    return None


@dataclass(frozen=True)
class LibraryNode:
    """Judge-facing view of one library entry."""

    id: str
    did: str
    label: str
    definition: str
    type: str
    aliases: tuple[str, ...] = ()
    body: dict[str, Any] = field(default_factory=dict)
    section: str | None = None
    contested: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_entry(cls, entry: Entry) -> LibraryNode:
        body = entry.body.public_fields(mode="json") if hasattr(entry, "body") else {}
        contested = entry.contested
        contested_text = contested.summary if contested is not None else None
        return cls(
            id=entry.id,
            did=entry.display_id,
            label=entry.label,
            definition=entry.definition,
            type=entry.type,
            aliases=tuple(entry.aliases),
            body=body,
            contested=contested_text,
        )


@dataclass(frozen=True)
class PathwayPart:
    """One of a pathway's ``composition_parts``, resolved for the judge.

    ``node`` is ``None`` for an owed part, which names no entry; ``owed`` then
    says what the part must be. ``part`` is the library's own part, which
    :func:`coralmap.models.holds_for_test_case` takes for a context-value gate.
    """

    kind: str
    role: str | None
    absent: bool
    node: LibraryNode | None
    owed: str | None
    part: Any

    @property
    def is_gate(self) -> bool:
        """The part must hold for the pathway to apply, rather than composing it."""
        return self.role == "condition"

    @property
    def ref(self) -> str:
        """The part's entry id, or ``OWED`` for an owed part."""
        return self.node.id if self.node is not None else "OWED"


class LibraryIndex:
    """The questions the two-stage judge asks of a loaded library."""

    def __init__(self, library: KnowledgeLibrary) -> None:
        self.library = library
        self._by_id: dict[str, LibraryNode] = {}
        self._by_ref: dict[str, LibraryNode] = {}
        for entry in library.all_entries():
            node = LibraryNode.from_entry(entry)
            self._by_id[entry.id] = node
            self._by_ref[entry.id] = node
            self._by_ref[entry.display_id] = node
            for alias in entry.aliases:
                self._by_ref[alias] = node

        self.withheld: dict[str, str] = {}
        unresolved: list[str] = []
        for problem in library.validate():
            found = _PROBLEM.match(problem)
            if found is None:
                continue
            if found["field"] in STRUCTURAL_EDGE_FIELDS:
                self.withheld.setdefault(found["carrier"], problem)
            else:
                unresolved.append(problem)
        self.unresolved_cross_references: tuple[str, ...] = tuple(unresolved)
        if self.withheld:
            logger.warning(
                "Withholding %d entr%s from judging: an edge in %s does not resolve. "
                "They stay resolvable for cross-references but are not judged.",
                len(self.withheld),
                "y" if len(self.withheld) == 1 else "ies",
                ", ".join(sorted(STRUCTURAL_EDGE_FIELDS)),
            )

    @property
    def version(self) -> str:
        return self.library.version

    @property
    def digest(self) -> str:
        return self.library.digest

    def node(self, node_id: str) -> LibraryNode:
        return self._by_id[node_id]

    def get(self, node_id: str) -> LibraryNode | None:
        return self._by_id.get(node_id)

    def resolve(self, ref: str) -> LibraryNode | None:
        """Resolve a ``kl-`` id, display id, or alias."""
        return self._by_ref.get(ref)

    def behaviors(self, kind: str) -> list[LibraryNode]:
        if kind not in BEHAVIOR_TYPES:
            raise ValueError(f"kind must be one of {BEHAVIOR_TYPES}, got {kind!r}")
        entries = self.library.behaviors if kind == "ai_behavior" else self.library.user_behaviors
        return [self._by_id[e.id] for e in entries if e.id not in self.withheld]

    def pathways(self) -> list[LibraryNode]:
        return [self._by_id[p.id] for p in self.library.pathways if p.id not in self.withheld]

    def distinguished_from(self, node_id: str) -> list[tuple[LibraryNode, str | None]]:
        entry = self.library.entry(node_id)
        out: list[tuple[LibraryNode, str | None]] = []
        for row in entry.distinguish_from:
            neighbor = self.resolve(row.neighbor)
            if neighbor is not None:
                out.append((neighbor, row.boundary))
        return out

    def parts(self, pathway_id: str) -> list[PathwayPart]:
        """Every part of a pathway, in authored order, with role, negation and owed text.

        A part naming an entry that does not resolve is left out; the pathway
        itself is then withheld from judging.
        """
        entry = self.library.entry(pathway_id)
        out: list[PathwayPart] = []
        for part in _composition_parts(entry):
            owed = part.id == "OWED"
            node = None if owed else self.resolve(part.id)
            if node is None and not owed:
                continue
            out.append(
                PathwayPart(
                    kind=part.kind,
                    role=getattr(part, "role", None),
                    absent=bool(part.absent),
                    node=node,
                    owed=getattr(part, "owed_behavior", None) if owed else None,
                    part=part,
                )
            )
        return out

    def participants(self, pathway_id: str) -> list[tuple[LibraryNode, str | None]]:
        """The behaviors that make up an occurrence of the pathway, with their roles.

        Gates, parts that must not occur and owed parts are not participants;
        :meth:`parts` returns them.
        """
        entry = self.library.entry(pathway_id)
        out: list[tuple[LibraryNode, str | None]] = []
        for part in _composing_behaviors(entry):
            node = self.resolve(part.id)
            if node is None:
                continue
            out.append((node, getattr(part, "role", None)))
        return out

    def gates(self, pathway_id: str) -> list[PathwayPart]:
        """The parts that must hold for the pathway to apply (``role: condition``).

        The same parts :func:`coralmap.models.preconditions` returns.
        """
        return [p for p in self.parts(pathway_id) if p.is_gate]

    def pathways_for_behaviors(self, behavior_ids: set[str]) -> dict[str, set[str]]:
        """Pathways with at least one participant among ``behavior_ids``, and which ones."""
        hits: dict[str, set[str]] = {}
        for pathway in self.library.pathways:
            part_ids = {part.id for part in _composing_behaviors(pathway)}
            found = part_ids & behavior_ids
            if found:
                hits[pathway.id] = found
        return hits

    def related(self, node_id: str, relation: str) -> list[tuple[LibraryNode, dict[str, Any]]]:
        entry = self.library.entry(node_id)
        body = getattr(entry, "body", None)
        targets: list[Any] = []
        if relation == "reference_behavior" and body is not None:
            targets = [getattr(body, "reference_behavior", None)]
        elif relation == "candidate_cluster" and body is not None:
            targets = [getattr(body, "experience_cluster_candidate", None)]
        elif relation == "runs_mechanism" and body is not None:
            targets = list(getattr(body, "mechanisms", ()) or ())
        elif relation == "moderated_by" and body is not None:
            targets = [{"id": part.id, "absent": part.absent} for part in _context_parts(entry)]
        elif relation == "produces_impact" and body is not None:
            targets = [edge.impact for edge in getattr(body, "impacts", ()) or ()]
        out: list[tuple[LibraryNode, dict[str, Any]]] = []
        for raw in targets:
            nid = _as_id(raw)
            if nid is None:
                continue
            node = self.resolve(nid)
            attrs = raw if isinstance(raw, dict) else {}
            if node is not None:
                out.append((node, attrs))
        return out

    def impacts(self, pathway_id: str) -> list[tuple[LibraryNode, dict[str, Any]]]:
        return self.related(pathway_id, "produces_impact")

    def mechanisms(self, pathway_id: str) -> list[tuple[LibraryNode, dict[str, Any]]]:
        return self.related(pathway_id, "runs_mechanism")

    def moderators(self, pathway_id: str) -> list[tuple[LibraryNode, dict[str, Any]]]:
        """The context values that must hold for the pathway to apply."""
        return self.related(pathway_id, "moderated_by")


def load_library_index(path: str | Path | None = None) -> LibraryIndex:
    """Load the Knowledge Library and wrap it.

    Goes through :func:`corallab.kl.load_library`, so the load honours
    ``CORALMIND_KL_PATH`` and fails on any entry that does not parse.
    """
    return LibraryIndex(load_library(path))
