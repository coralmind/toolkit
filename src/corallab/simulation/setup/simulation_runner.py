"""L3 simulation of one test case record via the Burr conversation-loop graph."""

from __future__ import annotations

import logging
from typing import Any

from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.trace import SimulationResult
from corallab.schemas.eval.transcript import SimFailure

logger = logging.getLogger(__name__)

_MAX_DETAIL_CHARS = 500


def sim_result_to_turns(sim_result: SimulationResult) -> list[Turn]:
    """Convert SimulationResult.conversation_transcript to list[Turn].

    Raises:
        ValueError: If the simulation produced no turns (empty transcript).
    """
    if not sim_result.conversation_transcript:
        raise ValueError(
            f"Simulation for test_case_id={sim_result.test_case_id!r} produced "
            "an empty transcript. Cannot proceed to judge with zero turns."
        )
    return [
        Turn(turn_number=t.turn_number, role=t.role, content=t.content, timestamp=t.timestamp)
        for t in sim_result.conversation_transcript
    ]


def scenario_content_for(case: RunCase, manifest: RunManifest) -> dict[str, Any]:
    """What PersonResponseGenerator and the system under test are given for ``case``."""
    record = case.test_case
    content: dict[str, Any] = {
        "turns": record.turns,
        "scenario": record.scenario,
        "assertions": [a.assertion for a in record.behavior_assertions],
        "strategies": [s.model_dump() for s in record.scenario_strategies],
    }
    if manifest.simulation.inject_context_into_sut:
        rendering = case.persona.rendering
        content["target_system_prompt"] = "\n\n".join(
            [
                "The following is background information about the person you are speaking with.",
                f"ABOUT THIS PERSON:\n{rendering.persona_description}",
                f"CLINICAL CONTEXT:\n{rendering.clinical_context}",
            ]
        )
        logger.info(
            "inject_context_into_sut=True: persona and clinical context in the SUT system "
            "prompt for test_case_id=%s",
            case.test_case_id,
        )
    return content


async def _run_sim_graph(
    graph: Any, initial_state: dict[str, Any], app_id: str
) -> SimulationResult:
    """Build and run a simulation Burr sub-application, returning the SimulationResult."""
    from burr.core import ApplicationBuilder

    app: Any = await (
        ApplicationBuilder()
        .with_graph(graph)
        .with_entrypoint("send_opening")
        .with_state(**initial_state)
        .with_identifiers(app_id=app_id)
        .abuild()
    )
    _, _, final_state = await app.arun(halt_after=["assemble"])
    result: SimulationResult = final_state["sim_result"]
    return result


async def run_simulation(
    case: RunCase,
    manifest: RunManifest,
    evaluation_id: str,
) -> tuple[list[Turn], SimulationResult] | SimFailure:
    """Simulate ``case`` at L3: the persona's rendering talks to the system under test."""
    from corallab.llm_client import llm_client
    from corallab.simulation.graph import build_l3_sim_graph

    if manifest.simulation.person_response_model is None:
        return SimFailure(
            reason="error", detail="the run manifest has no simulation.person_response_model"
        )

    logger.info(
        "Starting L3 simulation: test_case_id=%s system_under_test_model=%s",
        case.test_case_id,
        manifest.primary_sut.model_id,
    )
    graph, initial_state = build_l3_sim_graph(
        persona=case.persona.rendering,
        scenario_content=scenario_content_for(case, manifest),
        run_config=manifest,
        sim_client=llm_client,
        sut_client=llm_client,
        test_case_id=case.test_case_id,
    )
    try:
        sim_result = await _run_sim_graph(
            graph, initial_state, app_id=f"sim-l3-{case.test_case_id}"
        )
        turns = sim_result_to_turns(sim_result)
    except Exception as exc:
        logger.error(
            "L3 simulation failed for test_case_id=%s: %s", case.test_case_id, exc, exc_info=True
        )
        return SimFailure(reason="error", detail=str(exc)[:_MAX_DETAIL_CHARS])

    logger.info(
        "L3 simulation complete: test_case_id=%s total_turns=%d exit=%s",
        case.test_case_id,
        sim_result.total_turns,
        sim_result.exit_reason,
    )
    return turns, sim_result
