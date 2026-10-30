"""Run-manifest loading, the L1/L2/L3 engine, and batch fan-out.

Modules:

- ``person_response_generator`` — LLM call that turns the person rendering
                                and scenario into the person's next message.
- ``graph``                   — L3 simulation graph builders
                                (``build_l3_sim_graph``, ``build_sim_subgraph``).
- ``actions/``                — Phase-level Burr action wrappers
                                (``RunSimulation``, ``SystemUnderTest``).
- ``actions/sim_loop/``       — Per-turn loop nodes used inside the L3
                                sim subgraph.
- ``setup/``                  — shared runners (SUT, L3 simulation).
- ``batch/``                  — load, parallel fan-out, write.
- ``data/``                   — test case record and persona loading.
- ``common/``, ``hooks/``     — shared eval utilities and Burr tracing.

The elicitation fidelity check and the per-turn consistency checker
live in ``corallab.fidelity``.
"""
