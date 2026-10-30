"""Cross-cutting utilities shared across eval phases.

Holds generic async / data helpers that don't belong to any single phase
(setup, simulation, judging, batch). Code here should have no dependency on
``corallab.judging`` / ``corallab.simulation`` /
``corallab.simulation.batch``.
"""
