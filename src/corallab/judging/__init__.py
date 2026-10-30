"""Post-conversation judging.

Runs on a finished transcript. Passes are independent and run in parallel
via Burr ``MapStates``.

Top-level modules:

- ``kl_index``                    — Knowledge Library index the two-stage judge
                                    reads behaviors and pathways from.
- ``graph``                       — judge subgraph + judge-only pipeline
                                    builders.

Subpackages:

- ``judges/`` — polymorphic ``JudgeSpec`` ABC (``spec.py``), the one-entry
                dispatch ``REGISTRY`` (``registry.py``), the two-stage judge
                itself (``kl_two_stage``), shared prompt/JSON helpers
                (``_shared``), and prompt assets under ``data/coralmind/``.
- ``actions/`` — Burr actions that compose the judging phase
                 (``PrepareJudgePasses``, ``FanOutJudges``,
                 ``RunSingleJudgePass``, ``MergeJudgeRuns``).
"""
