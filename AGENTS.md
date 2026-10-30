# AGENTS.md

Project guidance for AI coding assistants working on the CoralMind Toolkit.

The installable package is `corallab`. Both the distribution name and the import name are `corallab`: `pip install corallab`, `import corallab`. The CLI is `corallab`. Env overrides stay `CORALMIND_TOOLKIT_ROOT` and `CORALMIND_DATA_REPOSITORY_PATH`.

The results viewer in `viewer/` is a Next.js app. Its ingest, the Python package `coralmind-viewer` in `viewer/ingest/`, reads `data-repository/results` into the app's page data. Do not fold either into `corallab`.

Modules are gerund-named. Clinical knowledge comes from the Knowledge Library (`coralmap`). Do not add a scraped `knowledge/` tree under the data repository. Do not create `library/`, `seeds/`, `selection.py`, or `seeding.py`.

---

## Licenses

Two licenses, on purpose:

- Code in `src/`, `viewer/`, `scripts/`, `tests/`, and the software documentation is MIT (`LICENSE.md`). So are `data-repository/run_manifests/` and `data-repository/endpoints/`. Keep the contributor patent grant there: it incorporates Apache License 2.0 section 3, and the copyright terms stay MIT.
- Test-case records, persona records, `data-repository/TEST_CASE_RULES.md`, and the published sample runs in `data-repository/sample_results/` are CC BY-NC 4.0 (`CONTENT_LICENSE.md`).

The wheel does not bundle those records, so the package license is MIT only. Do not add `data-repository/` to the wheel or the sdist without changing `license` to `MIT AND CC-BY-NC-4.0` and listing `CONTENT_LICENSE.md` in `license-files`. Do not relicense either set. The CoralMind name and logo are trademarks.

---

## Version and release

The open-source line is `0.1.0`. Stay on `0.x` until the public API is stable enough that a break would be `2.0.0`. `1.0.0` is that promise; do not use it for a first public upload.

The version is `pyproject.toml`, `src/corallab/__init__.py` (`__version__`), and `CITATION.cff`. Release Please updates all three from Conventional Commits. Do not bump them by hand.

The pull request title is the commit when the pull request is squash-merged. Otherwise each commit on `main` counts.

| Title | Before 1.0.0 | From 1.0.0 |
| --- | --- | --- |
| `fix:` | patch | patch |
| `feat:` | minor | minor |
| `feat!:` or `BREAKING CHANGE:` | minor | major |
| `docs:`, `chore:`, `ci:`, `test:`, `style:`, `refactor:` | no release | no release |

Release Please opens one pull request with the version and changelog. Merging it tags `vX.Y.Z` and `.github/workflows/release.yml` publishes `corallab` to PyPI. Do not add a second publish workflow, and do not commit a PyPI token. The viewer package `coralmind-viewer` is not part of that release.

To set the version, end a commit message with a `Release-As:` footer. Pre-releases use PEP 440, for example `Release-As: 0.2.0a1`, `0.2.0b1`, or `0.2.0rc1`. A squash merge keeps the branch's commit messages in its body, so put the footer on the branch's last commit. A `BEGIN_COMMIT_OVERRIDE` block in the pull request body replaces those messages and drops the footer. A later `fix:` does not step `a1` to `a2`; set the next `Release-As`. `pip install corallab` skips pre-releases. The stable release is `Release-As: 0.2.0`.

Release Please authenticates as a GitHub App (`RELEASE_PLEASE_CLIENT_ID`, `RELEASE_PLEASE_PRIVATE_KEY`) so the release pull request runs CI. A pull request opened with `GITHUB_TOKEN` would not.

`coralmap` is a runtime dependency (`coralmap>=0.1.0`), installed from PyPI. A judge run still records `kl.version` and `kl.digest` from the library it loaded; those are not this package's version.

---

## Task skills

Task skills live in `.agents/skills/`. Cursor and Claude Code see that directory through `.cursor/skills` and `.claude/skills`. Read the skill whose description matches the task.

The agent does not merge pull requests. No skill runs `gh pr merge`. A request to merge is answered by saying the maintainer merges it.

| Skill | Read it when |
| --- | --- |
| `commit` | Changes are ready to record on the current branch |
| `create-issue` | Work should be tracked on GitHub rather than done now |
| `create-pr` | A branch is ready for review and the user has asked for a pull request |
| `push` | Local commits need to reach the remote |
| `clean-branches` | Merged, stale, or gone branches need pruning |

## Rules

Rules live in `.agents/rules/`. Read the one whose trigger matches the task, and leave the others unread.

| Rule | Load when |
| --- | --- |
| `working-agreement` | Starting an implementation, refactor, or bugfix |
| `pixi-environments` | Setting up the repo, running a command, or changing a dependency |
| `pre-commit-and-quality` | About to commit, open a pull request, or claim that checks passed |
| `contribution-discipline` | About to commit or open a pull request |

---

## What this package does

The toolkit evaluates how AI models respond in mental-health contexts against increasingly realistic simulated users:

| Level | Name | Description |
| --- | --- | --- |
| **L1** | Scripted single-turn | Fixed prompt, scored response |
| **L2** | Scripted multi-turn | Predetermined dialogue |
| **L3** | Persona-prompted multi-turn | An LLM plays a clinically grounded persona |

Stages:

- **`simulation/`** — reads test case and persona records; Burr graphs for setup, L1/L2/L3 simulation, batch fan-out
- **`fidelity/`** — `ElicitationFidelityChecker`, run by `corallab check` on an L3 run's transcripts. Also hosts `consistency_checker`, which the sim loop does not call
- **`judging/`** — judge graph; the only judge is `kl_two_stage`
- **`validation/`** — empty; ground-truth sets and agreement metrics
- **`viewer/`** — the results viewer, a Next.js app: a Suite page with a test case file's test cases, a Run page with one run's transcripts, and a transcript view per conversation. See "Results viewer" below.

Prompts stay beside the code that parses them. Generated artifacts go under `CORALMIND_DATA_REPOSITORY_PATH` (default `./data-repository` in the checkout), never in `src/`.

---

## Records

- `RunManifest.test_case_records_path` names a JSONL file of `TestCaseRecord`.
- Each test case's `persona_ref` names `persona_records/<persona_id>.json` (`RunManifest.persona_records_path`, default `persona_records`).
- `corallab sim` writes `Transcript` records. `corallab check --run-manifest M --transcripts T` writes one `FidelityCheckRecord` per transcript to `<stem>-check.jsonl` beside them. `corallab judge --run-manifest M --transcripts T` joins them to the test case records.
- A test case pinned to a different Knowledge Library than the one loaded does not run.
- `corallab validate test-cases` checks a file of test case records; `corallab validate run-manifest` checks a manifest.
- `corallab validate test-case-consistency FILE --model M` (`authoring/`) reads every part of each test case (strategies, assertions, opening, scenario, persona) against its expected behaviors' catalog definitions, one model call per case, and flags a part that, as written, works against them. Run it while writing test cases, not per run. Its `risk` verdicts vary from run to run, so a person reviews what it flags.

---

## 4-layer eval architecture

```text
Layer 4  post-conversation evaluation (judges, elicitation fidelity)
Layer 3  conversation loop (send opening → SUT → exit → person message)
Layer 2  L3 simulation (PersonResponseGenerator)
Layer 1  data model & setup (test case record loader, runners)
```

Clinical knowledge belongs in the Knowledge Library, not hardcoded in actions. Simulation and judging are temporally separated.

The file-by-file path of `corallab sim` is `docs/sim-call-stack.md`. Read it before changing the sim CLI, the batch graph, or the L3 loop:

- `corallab sim` reads `system_under_test` and `simulation`. `judges` is for `corallab judge` and `corallab run`.
- Cases load once in `cli/_runners.py` `run_batch_sim`, then `grouping.iter_groups` slices them by `simulation.batch_size`. The sim batch graph is `simulate` → `write`. Those cases arrive as `run_cases`.
- L3 runs a nested Burr app (`send_opening` → `call_sut` → `check_exit` → `generate_persona`). `PersonResponseGenerator` (`simulation/person_response_generator.py`) writes the person's lines, calling `simulation.person_response_model`. The opening is `turns[0]` verbatim. Every later person turn gets the full `scenario_strategies` list and the assertion prose; `available_from_turn` and `trigger` are prompt text, not filters. `pathway_id` is not in that prompt. A transcript `turn_number` is that exchange: the user message and the assistant reply share it, and `available_from_turn` counts the person's messages.
- `simulation.elicitation_fidelity` configures the fidelity check (`ElicitationFidelityChecker` in `fidelity/elicitation.py`), which `corallab check` runs on a file of transcripts, and `corallab run` on each batch when `enabled` is true. One model call reads the transcript, `scenario_strategies`, `behavior_assertions`, and the catalog entry for each asserted behavior, and the check writes the `FidelityVerdict` to `<stem>-check.jsonl`; the transcript's own `fidelity` stays null. A strategy is judged on user turns at or after `available_from_turn`; a trigger that never happens is `trigger_absent` and does not change `status`. `catalog_occurred` is whether the transcript shows that catalog entry: the model answers each lettered part, and code decides with the judge's `required_elements` and `needs_separate_turns`. A miss is `not_attempted` and makes `status` `invalid`, unless it is a multi-turn behavior an earlier AI reply made unnecessary: then it is `preempted` and `status` is `not_tested`. `assertion_occurred` is whether the assertion prose shows up, and a miss does not change `status`. `enabled` defaults to true; an unset `model` reuses `person_response_model`. There is no per-turn consistency check. The check does not read `pathway_id`, the persona, or the scenario.
- A relative `run_dir` resolves under the data repository (`paths.resolve_run_dir`), then the CLI appends `-YYYYMMDD-HHMMSS` (and `-2`, `-3`, … when that folder already exists), so `results/foo` is written to `data-repository/results/foo-20261007-154700`. Test-case and persona paths resolve through `paths.resolve_data_path`.

---

## Results viewer

`pixi run -e viewer viewer-dev` ingests `data-repository/results`, starts `next dev` at <http://localhost:3000>, and re-ingests when a run changes. The ingest (`pixi run -e viewer viewer-ingest`) writes `viewer/src/generated/`, which is gitignored; pages read it only through `viewer/src/lib/artifacts.ts`.

- The results folder alone decides what the viewer shows: `data-repository/results` by default, or the folder `--results` names (`--results data-repository/sample_results` for lab.coralmind.org). There is one build: it lists only suites with runs, writes no absolute path, and copies the records that `/record` and `/judge-call` serve into `records/`, so the app never reads `data-repository/`.
- `pixi run -e viewer viewer-build` builds from `viewer/src/generated/` as it is and fails when nothing was ingested. `viewer-smoke` checks a build; `viewer-leaks` checks the ingested data for home paths, endpoint addresses and internal fields, and is not part of the build.
- The ingest works out every per-row fact (buckets, route outcomes, copied turns, quotes found); the app selects and counts. Page wording lives in TypeScript (`viewer/src/lib/labels.ts`); Knowledge Library text comes through the ingest.
- The page data's shape is `viewer/ingest/src/coralmind_viewer/export_models.py`. After changing it, run `pixi run gen-viewer-types`; `pixi run check` fails on drift.
- Tests: `pixi run -e viewer viewer-test` writes fixtures from `tests/viewer/scenarios.py` through the ingest, then runs vitest.
- lab.coralmind.org is the viewer built from `data-repository/sample_results/`. `.github/workflows/viewer.yml` tests and builds it, running `viewer-leaks` on the ingested samples before the build; it does not deploy. Publish a run with `pixi run publish-sample data-repository/results/<run>`; it copies only the files pages read. `pixi run check` fails when a sample no longer reads cleanly.
- A public clone cannot build the viewer. `viewer/package.json` pins `@coralmind/brand` from the private `coralmind/brand` repository, and CI rewrites that URL with `BRAND_READ_TOKEN`. Leave that pin. The Python package does not depend on it.

---

## Tech stack

Python 3.12, pixi, pydantic, pyyaml, litellm, Apache Burr, typer, pytest, ruff, mypy.

```bash
pixi install
pixi run check          # ruff + format + mypy + schemas + tests
pixi run -- corallab --help
```

Repo root override: `CORALMIND_TOOLKIT_ROOT`. Data repository override: `CORALMIND_DATA_REPOSITORY_PATH` (env → bundled package data → `./data-repository`). The endpoint registry is `endpoints/endpoints.yaml` inside that directory.

---

## Code conventions

### Class-based Burr actions

All pipeline actions inherit from Burr's `Action` with `reads` / `writes` / `run` / `update`. Actions live in `simulation/actions/`, `judging/actions/`, and `simulation/batch/`.

### Pydantic everywhere

- All data models are Pydantic `BaseModel` subclasses
- YAML/JSON configs go through `model_validate`
- JSON Schemas are generated by `scripts/gen_schemas.py` next to the data they validate

### General style

- ruff: line-length 100, Python 3.12
- `from __future__ import annotations` at the top of every module
- Built-in generics: `list[...]`, `dict[...]`, `X | None`
- Type-annotate function signatures
- `logging.getLogger(__name__)` — no `print()`

### Ruff lint rules

Canonical list is `pyproject.toml`. First-party package is `corallab`. Ignored: `UP037`, `RUF001/002/003` (clinical Unicode).

### Async / graph / prompts

- LLM-calling code is `async`; no blocking I/O inside async functions
- Graphs via `GraphBuilder` + `ApplicationBuilder`; parallelism via `MapStates` / `MapActions`
- Prompt templates live next to the action that consumes them and are versioned, not inlined

---

## Testing

- Tests use mocks for LLM calls; no real API calls in the default suite
- `asyncio_mode = "auto"`
- Tests load the synthetic library at `tests/fixtures/kl/` via `CORALMIND_KL_PATH` (set in `tests/conftest.py`). Do not load the real entries and do not mock `coralmap.load`.
- Record builders for tests live in `tests/_records.py`; the mock judge and mock test case are in `scripts/coralmind_mock_run.py`.

---

## LLM access

Every LLM call goes through `LLMClient.complete()` → `ModelRouter`. Model ids must be declared in `data-repository/endpoints/endpoints.yaml`. An unknown id raises `UnknownModelError`. Never commit `.env`.
