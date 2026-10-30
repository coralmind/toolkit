# `corallab sim` call stack

How `corallab sim --run-manifest <file>` turns a run manifest into transcript JSONL. Written for coding agents. The worked example is `data-repository/run_manifests/l3_launch_entrenchment_azure.yaml`.

`corallab sim` generates conversations. It reads `system_under_test` and `simulation`. The `judges` list is consumed later by `corallab judge` and `corallab run`.

## Command

From the repository root (inside `pixi shell`, drop the `pixi run --` prefix):

```bash
pixi run -- corallab sim \
  --run-manifest data-repository/run_manifests/l3_launch_entrenchment_azure.yaml
```

`--dry-run` counts prompt tokens and makes no API calls. `--log-level` is a root option (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

A relative `run_dir` is stored under the data repository, and each invocation appends `-YYYYMMDD-HHMMSS` so a later sim does not overwrite an earlier one. `results/l3-launch-entrenchment-small-azure` lands at `data-repository/results/l3-launch-entrenchment-small-azure-20261007-154700`. Read it with `pixi run -e viewer viewer-dev`.

## Worked example

`l3_launch_entrenchment_azure.yaml` is an L3 run:

- One system under test: `azure/gpt-5.4-mini` at temperature 0.7.
- Person response generator: the same model at temperature 0.9.
- `simulation.batch_size` is 5 and `simulation.max_turns` is 10.
- Inputs: `test_case_records/l3_launch_entrenchment_small.jsonl` (20 cases). Personas come from `persona_records/` (the manifest default).
- `elicitation_fidelity` names `azure/gpt-5.4-mini` at temperature 0.0. `corallab sim` does not read it; `corallab check` does.
- `inject_context_into_sut` defaults to false, so the system under test gets no persona or clinical system prompt.

Twenty cases and a batch size of 5 produce four batches, in file order. Each batch finishes and is written before the next starts. Inside a batch every case is in flight together. `batch_size` is the only concurrency cap.

Output for this manifest:

```text
data-repository/results/l3-launch-entrenchment-small-azure-20261007-154700/
  l3-launch-entrenchment-small-azure-20261007-154700.log
  inputs/                                          # copy of the manifest and the JSONL
  batch-001/
    l3-launch-entrenchment-small-azure-20261007-154700-sim.jsonl   # 5 Transcript records
    l3-launch-entrenchment-small-azure-20261007-154700-sim.burr.db
    l3-launch-entrenchment-small-azure-20261007-154700-batch-001.trace.jsonl
  batch-002/ ... batch-004/
```

The file stem is the timestamped directory name. The parent path `results/.../<stem>-sim.jsonl` is never written. Each transcript id is `{run_id}:{sut_output_key}:{test_case_id}:1`, for example `l3-launch-entrenchment-10-azure:azure-gpt-5-4-mini:aimhef-32aaeebe:1`.

## Call stack

```text
pyproject.toml  [project.scripts] corallab = corallab.cli:main
  cli/__init__.py                     main() → Typer; callback load_dotenv()
    cli/sim.py                        sim()
      cli/_helpers.py                 resolve_run_manifest, init_logging, copy_run_inputs
      cli/sim.py                      _execute_sim
        cli/_runners.py               run_batch_sim
          kl.py                       load_library → coralmap.load
          simulation/data/test_case_records.py
          grouping.py                 iter_groups, batch_directory
          cli/_runners.py             _run_sim_group → _build_and_run
            simulation/graph.py       build_sim_subgraph(level)
            simulation/batch/graph.py build_sim_batch_graph
            simulation/batch/parallel_sim.py
              simulation/actions/run_simulation.py
                simulation/setup/simulation_runner.py
                  simulation/graph.py build_l3_sim_graph   # nested Burr app
                    actions/sim_loop/*                     # the conversation
                    llm_client.py     LLMClient.complete → ModelRouter
            simulation/batch/write_sim_results.py
```

### 1. Process entry

`src/corallab/cli/__init__.py` builds the Typer app and registers `sim`. The root callback calls `load_dotenv()` before the subcommand runs, so `.env` is already loaded. Typer dispatches to `sim()` in `src/corallab/cli/sim.py`.

### 2. Manifest, directory, inputs

`sim()` calls helpers in `src/corallab/cli/_helpers.py`:

- `resolve_run_manifest` reads the YAML and validates it as `RunManifest` (`src/corallab/schemas/eval/run_manifest.py`). The `yaml-language-server` schema comment is ignored. Unknown keys fail validation. L3 requires `simulation.person_response_model`.
- `RunManifest.per_model_manifests()` returns one `(output_key, manifest)` pair per `system_under_test` entry. One entry means one run. Several entries fan out: each model gets its own stem `<run_stem>__<output_key>` and a `<run_stem>.index.json` catalog.
- `init_logging` creates `<run_dir>-<YYYYMMDD-HHMMSS>` and a log file named after that directory.
- `copy_run_inputs` copies the manifest and the test-case JSONL into `run_dir/inputs/`. Persona records are not copied.

`_execute_sim` then calls `asyncio.run(run_batch_sim(...))`.

### 3. Load cases, then slice into batches

`run_batch_sim` in `src/corallab/cli/_runners.py`:

1. `load_library()` in `src/corallab/kl.py` calls `coralmap.load`. Resolution is the argument, then `CORALMIND_KL_PATH`, then the library bundled in the installed package. Tests pin `CORALMIND_KL_PATH` to `tests/fixtures/kl/`.
2. `load_test_case_records` in `src/corallab/simulation/data/test_case_records.py` resolves `test_case_records_path` and `persona_records_path` through `paths.resolve_data_path` (`src/corallab/paths.py`): absolute paths stay absolute; a leading `data-repository/` is dropped; a relative path that exists under the data root (`CORALMIND_DATA_REPOSITORY_PATH`, else `./data-repository`) resolves there. Each JSONL line is a `TestCaseRecord`. `persona_ref` `p-ocd-17-nt@1` loads `persona_records/p-ocd-17-nt.json` and the file's `ref` must match. The library pin and `evaluation_level` must match the loaded library and the manifest. Every problem is collected and raised together as `TestCaseRecordsError`. Each success is a `RunCase` (test case plus persona).
3. `iter_groups` in `src/corallab/grouping.py` slices those cases by `simulation.batch_size`. `batch_directory` names `batch-001`, `batch-002`, and so on.

Each group is simulated by `_run_sim_group` and written before the next group starts.

### 4. The batch Burr app

`_run_sim_group` builds two graphs and runs the outer one.

Per-case graph, `build_sim_subgraph` in `src/corallab/simulation/graph.py`:

- L1/L2: a single `SystemUnderTest` node (`simulation/actions/system_under_test.py`). Scripted turns, one model call, no persona loop.
- L3: a single `simulate` node (`RunSimulation`).

Batch graph, `build_sim_batch_graph` in `src/corallab/simulation/batch/graph.py`:

```text
simulate (ParallelSim) → write (WriteSimResults)
```

`_build_and_run` starts at entrypoint `simulate` and halts after `write` (`simulate` when `--dry-run`). The cases already sliced in `run_batch_sim` arrive on the initial state as `run_cases`.

`_build_and_run` also installs:

- `TracingHook` (`simulation/hooks/tracing.py`), writing `<stem>.trace.jsonl` beside the batch output. LLM calls log through a context variable, so `LLMClient` does not take the hook as an argument.
- A `TokenAccumulator` context variable (`llm_client.py`). `_execute_sim` prints the merged summary after every batch.
- An `AsyncSQLitePersister` on `<stem>-sim.burr.db` in the batch directory.
- A local Burr tracker, project `corallab`.

### 5. Fan-out over the cases in the batch

`ParallelSim` (`simulation/batch/parallel_sim.py`) is a Burr `MapStates` action. `states()` yields one sub-state per `RunCase`, each with a fresh `evaluation_id`. `reduce()` collects one `Transcript` per sub-state, including conversations whose `sim_failure` is set.

Each sub-state runs the per-case graph. For L3 that starts at `RunSimulation` (`simulation/actions/run_simulation.py`), which calls `run_simulation` in `simulation/setup/simulation_runner.py`.

### 6. One L3 conversation

`run_simulation` builds a second, nested Burr application with `build_l3_sim_graph`. That app starts at `send_opening` and halts after `assemble`. It does not share the batch app's persister.

```text
send_opening → call_sut → check_exit
check_exit --(should_exit)--> assemble
check_exit --(continue)-----> generate_persona → call_sut   # loop
```

| Node | File | What it does |
| --- | --- | --- |
| `send_opening` | `simulation/actions/sim_loop/send_opening.py` | Appends `turns[0]` from the test case as a user turn. No model call. |
| `call_sut` | `simulation/actions/sim_loop/call_sut.py` | Sends the history to `primary_sut` (temperature 0.7, `max_tokens` default 4096). Appends the assistant reply with the same `turn_number` as the user message it answers, and increments `round_count`. |
| `check_exit` | `simulation/actions/sim_loop/check_exit.py` | Stops after a system-under-test reply. `round_count >= max_turns` records `exit_reason` `max_turns` and wins if the persona also set `conversation_closing` on that same exchange. Otherwise a `conversation_closing` signal records `conversation_closing`. `scenario_enacted` does not stop the loop. The system under test speaks last. |
| `generate_persona` | `simulation/actions/sim_loop/generate_persona_message.py` | Asks `PersonResponseGenerator` for the next user message. |
| `assemble` | `simulation/actions/sim_loop/assemble_sim_result.py` | Packs the history into a `SimulationResult`. |

`scenario_content_for` in `simulation_runner.py` gives the person side the test case's `scenario`, `turns`, behavior-assertion prose, and every scenario strategy. The person text is `case.persona.rendering`. Which test-case fields reach the person prompt, and which do not, is in the next section.

`max_turns` counts system-under-test replies (`round_count`), which is the exchange count. With `max_turns: 10` the loop stops after 10 assistant replies. The opening user message is turn 1, the same number as the first reply. `total_turns` is that exchange count.

A raised exception becomes a `Transcript` with `sim_failure` and no turns (`simulation/setup/helpers.py` `make_transcript`). A finished conversation becomes a `Transcript` whose `turns` are copied from `SimulationResult.conversation_transcript` and whose `exit_reason` is `max_turns` or `conversation_closing`. An L1/L2 transcript leaves `exit_reason` null.

### 7. Person responses and elicitation fidelity

`PersonResponseGenerator` is `simulation/person_response_generator.py`. It calls `simulation.person_response_model` (`azure/gpt-5.4-mini` at 0.9 in the example) and returns JSON `{message, conversation_closing, scenario_enacted}`. There is no per-turn consistency check.

`_build_system_prompt` rebuilds the system prompt on every generated person turn. The scenario, the assertions, and the strategies are the same list every time. The turn-budget sentence is the part that changes.

| Test-case field | In the person prompt | How |
| --- | --- | --- |
| `persona_ref` | The three prose fields of the loaded rendering | `p-ocd-17-nt@1` loads `persona_records/p-ocd-17-nt.json`. The prompt gets `persona_description`, `clinical_context`, and `behavioral_description`. `traits`, `states`, and `elevated` stay on the record and are not inserted. |
| `pathway_id` | No | Names the catalog pathway. `corallab validate test-cases` checks that `behavior_assertions` match that pathway's user-behavior parts plus `setup_behaviors`. The judge reads it later. The simulator does not look it up while writing lines. |
| `scenario` | Yes, in full, every generated turn | Under `=== SCENARIO ===`. |
| `behavior_assertions` | The `assertion` string only | Under `=== WHAT YOU DO IN THIS CONVERSATION ===`. The catalog id (`kl-000410`) is dropped. |
| `scenario_strategies` | Every strategy, every generated turn | Under `=== HOW THE CONVERSATION GOES ===`, one line each: `From your turn N, when <trigger>: <strategy>`. A trigger of `none` is written as `at any point`. `available_from_turn` and `trigger` are instructions to the model. The code does not drop a strategy whose turn has not arrived or whose trigger has not fired. |
| `turns[0]` | Not as a prompt | `send_opening` copies it verbatim as the first user turn. Strategies and assertions apply from the second person line on. |

The simulation leaves `Transcript.fidelity` null. The fidelity check is its own command, `corallab check --run-manifest M --transcripts T`, which `corallab run` also runs on each batch when `simulation.elicitation_fidelity.enabled` is true (the default). `ElicitationFidelityChecker` (`fidelity/elicitation.py`) makes one model call per transcript. The prompt is the transcript, every `scenario_strategy` (`strategy`, `available_from_turn`, `trigger`), every `behavior_assertion` (`behavior`, `assertion`), and the catalog entry for each asserted behavior (definition, criteria, exclusions, detection window). It does not include `pathway_id`, the persona, or the scenario. The model is `elicitation_fidelity.model` when set, otherwise `person_response_model`. `elicitation_fidelity.prompt_version` selects `fidelity/prompts/<version>.txt`; omitted uses the checker's current prompt.

The check writes each `FidelityVerdict` to `<stem>-check.jsonl` beside the transcripts. Each strategy becomes a `StrategyVerdict`: `followed` when a user turn at or after `available_from_turn` shows it, `trigger_absent` when the trigger never happens or the conversation never reaches that turn, and `not_followed` otherwise. A trigger of `none` is always in play once the window exists. Strategy adherence does not change `status`. Each assertion becomes a `BehaviorVerdict`. The checker loads that behavior's catalog entry and asks two questions. `catalog_occurred` is whether the transcript shows the behavior as the entry defines it, with `citations` on the user turns that show it. The model answers each lettered part of the criteria (`parts`), and code decides `catalog_occurred` with the judge's `required_elements` and `needs_separate_turns`. `assertion_occurred` is whether the assertion prose shows up anywhere in the transcript. A catalog miss sets `miss` to `not_attempted`, which makes `status` `invalid` and adds a `reasons` entry naming that behavior. An assertion miss does not change `status`. A multi-turn behavior that an earlier AI reply made unnecessary is `preempted` instead of `not_attempted`: the model names why (`preempted_by`: `resolved` when the AI met the need first, `derailed` when it misread the person) and quotes the AI turn (`ai_citation`), and `status` is `not_tested` unless another behavior is `not_attempted`. A failed model call leaves `fidelity` null.

Shared retry settings, when the manifest omits them: `retry_max` 2, `retry_backoff_ms` 1000.

### 8. Every model call

Person responses and the system under test both call `LLMClient.complete` in `src/corallab/llm_client.py`. One shared `llm_client` serves every role.

`complete` resolves the logical model id through `ModelRouter` (`src/corallab/llm`). `data-repository/endpoints/endpoints.yaml` maps `azure/gpt-5.4-mini` to the `azure-default` endpoint (`AZURE_API_BASE`, `AZURE_API_KEY`, API version `2024-12-01-preview`) and the LiteLLM string `azure/gpt-5.4-mini`. An id that is not listed raises `UnknownModelError`. Azure sends `max_tokens` as `max_completion_tokens`. Usage is recorded on the token accumulator and the trace JSONL.

Confirm which env vars the registry names with `corallab check-env`. Values are never printed.

### 9. Write

`WriteSimResults` (`simulation/batch/write_sim_results.py`) writes one JSON line per `Transcript` via `model_dump_json()`. The results viewer (`viewer/`, `pixi run -e viewer viewer-dev`) reads those files from `data-repository/results`. Sim does not write HTML.

## What an agent should not assume

- `judges` and each judge's `model` are unused during `corallab sim`. Judging is `corallab judge` (or stage 2 of `corallab run`), which joins the transcript JSONL back to the test case records.
- Cases load in `run_batch_sim` before the batch graph runs. The sim batch graph is `simulate` → `write`.
- `ParallelSim` concurrency is the whole batch. There is no second in-flight cap on the sim path.
- A relative `run_dir` is placed under the data repository by `resolve_run_dir`, then `init_logging` appends `-YYYYMMDD-HHMMSS`. Data inputs follow `resolve_data_path`, which prefers the data root over the working directory.
- Omitting `elicitation_fidelity` leaves the post-conversation check on. An unset `elicitation_fidelity.model` reuses `person_response_model`, including its temperature when that temperature is set. There is no per-turn consistency check.
- Every generated person turn is shown the full `scenario_strategies` list. Filtering by `available_from_turn` or `trigger` is not implemented.
- `simulation.elicitation_fidelity` judges `scenario_strategies` on user turns at or after `available_from_turn`, and each `behavior_assertion` twice, then writes `<stem>-check.jsonl`; `corallab sim` does not run it. `catalog_occurred` is whether the transcript shows the catalog entry. `assertion_occurred` is whether the assertion prose shows up. A strategy whose trigger never happens is `trigger_absent` and does not change `status`. A catalog miss is `miss` `not_attempted` and makes `status` `invalid`, or `preempted` and `not_tested` when an earlier AI reply made a multi-turn behavior unnecessary. An assertion miss does not. A failed check leaves `fidelity` null. The person prompt still shows every strategy on every generated turn.
