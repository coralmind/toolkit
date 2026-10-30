# data-repository

Every authored artifact the CoralMind Toolkit evaluates against.
Code lives in `src/`, clinical knowledge lives in the Knowledge Library, and
everything a run *reads* lives here.

This directory lives in the toolkit repository. It is organized by role
rather than by pipeline stage, so each directory answers one question.

Test-case records, persona records, and `TEST_CASE_RULES.md` are
[CC BY-NC 4.0](../CONTENT_LICENSE.md). Run manifests, the endpoint registry,
and the generated schemas are MIT with the rest of the code. None of
this tree is in the `corallab` wheel.

## Directories

| Directory | Holds |
| --- | --- |
| `test_case_records/` | Authored test cases — the live surface an evaluation runs against — as JSONL of `TestCaseRecord`. `examples/` holds one-test-case examples at L1, L2 and L3. |
| `run_manifests/` | Run manifests: which models, which test case records, which persona directory, which judge fan-out. |
| `persona_records/` | Persona records, one `p-*.json` file per person and version (`PersonaRecord`), which test case records name by `persona_ref`. The `depression/` and `ocd/` folders are empty placeholders. |
| `endpoints/endpoints.yaml` | Endpoint and model registry — environment routing, not data. Only env-var *names* appear here; secrets stay in the environment. |

`TEST_CASE_RULES.md` says how a persona or test case record is written and what is checked. The records are updated alongside Knowledge Library releases, and `CHANGELOG.md` records what changed. The pipeline runs on these records: `corallab sim` reads the test case records and their persona records, and a test case pinned to a different library version does not run.

The examples in `test_case_records/examples/` are `l1_launch_smoke.jsonl`, `l2_launch_smoke.jsonl` and `l3_launch_smoke.jsonl`: one test case each, which exercise the pipeline. Their six run manifests (`l{1,2,3}_{local,azure}.yaml`) live in `run_manifests/` and set `persona_records_path` to the persona directory their test cases name. Each manifest starts with a `yaml-language-server` schema comment so the editor completes keys against `run_manifests/schema/run_manifest.schema.json`.

Generated JSON Schemas sit in a `schema/` directory beside the data they
validate (`test_case_records/schema/`, `run_manifests/schema/`, and so on), because the
`$schema` directives in the authored YAML reference them by relative path.
Regenerate with `pixi run gen-schemas`; `pixi run check-schemas` fails on drift.

## Run output

Simulation and judging write under `results/`, resolved from the manifest's
`run_dir`. A relative `run_dir` of `results/my-run` is
`data-repository/results/my-run`. That directory is gitignored. The viewer
(`pixi run -e viewer viewer-dev`) reads it.

## Writing paths inside a manifest

A manifest's `test_case_records_path` is resolved by
`corallab.paths.resolve_data_path`, which looks under this directory.
Write it relative to *this* root, with no prefix:

```yaml
test_case_records_path: test_case_records/ocd_priya_l1.jsonl
```

A leading `data-repository/` segment is dropped, so a path copied from the
repository root still resolves. New manifests should omit it and write the
path relative to this directory.

The data root can be pointed elsewhere entirely with `CORALMIND_DATA_REPOSITORY_PATH`.
