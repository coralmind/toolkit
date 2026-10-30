# CoralMind Toolkit

> **Content warning:** this repository discusses suicide, self-harm, psychiatric crisis, and other mental health harms in direct, clinical language. The test cases and persona records are constructed scenarios, and a run can produce transcripts about the same subjects.
>
> If you are thinking about suicide or are otherwise in crisis, contact local emergency services. In the United States, call or text **988** (Suicide & Crisis Lifeline), or chat at [988lifeline.org](https://988lifeline.org). Outside the United States, [findahelpline.com](https://findahelpline.com) lists free, confidential services by country. See also [SUPPORT.md](https://github.com/coralmind/.github/blob/main/SUPPORT.md).

Open-source tools that simulate conversations with an AI system and judge them against the Knowledge Library. The installable package is `corallab` (`import corallab`; CLI: `corallab`). Read a run with `pixi run -e viewer viewer-dev`.

Clinical taxonomies live in the Knowledge Library (`coralmap` on PyPI). The toolkit depends on it and stamps `kl.version` / `kl.digest` on judge runs. A run reads a JSONL file of test case records, each naming a persona record; a test case pinned to a different library version than the one loaded does not run. Modules are gerund-named (`simulation/`, `judging/`).

## Intended Use

The toolkit is for research on, and evaluation of, AI systems in mental
health contexts. It is not a clinical tool, a crisis service, or a medical
device, and nothing in it is medical advice.

- Do not use it to diagnose, screen or assess a person, or to make decisions
  about anyone's care.
- A test case states what a conversation should be checked for. It does not
  establish that any AI system behaves that way or that any harm occurred.

## Installing

```bash
pip install corallab
```

The package depends on `coralmap`, which brings the Knowledge Library. The
wheel is the simulator, the judges, and the command-line tool. Test cases,
persona records, run manifests, and the endpoint registry stay in this
repository under `data-repository/`.

A published install reads that tree from `CORALMIND_DATA_REPOSITORY_PATH`.
Clone the repository and set the variable to its `data-repository` directory
before `sim`, `check`, `judge`, or `run`:

```bash
git clone https://github.com/coralmind/toolkit.git
cd toolkit
export CORALMIND_DATA_REPOSITORY_PATH="$PWD/data-repository"
```

The worked example below uses those paths. From a pixi checkout the variable
is unnecessary: the package finds `data-repository/` beside the source tree.

### Development Environment

```bash
pixi install
cp .env.example .env
pixi run hooks-install
pixi run check
```

```bash
pixi run -- corallab --help
```

Pixi and a published install both take `coralmap` from PyPI.

## Commands

| Command | What it does |
| --- | --- |
| `sim` | Simulates every test case record and writes `Transcript` records under the data repository. Takes `--run-manifest`, `--dry-run`. |
| `check` | Runs the fidelity check on existing transcripts and writes its verdicts beside them. Takes `--run-manifest`, `--transcripts`. |
| `judge` | Runs the `kl_two_stage` judge on existing transcripts. Takes `--run-manifest`, `--transcripts`, `--dry-run`. |
| `run` | Runs `sim`, `check` and `judge` in one command. Takes `--run-manifest`, `--db-path`. It has no `--dry-run`. |
| `viewer-dev` | `pixi run -e viewer viewer-dev` serves the results viewer at <http://localhost:3000>: a Suite page with a test case file's test cases, a Run page with one run's transcripts, and a transcript view per conversation. It reads `data-repository/results` again when a run changes. |
| `validate test-cases` | Checks a file of test case records: the records, their personas, the library pin and the expected behaviors. |
| `validate run-manifest` | Checks a run manifest against `RunManifest`. |
| `validate test-case-consistency` | Reads every part of each test case (strategies, assertions, opening, scenario, persona) against its expected behaviors with a model (`--model`, one call per case) and flags a part that, as written, works against them. Run it while writing test cases, not per run. |
| `run-preview` | Draws the Burr graphs the commands execute. |
| `check-env` | Reports which environment variables the endpoint registry needs are set. |

## Worked Example

Validate the example test case records, simulate them, check and judge the transcripts:

```bash
pixi run -- corallab validate test-cases data-repository/test_case_records/examples/l3_launch_smoke.jsonl
pixi run -- corallab sim --run-manifest data-repository/run_manifests/l3_local.yaml
pixi run -- corallab check --run-manifest data-repository/run_manifests/l3_local.yaml \
  --transcripts data-repository/results/<run>/batch-001/<run>-sim.jsonl
pixi run -- corallab judge --run-manifest data-repository/run_manifests/l3_local.yaml \
  --transcripts data-repository/results/<run>/batch-001/<run>-sim.jsonl
pixi run -e viewer viewer-dev
```

`sim` writes its transcripts into the manifest's `run_dir` with `-YYYYMMDD-HHMMSS` appended, under `data-repository/` when that path is relative. Pass that file to `check` and `judge`, which write their results into the same directory. `pixi run -e viewer viewer-dev` shows the run. `validate test-cases` reads personas from `persona_records` unless you pass `--personas`. The examples `l1_launch_smoke.jsonl`, `l2_launch_smoke.jsonl` and `l3_launch_smoke.jsonl` are one test case each, derived from a launch record. Their manifests are `l{1,2,3}_{local,azure}.yaml`.

## Layout

- `src/corallab/` — `simulation/`, `fidelity/`, `judging/`, `validation/`, plus `schemas/`, `llm/`, `cli/`
- `viewer/` — the results viewer, a Next.js app with its Python ingest in `viewer/ingest/` (`coralmind-viewer`), which this release workflow does not publish. A public clone cannot build it: `viewer/package.json` pins `@coralmind/brand` to a commit of the private `coralmind/brand` repository, and CI rewrites that URL with `BRAND_READ_TOKEN`. The Python package does not use it. CoralMind builds the viewer from `data-repository/sample_results/` and publishes it at lab.coralmind.org. Publish a finished run with `pixi run publish-sample data-repository/results/<run>`
- `data-repository/` — test cases, run manifests, personas, endpoint routing, and `results/`; see its README

## Releasing

Versions follow [Conventional Commits](https://www.conventionalcommits.org/).
When a pull request is squash-merged, its title is the commit message;
otherwise each commit that lands on `main` counts.

Stay on `0.x` until the public API is stable enough that a break would be
`2.0.0`. `1.0.0` is that promise.

| Commit | Before 1.0.0 | From 1.0.0 |
|---|---|---|
| `fix:` | patch `0.1.0` → `0.1.1` | patch |
| `feat:` | minor `0.1.0` → `0.2.0` | minor |
| `feat!:` or `BREAKING CHANGE:` | minor | major |
| `docs:`, `chore:`, `ci:`, `test:`, `style:`, `refactor:` | no release | no release |

A breaking `refactor!:` bumps the minor component until 1.0.0. After a merge
to `main`, Release Please opens or updates a release pull request that sets
the version in `pyproject.toml`, `src/corallab/__init__.py`, and
`CITATION.cff`, and writes `CHANGELOG.md`. Merging that pull request tags
`vX.Y.Z`, publishes the GitHub release, and uploads the package to PyPI as
`corallab`.

To set a version, end a commit message on `main` with a `Release-As:`
footer. Release Please reads the footer only from a commit on `main`, not
from the release pull request, which it rewrites. A squash merge keeps the
branch's commit messages in its body, so put the footer on the branch's last
commit. A `BEGIN_COMMIT_OVERRIDE` block in the pull request body replaces
those messages and drops the footer.

Pre-releases use PEP 440 in that footer, for example
`Release-As: 0.2.0a1`, `0.2.0b1`, or `0.2.0rc1`. A later `fix:` does not step
`a1` to `a2`; set the next `Release-As`. `pip install corallab` skips
pre-releases. The stable release is `Release-As: 0.2.0`.

PyPI publishing uses a trusted publisher rather than a token: owner
`coralmind`, repository `toolkit`, workflow `release.yml`, environment
`pypi`. The GitHub environment `pypi` must exist before the first release.
Do not commit a PyPI token.

Release Please authenticates as a GitHub App so that its release pull request
runs CI. Install the app on this repository with read and write access to
contents and pull requests. Then set the repository variable
`RELEASE_PLEASE_CLIENT_ID` to the app's client ID and the secret
`RELEASE_PLEASE_PRIVATE_KEY` to its private key.

## Related Repositories

| Repository | Relationship |
|---|---|
| [`coralmind/knowledge-library`](https://github.com/coralmind/knowledge-library) | The clinical catalog this package loads as `coralmap` |
| [`coralmind/.github`](https://github.com/coralmind/.github) | Org-wide community files: Code of Conduct, contributing guide, security policy, support, and issue forms |

## Licensing

Code in `src/`, `viewer/`, `scripts/`, `tests/`, and the software
documentation is [MIT](LICENSE.md). So are
`data-repository/run_manifests/` and `data-repository/endpoints/`.
A contribution is licensed under MIT, and the contributor also grants a patent
license for it on the terms of Apache License 2.0 section 3.

The test-case records, persona records, `data-repository/TEST_CASE_RULES.md`,
and the sample runs in `data-repository/sample_results/` are
[CC BY-NC 4.0](CONTENT_LICENSE.md). They are not in the wheel, so the
package on PyPI is MIT only. The non-commercial terms apply to those
records wherever they are copied.

The CoralMind name and logo are trademarks of CoralMind and are not licensed
for reuse.

## Reporting a problem with the records

Open a [content report](https://github.com/coralmind/toolkit/issues/new/choose) if a test case or persona record is unrealistic, contradicts itself, portrays a condition or a group of people inaccurately, or checks for the wrong thing. Give the record's id as the item: a `test_case_id` (for example `aimhef-bd505036`) or a `persona_id` (for example `p-anx-01-nt`). For a problem in the code, use the bug report.

If the problem is in the Knowledge Library pathway a test case names, rather than in the test case itself, report it in the [Knowledge Library](https://github.com/coralmind/knowledge-library#reporting-a-problem-with-the-catalog).

If a persona or test case could identify a real person, report it privately as described in the [security policy](https://github.com/coralmind/.github/blob/main/SECURITY.md). The records are constructed scenarios, not case reports, so any resemblance to a real person is a problem we want to hear about.

Do not include personal information about anyone else. We review every report and reply.

## Contributing

Contribution guidelines are still being written, and until they are, this repository does not accept pull requests. Bug reports, content reports, and ideas are welcome as issues. If you would like to contribute expertise, such as your knowledge, data, tools, or experience, reach out through the [contact form on our website](https://coralmind.org/get-involved?sourceId=GitHub&involve=expertise#connect).

Community files are inherited from [`coralmind/.github`](https://github.com/coralmind/.github): Code of Conduct, contributing guide, security policy, support, issue forms, and pull request template.
