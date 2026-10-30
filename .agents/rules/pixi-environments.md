# Pixi Environments

**Load when:** setting up the repo, running a command, or changing a dependency.

## Configuration

Pixi configuration is `[tool.pixi]` in `pyproject.toml`.

- Channels stay `conda-forge`.
- Platforms stay `osx-arm64`, `osx-64`, and `linux-64`.
- Python is 3.12. `graphviz` is a conda dependency of the workspace.
- Environments are `default` (dev tools), `report` (dev tools plus the
  analysis stack in the `report` extra), and `viewer` (dev tools plus Node for
  the results viewer).

`pixi task list` is the full task list. The commands this rule names are:

- `pixi run check` — the CI check job: lint, format, types, schema drift,
  data-repository YAML indentation, and tests.
- `pixi run hooks-install` — install git hooks, once per clone
- `pixi run hooks` — run the hooks over the tree

## Commands

Development commands go through Pixi.

```bash
pixi install
pixi run <task>
```

Run `pixi install` before the first Pixi command in a fresh clone. It is safe
to run again.

`gh` is not a Pixi task. Call it directly.

## Dependencies

The published package installs with pip, and CI tests that path. Pip does not
need Pixi. `coralmap` installs from PyPI. Ask before adding a runtime
dependency: CI fails when the runtime set changes and the consumer-install
check was not updated with it.

Add a development dependency in both `[project.optional-dependencies].dev` and
`[dependency-groups].dev`, then lock it with Pixi. Commit `pixi.lock` when
dependencies change. Do not hand-edit the lockfile.
