# Pre-commit and Quality

**Load when:** about to commit, open a pull request, or claim that checks passed.

## Commands

```bash
# Once per clone
pixi run hooks-install

# Hooks over the whole tree
pixi run hooks

# What CI runs, before a pull request is prepared
pixi run check
```

`pixi run check` is the gate. Stop a pull request when it fails.

## When a hook rewrites a file

The hook reports that it modified files. Stage those files and run the hook
again. That rewrite is the hook doing its job.

```bash
git add <files>
pixi run hooks
```

## What the hooks cover

The hook set is `.pre-commit-config.yaml`. Read that file rather than a copy of
it. It includes YAML syntax, 4-space indentation for `data-repository/`,
codespell with persona and test-case records excluded, and secret detection.
mypy stays in `pixi run check`: an isolated pre-commit mypy cannot import
litellm, Burr, and coralmap without duplicating the pixi environment.

## Claiming a result

A claim that checks passed names the command and includes what it printed.
"Should pass" is not a result.
