---
name: commit
description: Use when changes are ready to record on the current branch.
---

# Commit

Commit staged and unstaged changes on the current branch.

## Instructions

1. Run these commands in parallel:

   - `git status` (never `-uall`)
   - `git diff --stat`
   - `git log --oneline -5`

2. Choose the conventional commit type from the release table below, and a scope
   when the change is local to one area. The subject is imperative and says why.
   Keep it under 72 characters.

3. Leave out secrets, `.env` files, and `.DS_Store`. If one of those is already
   staged, warn and unstage it.

4. Stage each included file by name. Do not use `git add -A` or `git add .`.

5. Commit with a HEREDOC. The subject is the conventional commit. The body says
   why.

   ```bash
   git commit -m "$(cat <<'EOF'
   type(scope): short imperative description

   Optional body explaining why.

   EOF
   )"
   ```

6. Run `git status` and confirm the commit succeeded.

7. If a hook fails, fix the problem and make a new commit. Amend only when the
   user explicitly asks. Never skip hooks.

## Release effect

On the current `0.x` line, a squash merge uses the pull request title as the
commit Release Please reads. This commit's subject should be a title that can
land that way.

| Type | Release |
| --- | --- |
| `feat` | minor |
| `fix` | patch |
| `feat!` or a `BREAKING CHANGE:` footer | minor, until 1.0.0 |
| `docs`, `chore`, `ci`, `test`, `style`, `refactor` | none |
| a `Release-As: X.Y.Z` footer | exactly `X.Y.Z` |

Scope is optional: `corallab`, `simulation`, `judging`, `fidelity`, `validation`,
`viewer`, `records`, `scripts`, `ci`, `docs`.

Do not edit `pyproject.toml`, `src/corallab/__init__.py`, or `CITATION.cff` to
bump the version. Release Please does that.

## Rules

- The message has no attribution trailer, no tool co-author line, and no
  sign-off from the agent.
- Force-push only when the user explicitly asks. Refuse a force-push to `main`.
- Do not commit files that look like credentials.
