---
name: clean-branches
description: Use when merged, stale, or gone branches need pruning.
---

# Clean Branches

Remove local and remote branches that are finished.

## Arguments

Optional: `all`, `stale`, `merged`, or a branch name. If omitted, show the
summary and ask which groups to remove.

## Instructions

1. Run these commands in parallel:

   - `git branch`
   - `git branch -r`
   - `git branch --merged main`
   - `git for-each-ref --sort=-committerdate --format='%(refname:short) %(committerdate:relative) %(upstream:track)' refs/heads/`

2. Group every branch except `main`:

   | Group | Meaning |
   | --- | --- |
   | Merged | Listed by `git branch --merged main` |
   | Stale | Last commit older than 30 days |
   | Orphaned remote | Remote branch with no local branch tracking it |
   | Orphaned local | Local branch whose remote is gone (`[gone]`) |
   | Active | Recent commits, and not merged |

3. With no argument, show the groups, with name, age, and local or remote
   status, and ask which groups to remove. Wait for the answer.

4. Before deleting, show the exact list and wait for confirmation.

5. Delete local branches with `git branch -d`. A squash-merged branch will not
   appear in `git branch --merged main`, and `-d` will refuse it. Before a
   force delete, confirm the pull request is merged
   (`gh pr list --state merged`) and that `git diff main...<branch>` has no
   unique work. Force-delete only with that confirmation.

6. Delete a remote branch with `git push origin --delete <name>`.

7. Finish with `git fetch --prune`. Show what was deleted and what remains.

## Rules

- Leave `main` alone.
- Do not force-delete without the confirmation in step 5.
- Do not merge a pull request. The maintainer merges it. This skill only
  removes branches whose work has already landed.
