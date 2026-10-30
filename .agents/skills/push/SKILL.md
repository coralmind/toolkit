---
name: push
description: Use when local commits need to reach the remote.
---

# Push

Push the current branch to the remote.

## Instructions

1. Run `git status` (never `-uall`) and note the branch, whether the tree is
   dirty, and whether the branch tracks a remote.

2. Show the commits that will go up:

   ```bash
   git log --oneline origin/$(git branch --show-current)..HEAD
   ```

   If the remote branch does not exist yet, show the commits since the branch
   diverged from `main`.

3. If the tree is dirty, ask whether to commit first. Use the `commit` skill
   for that commit. Do not push a dirty tree unless the user says to push the
   commits that are already made.

4. Push.

   - No upstream yet: `git push -u origin <branch>`
   - Already tracking: `git push`

5. Confirm the push succeeded, and show the branch and the commits that went
   up.

## Rules

- Force-push only when the user explicitly asks.
- Refuse a force-push to `main`.
- If the push is rejected because the remote has new commits, suggest
  `git pull --rebase` and wait for the user before doing it.
- Do not merge a pull request. The maintainer merges it.
