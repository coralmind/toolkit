---
name: create-pr
description: Use when a branch is ready for review and the user has asked for a pull request.
---

# Create PR

Open a pull request for the current branch after the user has asked for one and
has approved the diff.

## Instructions

1. Run these commands in parallel:

   - `git status` (never `-uall`)
   - `git diff --stat`
   - `git branch --show-current`
   - `git log --oneline main..HEAD`
   - `git diff main...HEAD --stat`

2. If the head and base would both be `main`, stop.

3. If anything is uncommitted, ask whether to commit first. Use the `commit`
   skill for that commit.

4. Run `pixi run check`. If it fails, stop and report the output.

5. Search open and closed issues and pull requests. If the same work already
   exists, stop and return that URL.

   ```bash
   gh issue list --state all --search "<terms>" --limit 20
   gh pr list --state all --search "<terms>" --limit 20
   ```

6. Read the current pull request template:

   ```bash
   gh api "repos/coralmind/.github/contents/.github/pull_request_template.md?ref=main" --jq .content | base64 -d
   ```

7. Draft the pull request.

   - **Title:** `type(scope): imperative summary`, under 70 characters. This
     title is the commit that lands when a maintainer squash-merges, and it is
     what Release Please reads. Use the release table in the `commit` skill.
   - **Body:** the template, with each kept section filled in. Drop "How to
     test" and "Notes for reviewers" when they have nothing to say. Include
     `Closes #N` when an issue exists.

8. Show the title, the body, and the diff. Wait for the user to approve that
   diff. Do not open the pull request before that approval, and do not open one
   the user did not ask for.

9. After approval, push, then create the pull request. Use
   `git push -u origin <branch>` when the branch has no upstream, and `git push`
   when it already tracks one.

   ```bash
   gh pr create --title "type(scope): description" --body "$(cat <<'EOF'
   <filled template>
   EOF
   )"
   ```

10. Return the pull request URL.

The public contributing guide says outside pull requests are not being accepted
yet. That does not stop a pull request the user asked for.

## Rules

- One problem per pull request.
- The description reports a problem that was actually seen and checks that were
  actually run.
- The draft excludes real patient data, identifiable crisis narratives, and
  other personal information. Security vulnerabilities and safety-sensitive
  content belong in the private process at
  <https://github.com/coralmind/.github/blob/main/SECURITY.md>.
- Do not name the model or the harness.
- Do not merge the pull request. The maintainer merges it.
- Target branch is `main`.
- Do not edit `pyproject.toml`, `src/corallab/__init__.py`, or `CITATION.cff`
  to bump the version.
