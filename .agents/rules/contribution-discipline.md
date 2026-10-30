# Contribution Discipline

**Load when:** about to commit or open a pull request.

## Before a pull request

1. Search open and closed issues and pull requests. If the same work already
   exists, stop and return that URL.

   ```bash
   gh issue list --state all --search "<terms>" --limit 20
   gh pr list --state all --search "<terms>" --limit 20
   ```

2. Keep the pull request to one problem.

3. The title is a conventional commit, `type(scope): imperative summary`, under
   70 characters. On squash merge that title is the commit Release Please
   reads. Use the release table in the `commit` skill.

4. The body is the current template at `.github/pull_request_template.md` in
   `coralmind/.github`. Read it when preparing the pull request. Fill each kept
   section. Drop "How to test" and "Notes for reviewers" when they have nothing
   to say. The `create-pr` skill has the steps.

5. Show the title, the body, and the diff. Open the pull request only after
   the user approves that diff, and only when they asked for one.

6. Run `pixi run check` before preparing the pull request. The description
   reports a problem that was actually seen and checks that were actually run.

## Standing limits

- Do not merge a pull request. The maintainer merges it.
- Target branch is `main`.
- The draft excludes real patient data, identifiable crisis narratives, and
  other personal information.
- A security vulnerability or safety-sensitive report goes to the private
  process in <https://github.com/coralmind/.github/blob/main/SECURITY.md>, not
  into a pull request.
- Do not name the model or the harness.
- Leave `AGENTS.md`, `.agents/rules/`, and `.agents/skills/` as written unless
  the user asks to change them.
- Do not bump the version in `pyproject.toml`, `src/corallab/__init__.py`, or
  `CITATION.cff`. Release Please does that.
