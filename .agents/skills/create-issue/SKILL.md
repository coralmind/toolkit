---
name: create-issue
description: Use when work should be tracked on GitHub rather than done now.
---

# Create Issue

Open a GitHub issue using the form that matches the work.

This repository does not ship issue templates. Blank issues are disabled. The
forms live in `coralmind/.github` on `main`, under `.github/ISSUE_TEMPLATE/`.
Read the matching form when you run. `gh issue create --template` only sees
templates inside this repository, so pass `--title`, `--body`, and `--label`
yourself.

## Which form

| Work | Form file | Title | Labels |
| --- | --- | --- | --- |
| Code or tooling | `bug_report.yml` | `[Bug]: <short description>` | `bug`, `needs-triage` |
| Test-case records, persona records, or other evaluation content | `content_report.yml` | `[Content]: <short description>` | `content`, `needs-triage` |
| README, guide, or reference | `documentation.yml` | `[Docs]: <short description>` | `documentation`, `needs-triage` |
| A new capability or improvement | `feature_request.yml` | A short imperative title, under 70 characters. That form sets no prefix. | `enhancement`, `needs-triage` |

Issue titles follow that table. They are not conventional commits.

A security vulnerability or safety-sensitive report is not a public issue. Stop
and point to the private process in
<https://github.com/coralmind/.github/blob/main/SECURITY.md>.

## Instructions

1. If the request is too vague to title, ask before creating the issue.

2. Read the form from `main`:

   ```bash
   gh api "repos/coralmind/.github/contents/.github/ISSUE_TEMPLATE/<file>?ref=main" --jq .content | base64 -d
   ```

   Use the fields in that file. The table above is only the routing.

3. Search existing issues. If one already covers the same work, stop and return
   its URL.

   ```bash
   gh issue list --state all --search "<terms>" --limit 20
   ```

4. Build the body from the form. Each field's `label` is a heading, and the
   answer sits under it. Fill every required field. Include an optional field
   when there is something to say. Check a "Before you continue" box only when
   that box is on the form and the statement is true. The feature-request form
   has no search box; the search in step 3 still happens.

5. Create the issue and return its URL.

   ```bash
   gh issue create --title "<title>" --label "<label>" --body "$(cat <<'EOF'
   ### Field label

   Answer.

   EOF
   )"
   ```

   Pass every label the form names. Do not create labels. If GitHub rejects a
   label, create the issue without that label and say which one was skipped.

## Rules

- The draft excludes real patient data, identifiable crisis narratives, and
  other personal information.
- Omit sections the form does not have.
- Do not put vulnerability details or safety-sensitive content in a public
  issue.
