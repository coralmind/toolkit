# Working Agreement

**Load when:** starting an implementation, refactor, or bugfix.

## 1. Think before coding

State assumptions. If a request has two readings, present them and ask. If a
simpler approach does the job, say so. If something is unclear, name it and
ask before implementing.

## 2. Smallest change

Write the change that solves the request. Skip features, abstractions, and
configuration that were not asked for. If a short version does the same work,
use the short version.

## 3. Touch only the request

Match the surrounding style. Do not restyle, reformat, or refactor adjacent
code. If you notice unrelated dead code, mention it and leave it in place.
Remove imports, variables, and functions that this change made unused. Every
changed line traces to the request.

## 4. Name the check

For each step, name the command that will show it worked. For a code or
record change, that command is `pixi run check`.

## 5. Report what ran

Claim a result only after running the check and reading its output. If a
command failed, say so and include the relevant output. If a step was skipped,
say which one and why.
