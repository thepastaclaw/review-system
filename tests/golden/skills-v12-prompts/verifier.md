# Verify and match review candidates

Independent reviewers raised the candidates below on pull request **#{pr_number}** in
**{repo}**, checked out here at `{head_sha}`. Your job, for every candidate:

1. decide whether it is real, from the code, not from the reviewer's confidence;
2. group the candidates that describe the same problem;
3. say what each problem is in this PR's issue ledger (new, or an issue already raised);
4. write the final finding a maintainer reads, in one voice.

Take the time to read what each claim depends on: the enclosing function, its callers, the
guards around it, the tests. The orchestrator turns your verdicts into the review: it decides
what is posted, what blocks, and the review's verdict.

{phase_note}

## The pull request

- Title: {pr_title}
- Review range: `git diff {coverage_from}..{head_sha}`

{pr_description}

## Project context

{project_skill}

{review_skill}

## Candidates

Each has an `id`; use it exactly. `failure_scenario` is the reviewer's account of how the code
goes wrong.

{candidates}

## CodeRabbit inline comments (by comment id)

These are another bot's open comments on this PR. Only use them to say that a candidate
restates one (`coderabbit:<id>`); they are not candidates themselves.

{coderabbit}

## The issue ledger

What this PR's earlier review rounds raised and where each issue stands: `open` (still
counting; `thread_lane` is what the thread lane decided this round, `last_message` the latest
word on its thread), `settled` (withdrawn, deferred or outdated: the people on the PR settled
it), and `fixed`.

{ledger}

## Evidence

{context}

## Verdict, per group

- `CONFIRMED`: you can name the inputs or state that trigger the failure and the wrong result
  it produces. Quote the line that shows it.
- `PLAUSIBLE`: the mechanism is real, but whether it triggers depends on something you cannot
  settle from the code (timing, environment, configuration, a caller outside this repository).
  Say in `confirm_by` what would confirm it. A blocker that is only PLAUSIBLE is published as a
  suggestion carrying that line.
- `REFUTED`: the code shows the claim is wrong. Quote the line that contradicts it, show the
  invariant or type that rules it out, cite the guard that already handles it, show that this
  diff already fixes it, show that it lies outside what this PR changes or needs, or show that
  it is style with no observable effect.

Some candidates look speculative but describe real failures. A race on shared state, a null or
error on a rare but reachable path (an error handler, a cold cache, a missing optional field),
zero or empty treated as missing, an off-by-one at a boundary the code does not exclude, a
retry or partial-failure path, an allowlist or pattern that lost its anchor: these are
PLAUSIBLE unless the code rules them out. "It depends on runtime state" is not a refutation
when that state is realistic.

For a backport of an upstream merge: a hunk that is missing because Dash lacks the code it
would edit is the gap the candidate describes, not a refutation. A declared omission (the PR
description, a `partial` subject, or a `not_for_dash` adaptation note) is evidence against it.
Upstream's own design choices are not findings; a backport must match upstream.

## Grouping and matching

- Candidates that describe the same defect (the same root cause, even at different lines or
  in different words) share one group. Different defects in the same lines do not.
- `match` says what the group's problem is in the ledger:
  - `open:<hash>`: an open issue says the same thing. Nothing new is posted; the issue stays
    open. Use it even when the thread lane decided FIXED: a reviewer finding it again on this
    head is evidence it is not fixed.
  - `settled:<hash>`: a settled issue says the same thing. It stays settled unless the code at
    its location changed since; the orchestrator checks that, so match it honestly either way.
  - `fixed:<hash>`: a fixed issue is back (a regression).
  - `coderabbit:<id>`: a CodeRabbit comment above says the same thing.
  - `new`: none of these.
- Severity: judge it yourself; the reviewers' severity is only a starting point. `blocking`
  for a bug, security hole, data loss, consensus failure or crash that must be fixed before
  merge; `suggestion` for a real improvement that is not a defect; `nitpick` for something
  minor. In-scope only: a pre-existing problem this PR does not touch or worsen is REFUTED as
  out of scope, whatever its severity.

## Final wording

`title`, `body` and `suggestion` are what the maintainer reads. Direct and specific; explain
why something is a problem, not only what; cite the lines; no hedging; constructive. Combine
what the group's candidates saw into one explanation better than either. `suggestion` is
replacement code for the selected lines that can be committed as is, or omitted; never prose.

`summary`: two or three sentences on the change and what the verified problems mean for it.
Never request a CodeRabbit re-review and never write `@coderabbitai review`.
