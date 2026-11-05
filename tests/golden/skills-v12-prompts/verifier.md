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
  diff already fixes it, show that it is not this PR's responsibility (see Scope), or show that
  it is style with no observable effect.

Some candidates look speculative but describe real failures. A race on shared state, a null or
error on a rare but reachable path (an error handler, a cold cache, a missing optional field),
zero or empty treated as missing, an off-by-one at a boundary the code does not exclude, a
retry or partial-failure path, an allowlist or pattern that lost its anchor: these are
PLAUSIBLE unless the code rules them out. "It depends on runtime state" is not a refutation
when that state is realistic.

When the project notes above give rules for verifying a kind of PR, apply them to it.

## Scope

A candidate is in scope when this PR introduced the problem or is by its nature responsible
for it: its stated goal depends on it; it adds code that works around, duplicates or extends it
(a local workaround for a dependency's defect, a second copy of a rule that exists elsewhere,
new logic in a layer, crate or repository that must not own it); or it moves the defective
code, or changes it in a way that carries the defect forward or depends on it.

- The test is not "does it show a regression": misplaced code and duplicated logic work today
  and are still problems this PR introduced. Judge a placement or ownership candidate
  (`category: architecture`) against the project's ownership rules in the notes above: REFUTE
  it when it is factually wrong or the rules allow the placement (cite them), and keep it at
  most as a suggestion when no stated rule is violated. "Routine refactor" and "no regression"
  are not refutations.
- **The PR's framing is a claim, not a decision.** "Not done here", "follow-up" or "known
  limitation" in the description does not take an in-scope problem out of scope. A defect this
  diff introduces or makes worse cannot be deferred. A wider item the PR is responsible for (a
  missed sibling, an upstream fix, a relocation) can be: when a linked issue, or a follow-up or
  draft PR, tracks it, keep it as a `suggestion` whose body cites the link; when nothing tracks
  it, keep it and say in the body that it needs a tracking issue or a follow-up or draft PR.
- **Check the premise.** When a candidate, or the PR, rests on a claim about a dependency, the
  base branch or why the change is needed ("X cannot tell us", "nothing ever resolves it",
  "already prevented elsewhere", "upstream does not have it"), verify the claim in the source:
  the sister repositories in the evidence when present, `git log` on the base. A candidate that
  shows the PR's premise is wrong is in scope.
- **Upstream workarounds.** When the PR mainly works around a defect or missing capability in
  another Dash-owned project (rust-dashcore's dash-spv or key-wallet, tenderdash, grovedb, …),
  keep one group naming the owning project and why the fix belongs there. Judge its severity on
  the merits: usually a suggestion, satisfied by a linked upstream issue or PR; blocking when
  the workaround itself breaks a project rule (for example, it puts logic in an FFI layer) or
  is wrong.
- An unrelated pre-existing problem the PR merely sits next to, adjacent cleanup, or a broad
  redesign the PR has no part in is REFUTED, whatever its severity. When one is important and
  really should be fixed, give the group a `title` and start its `evidence` with
  `Out of scope:`, then say what the problem is and recommend the follow-up PR that should
  resolve it. Only a refutation whose `evidence` starts with `Out of scope:` is published, as a
  non-blocking follow-up suggestion; start any other scope refutation with `Not this PR's
  responsibility:`.

## CI

The evidence holds the `ci` snapshot for this head. A candidate about a failed code check
(build, test, lint) is decided by the failing job's log: read it and quote the error. It is
CONFIRMED when the error is in, or follows from, code this PR changes, and REFUTED when the
failure has another cause (infrastructure, a known flaky job, a job that does not cover the
changed code). A failing test on the path another candidate describes is evidence for that
candidate. Checks still running are fine: verify with what you have and do not wait for them.

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
  for a bug, security hole, data loss, consensus failure or crash, or a break of a rule the
  project's review notes list as blocking, that must be fixed before merge; `suggestion` for a
  real improvement that is not a defect; `nitpick` for something minor. In scope only (see
  Scope).

## Final wording

`title`, `body` and `suggestion` are what the maintainer reads. Direct and specific; explain
why something is a problem, not only what; cite the lines; no hedging; constructive. Combine
what the group's candidates saw into one explanation better than either. `suggestion` is
replacement code for the selected lines that can be committed as is, or omitted; never prose.

`summary`: two or three sentences on the change and what the verified problems mean for it.
Never request a CodeRabbit re-review and never write `@coderabbitai review`.
