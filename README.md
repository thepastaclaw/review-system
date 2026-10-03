# reviewsys

PastaClaw's PR review system: polls GitHub, schedules two-phase LLM reviews
(triage → cheap Phase-1 reviewers → verifier → blocker gate → Phase-2 reviewers →
final verifier), and posts one GitHub review per PR head with inline comments.

It replaces the crontab + `review_orchestrator.py` pipeline that lived in the
OpenClaw workspace. Design goals, in order: **no silent stalls**, small, typed,
tested, one daemon, one SQLite file, no lock files.

## Layout

| module | responsibility |
|---|---|
| `daemon.py` | single-threaded tick loop: ingest, notify, route, supersede, reap, schedule, watchdog, gc |
| `ingest.py` | GitHub GraphQL poll → `prs`/`heads`; notifications and inline-comment replies → `inbox` |
| `router.py` | inbox → priority heads (`@thepastaclaw review`, review_requested, replies under a bot finding) or own-PR comment batches → OpenClaw wake |
| `scheduler.py` | starts every eligible head (priority first; `max_runs` safety cap), debounce, single-flight per PR, retries with backoff, supersede/cancel |
| `reaper.py` | heartbeat + deadline enforcement; kills process groups; no run can be ghosted |
| `worker.py` | one run: worktree → select + triage → context → phase1 → verify1 → gate → phase2 → verify2 → publish (deep backlog: context → phase2 → verify2 → publish) |
| `prep.py` | one cheap lane answers both pre-review questions: the effort tier and the discretionary specialists |
| `triage.py` | the tier guide and rules (trivial/low/normal/critical); the tier picks each phase's `--effort`; runs alone when there are no specialists to choose |
| `select.py` | specialist selection: the selector lane (runs alone without triage) and the heuristic fallback |
| `quota.py` | Phase-1 model ladder: remaining Antigravity / Z.AI quota via the proxy's management `api-call`; first rung with quota runs |
| `lane.py` | runs `claude --bare --permission-mode plan` in the worktree, captures JSON + token usage; stoppable mid-run; correction turns in the lane's own session |
| `lanepool.py` | machine-wide lane slots per model family (`flock` files) and one priority line per pool, shared by every worker; the only concurrency limit on reviews |
| `prompts.py` | assembles prompts from the `thepastaclaw/skills` repo templates |
| `contract.py` | reviewer/verifier JSON contracts and their deterministic normalization (echo fields, prior hashes); legacy-compatible `finding_hash` / `dedupe_key` |
| `dedupe.py` | within-batch same-root collapse; cross-round matching against existing inline comments |
| `publish.py` | pure `render(ReviewModel)`; diff position mapping; posting; CodeRabbit reactions |
| `labels.py` | mirrors the bot's standing verdict onto a `pastaclaw:*` PR label (repos that define them) |
| `github.py` | PR metadata, review threads, evidence bundle, gate/status comment |
| `status.py` | `reviewsys status --json` + watchdog predicate |
| `doctor.py` | gh auth, claude launcher, skills, proxy single-`stop_sequences` probe per model, quota per Phase-1 rung |

State: `~/.reviewsys/review.db` (SQLite, WAL). Tables: `prs, heads, runs, steps, lanes, findings, posted_findings, reviews, inbox, events, kv`.

## Model policy and effort tiers

Models, agents and default reasoning levels come from `review_model_policy` in the
skills repo's `config.json`, read at every config load (no redeploy to change them).
If the policy has a `triage` block, a `gpt-6.1-sol --effort low` lane rates each PR
and the tier picks the `--effort` of the reviewer lanes; verifiers keep their fixed
level. The same lane also picks the discretionary specialists (see
[The prep lane](#the-prep-lane-tier-and-specialists-in-one-call)). Blockers always win: a Phase-1 blocker publishes the preliminary review
regardless of tier. `trivial` with no blockers publishes a *final* review from
Phase 1 only and says so in the provenance block. Triage failure falls back to
`fallback_tier` and is recorded as a `triage.degraded` event. `normal` is the
default tier; `critical` requires both a large or intricate diff *and* a change to
a critical surface (consensus, funds, cryptography, key handling, peer-facing
deserialization, migrations), and the triage must name what meets the bar. Size
alone never qualifies, and the tie-break is the lower tier. This replaced the
"large *or* sensitive, when unsure go higher" wording on 2026-09-10 after 87 % of
runs came out critical.

| tier | Phase 1 (ladder rung, capped by the rung's `reasoning`) | Phase 2 (gpt-6.1-sol) |
|---|---|---|
| trivial | high | skipped |
| low | high | medium |
| normal | max | high |
| critical | max | xhigh |

Efforts are validated against `low|medium|high|xhigh|max` at load. Every lane's
effort is stored in `lanes.effort`, the tier in `runs.tier`, and both are printed
in the review's provenance block and the gate comment. Without a `triage` block the
policy behaves as a single `normal` tier at the configured reasoning levels.

### The prep lane: tier and specialists in one call

Before the review starts two questions are asked: which tier the PR is (above) and which
discretionary specialists review it. They used to be two lanes in a row, the selector on
`selector_model` (`gpt-5.6-terra`) and then triage, each with a pool slot and a cold start
of its own, though neither reads the other's answer (since v0.23: selection 43 s on
average, max 126; triage 14 s, max 115). Since 2026-10 one lane (`prep.py`, role `prep`,
artifacts in `run-N/prep/attempt-*`) on the triage model and effort asks both, in two
separated sections of one prompt: triage's tier guide and decision rules and the
selector's specialist list and rule word for word. It replies with one object,
`{"tier", "tier_reasoning", "selected", "selection_reasoning"}`.

- Each half is validated on its own: an unknown tier falls back to `fallback_tier`
  (`triage.degraded`), a missing or non-list `selected` to the selector's trigger
  heuristics (`select.degraded`); unknown specialist ids are dropped as before. One broken
  half never discards the other: the second attempt only fills the half still missing.
  Both attempts run on the triage model; the selector's second attempt used to go to the
  Phase-2 model, which is the same `gpt-6.1-sol` pool since policy v10.
- A quota-shaped lane failure (stderr, never model output) flips the run into degraded mode
  and asks the stand-in once more for the half (or both) that fell back, as the two lanes
  did; the event detail reads `prep lane: …`.
- Outputs are unchanged: `selector.json` / `triage.json`, `runs.tier`, `Triage.method`
  (`llm:<model>` / `fallback`), the provenance line, and both the `select` and `triage`
  step rows. The rows start and end together, so the progress profile (step durations by
  name over the last 40 runs) reads old runs, where they ran one after the other, and new
  ones alike: while both run the estimate counts the longer, and while both are still to
  come it counts both, about 45 s too much at the very start of a run.
- Without triage (a policy with no `triage` block, or a light audit, which reviews at a
  fixed tier) the selector runs alone as before; with no discretionary specialists (the
  selection comes from config) triage runs alone. Conversation runs ask neither.

### Phase-1 roster, points gate, single-stage tiers, repos without Phase 1

Four policy knobs trade Phase-1 latency for value. They were set from the 2026-09-30
per-agent audit: across 14 days, the Phase-1 specialists produced almost nothing no
other lane found, only 10-18% of Phase-2 findings had been raised in Phase 1, and
Phase 1 scored 4-10x lower value per token than Phase 2. All four are off when absent.

- `phase1.specialists` (list of specialist ids): Phase 1 runs `general` plus only these,
  when the selector picked them. Always-run specialists are not exempt. Phase 2 still
  runs every selected specialist.
- `phase1.gate` (`{"block_above": N, "weights": {...}}`): the verified Phase-1 findings
  are scored (default weights `blocking` 3, `suggestion` 1, `nitpick` 0), and Phase 2 is
  deferred only when the total is **above** `block_above`. Scored are the findings new to
  this head plus carried-forward (STILL_VALID) blockers; carried suggestions are not, or an
  earlier round's unaddressed suggestions would hold every later head at Phase 1. Under the
  budget, Phase 2 runs and the final verifier re-adjudicates the Phase-1 findings with
  everything else. A preliminary review held back by suggestions alone is published as
  COMMENT, never APPROVE, and a head whose final review already stands (a same-sha
  re-review) is never held back by suggestions alone. The gate only defers a Phase 2 that
  would run: a tier without Phase 2 (`trivial`) publishes Phase-1-only as before, a verified
  blocker still as a preliminary REQUEST_CHANGES. The `gate` step records `points` and
  `block_above`; the gate comment prints them when the gate deferred. Without a gate, any
  verified blocker defers Phase 2.
- The Phase-1 roster only slims down when a Phase 2 follows; a Phase-1-only review (the
  `trivial` tier, light audits) keeps every selected specialist.
- `triage.tiers.<tier>.single_stage: true`: no gate for that tier. The Phase-1 reviewers
  (on their ladder model) run *beside* the Phase-2 reviewers, and the final verifier
  weighs both sets. The verifier is told the Phase-1 claims are unverified. There is no
  `verify1` or `gate` step. It is meant for `critical` changes, whose authors are expected
  to have reviewed them closely already, so a gate only adds latency. Phase 1 is extra
  coverage here: if it fails on every rung its output is dropped (`phase1.failed_single_stage`,
  disclosed as for a Phase-1 failure), never the review. The failure is recorded the moment it
  happens (the `phase1` step's error and the event), so the status page shows why Phase 1 is
  red while Phase 2 is still going; the output is dropped once Phase 2 is done. If Phase 2 fails, the Phase-1
  lanes are stopped and the run fails as usual. Disclosed as "Single stage: Phase 1 and
  Phase 2 reviewed this head side by side".
- `repos[].phase1: false` in the skills config: reviews of that repo skip Phase 1 and go
  straight to Phase 2 (`phase1` step `skipped`, event `phase1.skipped_repo`), titled
  "Final validation — Phase 2 only (no Phase 1 for this repository)". A `trivial` change
  there reviews in Phase 2 at `low` effort. Audits keep their own flow.

CLIProxyAPI clamps `--effort` to the `thinking.levels` declared per model in its
config; the zai GLM entries must declare `[low, high, max]` or `max` reaches z.ai
as `high` (fixed on the box 2026-09-08).

### Phase-1 model ladder (spend included quota first)

`review_model_policy.phase1.candidates` is an ordered list of Phase-1 reviewer
models; each may name the subscription whose remaining quota gates it:

```json
"candidates": [
  {"model": "gemini-3.8-flash-high", "reasoning": "high", "quota": {"provider": "antigravity", "group": "Gemini Models"}},
  {"model": "glm-5.3-flash", "reasoning": "max", "use_up_to": "high", "quota": {"provider": "zai"}},
  {"model": "muse-spark-1.3-contributor", "agent": "muse-reviewer", "reasoning": "xhigh"}
],
"quota_reserve": 0.15
```

At the start of Phase 1 the worker walks the ladder and runs every Phase-1 lane
on the first rung whose subscription still has at least `quota_reserve` of
*every* window (5-hour and weekly) left. A rung without `quota` is never gated,
so the pay-per-token last rung always resolves. A failed lookup skips the rung
rather than gambling on a lane that may die on 429 an hour in. A rung's
`reasoning` is a *cap*: the lane runs at the lower of the tier's `phase1` effort
and the rung's (Gemini through Antigravity tops out at `high`, the Muse
contributor tier at `xhigh`). If a Phase-1 lane still fails on its rung (rate
limit, dead upstream, malformed output twice) the run drops to the next rung
for that role and the rest (`phase1.model_fallback`) instead of failing; only
the last rung's failure fails the run. A rung may also carry `use_up_to`, an
effort ceiling: when the tier asks for more than that the rung is passed over
without a quota lookup ("not used above high effort; tier asks max"), and the
fallback below a failed rung honours it too. This exists for GLM, which is a
good reviewer up to `high` but at `max` thinks for 50–150 min per lane
(measured 2026-09-09..16 over 118 lanes: median 51 min, p90 136; Gemini at
`high` median 12, Muse at `xhigh` median 5), so `normal` and `critical` tiers
go to Gemini or Muse and only `trivial`/`low` still use GLM. Phase 1 always
runs; the ceiling only changes which rung runs it. Only the last rung may be ungated; the
`phase1.reviewer` node stays the declared default whose `reasoning` seeds the
tier table. The choice is recorded
as `phase1.model_selected`, stored per lane in `lanes.model` / `lanes.effort`,
and disclosed in the review: "Phase 1 model: gemini-3.8-flash-high — antigravity quota: 5h 90% left,
weekly 60% left; passed over …". Without a
`candidates` list the policy is a single ungated rung (`phase1.reviewer`).

Quota is read through CLIProxyAPI's management API (`~/.cli-proxy-api/management-key`),
which has no quota endpoint of its own but relays a request signed with a stored
credential (`POST /v0/management/api-call`, `$TOKEN$` placeholder), so keys never
leave the proxy: Antigravity via `cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary`
(per quota group, `remainingFraction` per bucket), Z.AI via
`api.z.ai/api/monitor/usage/quota/limit` (`CREDIT_LIMIT` percentage used). With
several credentials the best one counts, since the proxy fails over between
them; `account` in a rung's `quota` pins one credential (Antigravity email or
zai provider name). `reviewsys doctor` prints each rung's account and windows and
runs one real launcher lane per rung. Models whose
upstream rejects `stop` (Gemini through Antigravity, Meta) pass the doctor probe
on a stop-free request; Claude Code lanes never send one. New model aliases must
also be allowed by the launcher's `resolve_model_alias` (`gemini-*`, `muse-*`).
Effort clamping per rung, verified in the proxy request logs on 2026-09-10:
Gemini through Antigravity tops out at `thinkingLevel: high` (`max` → `high`);
the Muse contributor tier has no `max`, so `max` → `reasoning_effort: xhigh`;
GLM honours `max`. The Antigravity quota endpoint answers 403 "no valid license"
unless the request carries an Antigravity client `User-Agent`.

### Sharing the box with CI

The Mac Studio also hosts a GitHub Actions runner whose Rust builds drive the
load average past 100. Under that load CLIProxyAPI's management API stalled
past a 10 s timeout and one review fell to the paid Muse rung although Gemini
had quota (dash#7107, 2026-09-10). Mitigations: `claude` lanes spawn at nice 10
(`lane.LANE_NICE`), quota lookups use a 30 s timeout with one retry and then
fall back to the last successful reading if it is under an hour old
(`quota.cached_reader`, kv `quota.last:<provider>[:<group>]`); only with no
usable reading is the rung skipped. On the box itself the proxies should run as
`ProcessType Interactive` / `Nice -10` and the runner as `Background` / `Nice
10`, priorities only so CI keeps full parallelism when the box is otherwise
idle; that needs sudo (`~claw/prioritize-proxies.sh`).

### Reviews are static: lanes never build

Every lane runs under `sandbox-exec` with a profile that allows everything but
executing anything under `[lanes] deny_exec` (default
`config.DEFAULT_LANE_DENY_EXEC`: the rustup toolchains and cargo proxies, make,
cmake, ninja, go, xcodebuild and the Xcode toolchains, clang/cc/ld, swift, java,
npm/yarn/pnpm, sccache). The kernel enforces it for every process the lane
starts, so a project script that runs cargo is stopped too. `git` stays usable:
`/usr/bin/git` goes through `xcrun` into Xcode's `usr/bin`, which is why those
two are not on the list. Set `deny_exec = []` to turn the sandbox off; with no
`sandbox-exec` (Linux) lanes run unsandboxed. Reviewer and verifier prompts say
the review is static and point at the PR's CI instead: the head's check runs
and commit statuses are in the evidence as `ci`, and lanes may run
`gh pr checks` / `gh run view --log-failed` for the current state. Why
(2026-10-01): with parallel lanes and up to 9 runs, lanes built dashd, the
platform workspace and the iOS FFI side by side (each run in its own worktree,
so each a cold build); the box went to load 900 and 60 GB of swap, and a
`low`-tier dashwallet-ios review spent a 3-hour lane timeout polling an FFI
build with `sleep 420`.

### Output contract: normalization, correction turns, repair

A reviewer lane must answer one JSON object (`summary`, `findings`, `out_of_scope_findings`,
`review_phase`, `head_sha`, and on a re-review one `prior_finding_reconciliation` row per
prior finding); a verifier its own schema (`contract.parse_verifier_output`). Measured
2026-09-26..10-01, about a tenth of runs failed, and the output contract was a large part of
it: a broken answer failed the whole phase (its siblings stopped) and usually the run. Three
layers now stand between a broken answer and a failed phase, in this order:

1. **Deterministic normalization** (`contract.normalize_reviewer_output`, no model call).
   `review_phase` and `head_sha` are echo fields: the worker states both in the prompt and
   knows them. A missing or wrong `review_phase` (seen: `final`, `complete`, `planning`,
   `in_progress`, `''`; a Gemini lane on run 2883 returned only `findings` and
   `out_of_scope_findings` after 28 minutes of good work) is set to the expected phase; a
   missing `head_sha`, or a prefix of the assigned head of at least 7 hex characters, becomes
   the full sha. A `head_sha` naming any other commit is still rejected: that lane may have
   reviewed the wrong code. The verifier's `review_phase` is an echo field too and is
   normalized the same way. On a re-review, a reconciliation row's (or a carried finding's)
   `finding_hash` that is a prefix (7+ chars) of exactly one prior hash, or has exactly one
   prior hash as its prefix (`ccf9a4a010e` / `a265b2d54f12f` for 12-char hashes), becomes that
   hash; an ambiguous prefix is never guessed. Statuses only get their canonical spelling
   (`still valid` → `STILL_VALID`); a word from another vocabulary (the verifier's
   `REFUTED`, `INTENTIONAL_EXCLUSION`) is a judgment call left to the correction turn, as are
   duplicate rows, missing hashes and a STILL_VALID row no finding carries. The verifier is
   shown the normalized output. Every fix is noted: event `lane.output_normalized`
   (phase/role, model, what changed) and the lane row's `reason`, never silent.
2. **Correction turns in the lane's own session** (`[lanes] correction_turns`, default 2,
   clamped to 0..5). Reviewer and verifier lanes run with `--session-id <uuid>` instead of
   `--no-session-persistence`. When the answer has no JSON object, or still breaks the
   contract after normalization, the lane's session is resumed (`--resume <uuid>`) with one
   follow-up turn: the exact validation errors (all reconciliation problems at once), the
   schema keys, the expected `review_phase` / `head_sha`, and on a reconciliation problem
   every prior hash with its original title, the allowed statuses and the carry rule; it asks
   for the complete corrected object only. The model that wrote the answer still has
   everything it read, so it fixes what is named instead of a context-free model guessing.
   The turn runs through the same lane machinery (model, effort, launcher, `deny_exec`
   sandbox, `should_stop`, nice) inside the lane's pool slot (`lanepool.gated` holds it across
   the corrections: never released, never lined up again; a comparison lane stays in the
   `compare` pool), with a timeout of at most 30 minutes, artifacts in the lane's
   `correction-N/`, and its tokens added to the lane's row (status `corrected`). Events
   `lane.corrected` and `lane.correction_failed` (phase/role, model, turns, first error, and
   for a failure whether the answer stayed invalid or the session could not be resumed). The
   saved session (`$CLAUDE_CONFIG_DIR` or `~/.claude`, `projects/<worktree slug>/<uuid>.jsonl`,
   the only file a session leaves; megabytes for a long lane) is deleted when the lane's
   turns end; `gc` removes what a dead worker left behind, and the emptied project dirs, for
   finished runs' worktrees only (`runs.worktree`), after a day. Verified with Claude Code
   2.1.286: `--bare --print --output-format json` keeps and resumes sessions this way (from
   any cwd, prompt cache included), and a missing session exits 1 ("No conversation found
   with session ID"). The envelope's `usage` is per call, so a lane row's tokens are the sum
   of its turns; `total_cost_usd` is the session's running total, so a correction turn's
   `lane-meta.json` cost includes the turns before it, and `--max-budget-usd` (passed again)
   bounds the lane and its corrections together. `reviewsys doctor` runs one real
   keep-resume-delete round trip through the launcher ("lane correction turns") whenever the
   knob is on, and fails it when the session lands where the cleanup cannot see it (a
   launcher with its own config dir) or cannot be deleted.
3. **The repair lane**, the last resort for an answer that still has no readable JSON object
   (the session could not be resumed, the corrections ran out, or the knob is 0): the
   context-free `repair_model` (low effort) reformats it, now told the expected
   `review_phase`, full `head_sha` and prior hashes, and its output goes through the same
   normalization and validation (lane status `repaired`). Before, 44% of repaired outputs
   failed the contract on exactly those fields.

An answer still invalid after all of that fails the attempt; the lane's second attempt
starts fresh (with its own corrections), and only then does a Phase-1 lane fall down the
ladder. (Before, an answer whose JSON parsed but broke the contract failed the phase
immediately, with no retry and no ladder.) A lane told to stop between its turns (a sibling
failed, the run was cancelled) is recorded as stopped: no correction, retry or ladder fall.

### Review body layout

The body leads with what the developer needs: title, verifier summary, severity
counts, then every finding that could not be posted inline (line outside the
PR's diff, or GitHub refused the diff) rendered in full with its location,
body and suggestion. Provenance (the Source line, triage, per-lane models and
efforts, the Phase-1 ladder choice) follows inside a collapsed `<details>`
block, as do the agent prompt and out-of-scope notes. `parse_diff` strips the
tab git appends after paths containing spaces; without it every finding in
such a file was reported as "not in diff" (dashwallet-ios#1118, 2026-09-10).

### Finding attribution

The verifier labels each finding's `source` with the legacy template slots
(`codex` = Phase-1 lanes, `claude` = Phase-2 lanes, `coderabbit`). Before
publishing, `publish.attribute_sources` rewrites that into the lanes that
actually raised it: within each named phase, the lanes whose own output carries
the same `finding_hash`, or every lane of the phase when the verifier retitled
or merged. The inline footer then reads
"source: gpt-6.1-sol (phase2-reviewer: general); coderabbit". On a
[comparison run](#model-comparison-a-second-model-on-a-sample-of-runs) each lane is
matched by its own output key (`general#2` for the comparison twin), so the footer
names the model that actually raised the finding.

## Backlog mode: Phase 2 only

`backlog_skip_phase1_above` in `config.toml` (default 10, `0` disables) trades depth
for throughput. When a run starts and more heads than that are queued, the worker
records the `phase1` step as `skipped`, emits `phase1.skipped_backlog`, and goes
straight to the Phase-2 reviewers and final verifier. The verifier is
told the Phase-1 block is intentionally empty. The review is titled "Final validation
— Phase 2 only (queue backlog)", the provenance says
"Phase 1 reviewers: **not run (skipped for throughput: N PRs queued, above the L
limit)**", and the gate comment carries "Phase 2 only (queue backlog)". Blockers found
by the final verifier still publish REQUEST_CHANGES. The rule never applies to a
`trivial` tier (which has no Phase 2) or when Phase 2 is disabled. Measured
2026-09-08: GLM Phase-1 lanes took 65–140 min each, sequentially; astra Phase-2 lanes
3–11 min.

The rule also applies in [degraded mode](#degraded-mode-stand-in-models-when-the-openai-pool-is-dry),
where Phase 2 runs on the stand-in: a degraded backlog is the case that can least afford
the slow Phase-1 rungs. Set `backlog_skip_phase1: false` in the degraded policy to keep
both phases (and the cross-model check) at that cost.

The rule counts queued live heads. Since reviews are bounded by model slots rather than
a run count (see [Parallel reviewer lanes](#parallel-reviewer-lanes-and-model-slots)), an
eligible head starts on the next tick, so the queue only gets deep (and the rule only
triggers) while `max_runs` holds heads back, or with heads still in debounce/backoff.

## Degraded mode: stand-in models when the OpenAI pool is dry

The primary OpenAI model (`gpt-6.1-sol` since policy v10, `gpt-6-astra` before) sits on the critical path of every run (triage, Phase-1 gate verifier,
Phase-2 reviewers, final verifier, conversation lane) and the selector/repair models
share the same Codex pool, so when that pool is out of quota every run dies on a 429
and PRs silently get no review (2026-09-17: 60 failed runs in 8 h). With a
`review_model_policy.degraded` block in the skills config the pipeline keeps going:

```json
"degraded": {
  "sentinel": "gpt-6-astra",
  "phase1_effort_cap": "high",
  "substitutes": {
    "gpt-6-astra":  {"model": "muse-spark-1.3-contributor", "effort_cap": "xhigh"},
    "gpt-5.6-terra": "muse-spark-1.3-contributor",
    "gpt-5.6-luna":  "muse-spark-1.3-contributor"
  }
}
```

- **Detection.** Before every run the worker sends one 1-token request for the
  `sentinel` through the proxy (cached 2 min in `kv degraded.probe`). Only a
  quota-shaped answer (HTTP 429, "cooling down", "usage limit") counts; a dead proxy
  or a 5xx does not, because swapping models would not help. A lane that dies on a
  quota error mid-run (reviewer, verifier, prep, triage or selector) flips the run over on
  the spot, gets its attempts again on the stand-in, and records a 20-minute hold so
  the following runs start degraded even if a tiny probe happens to get through.
  The daemon re-probes every 2 minutes so the mode clears by itself once quota is
  back (a lane-observed hold keeps it on for its 20 minutes first), and sends one
  Slack alert per transition (`degraded.transition` event). Only a lane's *stderr*
  is classified, never model output: a reviewer discussing rate limits in the PR
  under review must not flip the system.
- **What changes.** Every lane whose model has a substitute runs on it, with the
  lane's effort clamped to the substitute's `effort_cap`. Phase 1 stays on its own
  ladder but the tier effort is capped at `phase1_effort_cap` (`high`), so the GLM
  rung (passed over above `high`) and Gemini stay eligible instead of every Phase 1
  landing on the paid Muse rung too. Both phases run, except under a deep queue: the
  backlog shortcut still applies (`backlog_skip_phase1`, default true), so a degraded
  backlog run goes straight to Phase 2 on the stand-in instead of waiting on the slow
  Phase-1 rungs — a review then has no cross-model check, which is the trade the queue
  buys. Set it false to keep both phases regardless of the queue.
  A degraded review never APPROVEs (it stops at COMMENT);
  blockers still REQUEST_CHANGES; and a same-sha re-review in degraded mode never
  retracts a standing full-strength approval (`review.verdict_kept`).
- **Disclosure.** Review title `⚠️ DEGRADED — …`, a warning block under the title
  naming the reason and the stand-ins, `- **Degraded mode**: …` and
  "`muse…` (standing in for `gpt-6-astra`)" in the provenance, the gate comment and
  queue comment carry `⚠️ DEGRADED`, `runs.degraded=1`, events `degraded.run` /
  `degraded.entered_midrun`, and `reviewsys status` prints a `mode:` line.
- **Operator override.** `reviewsys degraded on|off|auto` (`--probe` re-probes now);
  `doctor` prints the current state and probes every stand-in model.
- Without a `degraded` block nothing changes: a primary-model outage fails runs as
  before.

## Model comparison: a second model on a sample of runs

A `review_model_policy.comparison` block runs a second model beside the primary one
on a sample of runs, so a new model can be judged against the one it replaced on
real PRs (policy v10: `gpt-6.1-sol` primary, `gpt-6-astra` on 25 % of `critical`
runs):

```json
"comparison": {"model": "gpt-6-astra", "tiers": ["critical"], "fraction": 0.25}
```

- **Selection.** After triage, a run whose tier is listed is picked when a stable
  hash of `repo#number@sha` falls under `fraction`, so a retried run makes the same
  choice. Never on an audit or in degraded mode (both sets would run on the same
  stand-in). The comparison model must differ from the Phase-2 model (checked at
  load). Event `compare.selected` records the pair.
- **What runs.** Every Phase-2 reviewer lane (general and each specialist) also runs
  on the comparison model at the same effort, keyed `<role>#2`, on a thread pool of
  its own so the primary lanes keep their `phase_parallelism`. Not on the fresh final
  pass: it would double an already large verifier prompt. The verifier gets both sets
  in the one Phase-2 block under neutral labels (`general/a`, `general/b`; which one
  is the comparison model is drawn per run), and the primary verifier alone decides
  the verdict. Comparison output never answers or resolves a finding thread.
- **It never holds the review up.** Comparison lanes take their slots in a
  machine-wide `compare` lane pool (`[scheduling] lane_pools.compare`, default 2),
  never the production `gpt` one: a slot is held for the whole lane, so a comparison
  lane in the production pool could make a primary lane (even its own run's retry)
  wait. They still use the same OpenAI accounts, whose stream budget the `gpt`
  pool is, so one only starts while no lane is in the `gpt` line and that pool has an
  idle reviewer slot at that moment; the `compare` pool bounds how far they can overshoot when production
  picks up after they started. Each gets one attempt. A failed one is dropped (`compare.lane_dropped`), and
  one still running (or still waiting for a slot) 20 minutes (`COMPARE_GRACE_MINUTES`)
  after the last primary lane of its phase finished is stopped and dropped the same
  way, with a reason saying which. A quota error on one
  never flips the run into degraded mode or onto a stand-in. A phase that fails
  stops its comparison lanes too. Per phase, `compare.lanes` records `kept=N
  dropped=M`.
- **Disclosure.** Only when a comparison lane finished: the provenance lists it with
  the other Phase-2 reviewers and adds "Model comparison: every Phase-2 reviewer also
  ran on `…`". Inline footers name the model whose lane actually raised the finding.
- **Reading it.** Reviewer-lane findings are stored with the role in the stage:
  `lane:<role>` (primary), `compare:<role>` (comparison), `fresh:<role>` (fresh final
  pass); the fresh final verifier writes `verified-fresh`, the first one `verified`.
  `reviewsys compare [--since ISO] [--json]` compares per role and first round only: a
  role counts when both its lanes finished and the comparison lane was not dropped.
  Every finding the first final verifier kept is credited to the model(s) whose paired
  lanes raised it (same `finding_hash`, or same file and category with overlapping
  lines when the verifier retitled); one only an unpaired role raised is left out. Per
  model: raised, kept, kept blockers, kept findings *only* that model raised, output
  tokens. Runs count when finished, never degraded, only the two models in the first
  Phase 2, and at least one paired role (a run the Phase-1 gate stopped has no Phase
  2). Caveats: the verifier is the primary model itself, so a preference for its own
  model's wording cannot be ruled out; comparison lanes dropped for time or slots skew
  towards large PRs; and with 2 compare slots admitted only on idle production
  capacity, runs with many specialists mostly pair `general` and one or two of them,
  so the numbers lean towards the general reviewer. The report says how many roles
  were left out.

## Final approval after an iterative review

An incremental review is a reconciliation pass, not by itself a new approval gate. It
first checks the prior findings, discussion, and replies and stops if any blocker remains.
When that pass is clear, the worker starts a fresh Phase-2 review of the complete current
merge-base range. Those reviewer lanes receive the full diff and evidence without the old
finding checklist, which reduces anchoring on the previous conclusion. The fresh verifier
receives all PR context—including prior findings, discussion, CodeRabbit evidence, and both
sets of reviewer outputs—as historical evidence to reconcile before deciding the verdict.

Approval is tied to the exact reviewed head. The worker checks the live PR head immediately
before publishing, so a push or base update during review invalidates the run and queues the
new head instead of approving an obsolete commit. The published provenance identifies when
the fresh final gate ran.

A head that is obsolete when the worker checks it (at checkout, before publishing, before a
conversation posts) is not a failed review: a push moved the PR's head past it, or the PR
closed or merged. The run ends `cancelled` (reason `head superseded: live head … != assigned
…` or `head closed: PR is closed (merged)`, no `fail_kind`), the head `superseded` / `closed`
with a `head.superseded` / `head.closed` event, exactly as when ingest notices first. No
retry, no `head.failed` alert, no "could not complete" gate comment: the first "in progress"
status is only posted after the checkout check, and for a moved head the newer head's queue
comment takes the PR's comment over. For a closed PR nothing would, so an existing gate
comment is set to "⏹️ Not reviewed — PR is closed (merged)" (none is created). Until 2026-10 these ended as "failed · fatal" on the dashboard (7 runs in
5 days for a moved head, 4 for a merged PR, which also paged as `head.failed`). A moved
*base* still fails `fatal` and re-queues the same head.

## Replies to findings

A human reply under one of the bot's inline finding comments is the highest-value
signal the system gets, and notifications do not carry a comment URL for it, so the
`notify` task also lists each posted-on repo's inline comments updated since a per-repo
cursor (`replies.cursor:<repo>`, trailing the poll start by 2 min so late-listed comments
are still seen; dedupe is by comment id) and puts replies from non-bot users into
`inbox` as `review_reply`. The router fetches the thread root; when it is a bot finding
and the replier is a member/collaborator/contributor or in `trusted_reviewers`, the PR's
live head is queued as priority with trigger `review_reply` (no debounce). An
already-reviewed commit is re-opened for the same reason (`head.requeued`). A reply
while that exact head is *running* is deferred until the run ends; a transient GitHub
failure while looking up the thread root leaves the row for the next tick. Both waits
are bounded by `DEFER_MAX` (6 h), after which the row is marked
`review_reply_expired` / `ignored_unfetchable`. Replies to other people's threads, on
draft PRs, or on the bot's own PRs are not review triggers (the last go to the own-PR
comment batch).

The re-review carries the thread: prior findings that were replied to go into the
prompt with their original `body` and `thread_replies`, the evidence bundle keeps the
bot's own root comment in those threads (it is stripped everywhere else), and findings
the legacy pipeline posted are reconstructed from the comment when this database has no
row for them. Reviewers reconcile each with `STILL_VALID | FIXED | OUTDATED |
INTENTIONALLY_DEFERRED | WITHDRAWN` plus a `reason` addressed to the author. At publish
the bot replies on each thread whose newest human reply is newer than the bot's last
answer there ("**Withdrawn** (re-reviewed at `sha`): …", "**Still applies** …",
"**Resolved** …"), marked `<!-- thepastaclaw-thread-answer v1 sha=… reply=… -->` so
each human reply is answered exactly once per head, and resolves withdrawn/fixed/outdated
threads only when a reviewer stated that status. The verifier's kept set overrides the
reviewer (a kept finding is "still applies" even if a lane said otherwise; a dropped one
is "withdrawn"); a defaulted withdrawal is answered but never resolves the thread.
Threads a maintainer has already resolved are left alone. Reasons are capped at 1200
chars, `@`-mentions are defused, and any CodeRabbit retrigger text is discarded. Open bot
threads nobody replied to get the same note and resolution when a reviewer explicitly
withdrew that finding, so a dropped finding never lingers. If the commit already has a
review (same sha), the standing review is not re-posted; the thread answers are the
output, plus, only when the verdict moved, one short follow-up review ("Re-review after
discussion", marker `thepastaclaw-review-update v1`, no inline comments) that moves the
standing verdict (`review.verdict_updated`). The comparison is against the bot's latest
review state for that commit, follow-ups included, so a correction is posted once; a
review a maintainer dismissed is never re-asserted; a re-review that finds blockers after
a clean final review corrects that final verdict instead of stacking a preliminary
review. `reviewsys enqueue` on an already-reviewed head re-opens it the same way. Every
answer is an event `thread.answered`.

### Replies on an already-reviewed commit: the conversation lane

The paragraph above describes what happens when the reply arrives on a commit that has
*not* been reviewed yet (a push landed since): a normal review with the thread in context.
When the commit **already has a standing review** (the common case: someone answers a
finding on the reviewed head), re-running the pipeline is the wrong tool. It re-derives
the finding from scratch, never sees what it already said, and answers every reply with
the same "Still applies" paragraph; on dashpay/dash#7675 that produced five near-identical
restatements, no engagement with the maintainers' arguments, and no answer to a proposed
patch. So a `review_reply` head whose sha has a standing review runs **conversation mode**
instead (`reviewsys/converse.py`, step `converse`; no selector, triage, reviewer or
verifier lanes):

- One lane (`review_model_policy.conversation`, default: the Phase-2 model at high
  effort, agent `conversation`) gets every thread awaiting an answer with the **whole
  exchange in order, our own earlier answers included**, the finding body, the PR
  discussion, and any commits humans linked in the thread (`github.com/.../commit/<sha>`
  or `/pull/<n>/commits/<sha>`, up to 5, forks of this repository only; fetched into the
  mirror so `git show <sha>` works in the worktree; a commit that cannot be fetched is
  disclosed to the model with the reason and never fails the run).
- It answers per thread with `STILL_VALID | FIX_PENDING | WITHDRAWN |
  INTENTIONALLY_DEFERRED | NO_REPLY` and free prose. The rules it is given: verify claims
  against the code, never repeat a point already made, evaluate a proposed change on its
  merits and say whether it resolves the concern, concede when wrong, stay silent
  (`NO_REPLY`) when a reply is addressed to someone else or there is nothing new to add.
  Severity cannot change in a reply. There is no `FIXED`: the checkout is the reviewed
  commit, so a change that would resolve the finding (linked, sketched, or "changed it"
  but not pushed) is `FIX_PENDING`, which lifts nothing; the push is reviewed in full.
  When the author both argues the current code is correct and offers a change, the lane
  rules on the argument first (`WITHDRAWN` if they are right). A model that still answers
  `FIXED` gets `FIX_PENDING` (dashpay/dash#7778 lifted a blocker on an unpushed change).
- Posting: the prose is the comment (marker `thepastaclaw-thread-answer v1`, once per
  human reply and head, `@` defused, CodeRabbit retriggers discarded), with a quiet
  trailing note for FIX_PENDING / WITHDRAWN / INTENTIONALLY_DEFERRED instead of a bold
  verdict lead; WITHDRAWN threads are resolved (where the bot may). `NO_REPLY` posts
  nothing, is recorded as `thread.answered` with `action: no_reply`, and the decision is
  remembered per (finding, reply) in `kv` (`converse.silent:<repo>#<n>:<hash>`), so a later
  conversation on the same PR never posts a late second opinion to that message; the
  thread is re-considered only when someone speaks on it again; markers for PRs with no
  head queued inside the artifact retention window are pruned by `gc`. Replies from other
  bots (`*[bot]` logins, CodeRabbit) are part of the transcript but never count as a human
  waiting for an answer, and the router does not queue a `review_reply` head for them at
  all, so no bot-to-bot loop can start.
- Lifted findings on threads that stay open (the bot has no write access, e.g.
  dashpay/dash, so `resolveReviewThread` is refused; deferrals are never resolved) would
  still read as an open 🔴. The bot edits its own root comment instead: a quoted status
  line (``> ✅ **Withdrawn** at `<sha>` ``, `⏭️ **Deferred**`, …) under the finding marker,
  tagged `<!-- thepastaclaw-thread-status v1 -->`, replaced on change and removed when a
  later answer says the finding applies again, or a later review keeps it
  (`thread.answered` → `marked`). The full re-review path does the same for threads it
  would resolve. Dedupe strips the line before matching.
- Verdict: the verdict on a commit is its full review minus what the discussion withdrew
  or deferred. A conversation never adds blockers. The standing set is the final verifier
  set (`verify2` rows, the fresh audit's when it ran) of the run that last published
  findings for the sha, so findings dedupe carried onto other (even resolved) threads
  count; plus unresolved bot threads this database never posted (legacy). Threads from
  earlier heads that run re-adjudicated do not count. Lifted = `conceded` rows after that
  run, matched by hash or by file and title. Once no blocker stands, the follow-up
  **APPROVEs** (from `CHANGES_REQUESTED` or `COMMENTED`) when nothing above a nitpick is
  left, something of the verifier set was lifted, and that run is one the pipeline itself
  would approve on: Phase 2 `ok`, not degraded, and on a later review round (an earlier
  review of the PR exists) the fresh Phase-2 audit `ok`. A COMMENT never clears a
  REQUEST_CHANGES on GitHub, so anything less leaves a ready PR showing a blocking review
  (dashpay/dash#7778). If only the fresh audit is missing (it is skipped while blockers
  stand), the follow-up is COMMENT and the commit is re-queued once
  (`review.fresh_audit_requested`, kv `rereview.after_run:<run>` read by
  `scheduler.finish_run`; guard `converse.fresh_audit:<repo>#<n>:<sha>`), so the audit
  runs and approves through the normal same-sha path. Otherwise a standing
  `CHANGES_REQUESTED` moves to COMMENT and the body says why it is not an approval. The
  live head and the bot's reviews are re-read right before posting, so a push or a
  dismissal during the lane never gets an approval. The follow-up is the same short
  "Re-review after discussion" review, with provenance stating that no code was
  re-reviewed. A concession is persisted as a `conceded`-stage `findings` row for the
  sha, so findings conceded in earlier conversations stay lifted. A thread a maintainer
  resolved by hand, without the bot conceding it, still counts. In this lane a deferral lifts a blocker only when a
  maintainer (OWNER, MEMBER or COLLABORATOR, the PR author included) is among the replies
  since the bot's last answer, and its thread stays open as the record; otherwise the
  reply is posted with a note that the blocker stands. (The full re-review path does not
  apply this check yet: a reviewer's deferral drops the finding from the kept set.) A later
  full review of the same sha voids earlier concessions (it re-adjudicated them).
  Known gap: if the worker dies between posting a concession
  and recording it, that blocker keeps counting until the next push is reviewed.
- A conversation posts no "Re-review" summary review and never runs the fresh final
  gate: nothing about the code was re-reviewed, so there is nothing to summarise. The
  live head is re-checked before anything is posted, exactly as before publishing.
- Only a **final** standing review qualifies. A reply on a commit whose standing review
  is *preliminary* (blockers found, Phase 2 deferred) runs the full pipeline, because
  talking a blocker down there must still admit Phase 2 and produce a final review.
- Artifacts: `conversation.json` in the run dir (outcomes and private reasoning);
  invalid lane output is retried once with the rejection reason in the prompt, then
  fails the run (`contract`).

A reply on a commit with **no** standing review still runs the full review with the
thread in context, and `reviewsys enqueue` / a manual trigger on a reviewed commit still
forces a full re-review (with the reconciliation-based thread answers above), so the old
path remains available for "look at this again from scratch".

## Ad hoc reviews (repos without a skill)

`@thepastaclaw review` on a PR in a repo that has no entry in the skills `config.json`
queues a review when the requester is trusted (same rule as replies). A repo that is
listed but `enabled: false` stays off no matter who asks. The worker runs with a
generic project note instead of `project.md`/`review.md` (plus
`skills/_default/review-core.md` from the skills repo if that file exists), offers every
discretionary specialist to the selector (always-run ones are repo-specific and stay
off), and discloses it: the provenance starts with "Ad hoc review: this repository has
no PastaClaw review skill …" and the gate comment carries "ad hoc (no repo skill)".
Open-PR polling still covers only enabled repos; ad hoc is mention-driven by design.

## Verdict labels (filtering by the bot's approval)

GitHub counts a review toward the merge decision only when the reviewer has write
access, so on repos where the bot has triage its APPROVE / REQUEST_CHANGES is visible
in the timeline but invisible to `review:approved` and the merge box, and there is no
`approved-by:<user>` search qualifier. The one per-reviewer filter triage can drive is
a label. A repo that creates `pastaclaw:approved`, `pastaclaw:changes-requested` and
`pastaclaw:commented` gets exactly one of them mirroring the bot's latest verdict on
the live head, and none while no verdict stands.

The label is reconciled from database state, not written by the worker: every minute
the `labels` daemon task takes the PRs touched since its last pass (a `reviews` row
written, a head created or re-queued, the PR closed) and makes GitHub match
`labels.wanted()`, which is the latest `reviews.event` for the PR's newest head. That
single rule covers a review on a new commit, a same-sha follow-up that moved the
verdict, a push seen by ingest or by the router (label cleared before the re-review
starts), a run that published after being superseded (its sha is no longer the newest
head, so the label stays cleared), a run that died between posting and recording, and a
dismissed review (recorded as `DISMISSED`, no label). The canonical event is used, so a
bot-authored PR gets `approved` even though the review itself was submitted as COMMENT.

Filter with `label:"pastaclaw:approved"` (PR list, saved searches, project boards).
Repos opt in by creating all three labels; the daemon checks that before it ever writes
to a repo (on write repos GitHub would otherwise create a missing label on add), and a
repo without them is recorded once as `label.sync_failed` and skipped until restart. Successful changes
are `label.synced` events; a canonical verdict recorded without a GitHub review (own PR)
is `review.verdict_recorded`. The first pass after deploy backfills every open PR once;
a transient GitHub failure holds the cursor and backs off five minutes.

Two things the label does not see: a human dismissing the bot's review on GitHub (the
label keeps the verdict the bot last recorded), and a hand-edited `pastaclaw:*` label on
a PR with no further activity (reconciliation only visits PRs that changed).

## Queue comment and priority requests

As soon as a PR head is queued the daemon posts (and keeps updated, every
`queue_comment_interval_seconds`) one gate comment with the PR's place in line, an
estimated start time (any debounce/backoff still to elapse; with a `max_runs` cap, also a
slot-pipeline model over the median of recent run durations) and an estimated review
duration. The comment carries a checkbox, **Request priority review**; ticking it
promotes the head to the priority lane (front of the queue, its lanes ahead of live
lanes in every model pool, no debounce),
records `head.priority_requested` with the editor's login from the comment's edit
history, and re-renders the comment without the box. GitHub only renders the box as
clickable to users who can edit the comment, i.e. repository collaborators. The
same comment is reused by the worker for "in progress" / "done" / "failed" states,
so a PR never has more than one bot status comment. Writes are capped at 25 per
pass to respect GitHub's content-creation limits; the rest catch up next pass.

A draft PR, and a new head still waiting out the push debounce (not priority, not a retry),
get the "Review not started yet" body instead, with two boxes: **Request normal review**
(queue it now) and **Request priority review**. Each PR gets exactly one of the two bodies
per pass, and a body equal to what GitHub has is never rewritten. A retry waiting out its
backoff keeps the queue body, whose ETA counts the wait. (Until 2026-10 both renderers ran
for a debounced head in the same pass, so its comment flipped between the two every ~5 min.)

### Live progress in the gate comment

While a run is in progress the worker's heartbeat keeps the comment showing a progress bar
(estimated share done, time left, time running), the steps as chips (`✅ triage → ⏳ **Phase
1** (2/3 lanes) → ▫️ verify 1 → …`) and a link to the run on the dashboard.

- The first "in progress" status is posted once `step_worktree` has confirmed the head is
  still the PR's live head, with the checkout as the running step. Setup steps (checkout,
  lane selection, context) show only while they run; the selection and triage of the prep
  lane show as the one triage chip.
- Before triage has set the tier, the estimate measures the run against recent reviews of
  every tier (`progress.ALL_TIERS`, path-aware like the tier profile) and the footer says
  "Estimated from recent reviews"; from the tier on, "… of this tier"; a conversation is
  measured against conversations. A run queued by a reply gets no review estimate before
  triage on the dashboard, which cannot tell yet whether it is a conversation (the worker
  can). Without any history the comment shows the steps and no bar.
- Edits: the heartbeat re-renders the body when the run's steps (or a phase's lane count)
  changed, at most every 30 s (`GATE_PROGRESS_MIN_GAP_SECONDS`), and otherwise every 10 min
  (`GATE_PROGRESS_EVERY_SECONDS`), and compares it with the one last written. Nothing but
  the update time moved: no edit at all. The status line, a chip, the estimate's basis or
  the "taking longer than usual" mark changed: an edit. Only the bar and time left moved:
  an edit only at the 10-min interval. Status changes the worker posts itself (the tier
  after triage, Phase 1 dropped) go out at once, unless they render the same as what is
  shown. Before 2026-10 every step change re-posted the body, and
  since the early steps were hidden, the comment showed a bare "Review in progress" line
  re-posted two or three times in the first minutes of each run (dashpay/platform#5237).

## Status dashboard: priority vs normal timing

The public status page (`site/`, fed by `reviewsys export-status`, see
`deploy/publish-observability.sh`) compares the two paths a live review takes, over the
runs that started in the last 7 days (`history.path_timing` in `status.json`, each figure a
median, a p90 and its sample count n):

- **Path** (`runs.path`, the day table's `priority` / `post-merge audit` badges and its
  path filter) is fixed when the scheduler starts the run: a live head with
  `heads.priority` set (a mention, a review request, a ticked priority box, a reply under a
  finding, `enqueue`) is priority, every other live head is normal, an audit head is audit.
  It is the split the pool lines rank lanes by. It is recorded on the run because the head
  changes afterwards: a reply re-queues a finished head as priority, and an audit takes over
  a merged head. Runs from before the column were filled from their head, except where the
  head was queued again after they started (unknown, left out). Post-merge audits are left
  out (they only take capacity live review leaves idle, nobody waits on them), and so are
  conversations (replies on a reviewed commit: seconds long, not reviews).
- **Wait to start**: the day table's "Waited" figure, from the head's `queued_at` (or the
  end of the head's previous run, for a retry) to the run's start. The normal path's
  debounce is part of it. Every run that started counts, failed and cancelled ones too:
  they waited just as long before anyone knew how they would end.
- **Review time**: start to finish of completed runs only (a failure ends early).
- **Slot wait in run**: of those completed runs, the time the run only waited for model
  slots (at least one lane in a pool's line, none running), read from the stall credit
  added to `runs.deadline_at` (see below) less the run timeout configured now (so a change
  to `run_timeout_minutes` skews it until the window has passed). It exists since v0.23.0;
  earlier runs waited for their slot before they started, inside their wait to start, and
  are not counted here.

The "longer than usual" mark on an active run (a running step past 1.5× its typical
duration, `progress.estimate`) compares the run with recent reviews of its own tier *and*
path; a path with fewer than five recent reviews of the tier falls back to the tier's
reviews of both paths, and an audit is measured against both. The PR's gate comment uses the
same estimate ([live progress](#live-progress-in-the-gate-comment)). The day view filters
its runs by path.

A failed step of an active run shows its error under the run (collapsed or not) and as the
red pill's tooltip, e.g. a single-stage Phase 1 that failed while Phase 2 is still going.
The page is public and the error text can quote model output about a private PR, upstream
replies, hosts or keys, so the export never carries it: `exporter.public_error` rebuilds a
line from known words only (the lane named at the start of the message, a fixed failure
phrase, attempts and correction turns), e.g. "phase1/general: answer had no JSON object (2
attempts, 2 correction turns)"; anything unrecognised is just "failed". The full text stays
in the step and the events. A lane in a correction turn is listed live as
"general (correction 1)".

## Liveness model

The daemon spawns `reviewsys worker --run-id N` in its own session. The worker
heartbeats every 15 s and checks for cancellation; the reaper fails any run
whose heartbeat is older than 5 min, whose deadline passed, or whose process is
gone, and requeues the head with backoff (`infra` up to 3 attempts, `contract`
twice, `fatal` never; a stale head is no failure at all, see
[above](#final-approval-after-an-iterative-review)). Runs in flight are recounted from the DB every tick. The run
deadline (`run_timeout_minutes`, 360) does not count time a run only waits for model
slots: while at least one of its lanes waits for a slot and none runs, the worker pushes
`runs.deadline_at` out by that stretch (in one-minute slices while it lasts, so a long
wait is never reaped mid-stall; at most two run timeouts in all, so a line that never
moves still ends in a timeout). Lane timeouts start when the lane process starts.

Reviews are not limited by a run count. The owner's rule (2026-10): "instead of limits
for PRs, only limit for models; if we have open model slots, do it. That way we get Muse
out of the way ASAP; and prioritize verify over finder jobs." So every eligible head
starts on the next tick (one run per PR; priority heads first), and each of its lanes
waits for a slot in its model's pool ([below](#parallel-reviewer-lanes-and-model-slots)).
A run whose Phase 1 is on Muse or GLM gets on with it while gpt is busy, instead of
waiting for another run to end. No role moves to another model because its pool is
busy. `[scheduling] max_runs` (default 30, 0 = none) is only a runaway guard on runs in
flight, live and audit together. Audit heads start (up to the audit concurrency) only
when no live head is left waiting.

The `gpt` pool scales with the OpenAI (Codex) accounts that can take work, because the
primary models are limited per ChatGPT account (~3 concurrent streams, weekly
quota), not globally. Each usable account adds `max_concurrent + priority_overflow`
streams (2 + 1; the names are historical, from when they sized the run slots), for at
most `account_scale_max` accounts (3, so 9). Every 2 min the daemon reads the proxy's `auth-files`
(its record of each credential's cooldowns and `X-Codex-*-Used-Percent` quota
headers; nothing is sent to OpenAI). An account is usable when it is enabled, not
parked, and not cooling down for itself or for the sentinel model: it is spent to
the end. The exception is `account_reserves` in the box's `config.toml`, keyed by account
email (the list lives only on the box, since this repository is public): at
that floor reviewsys *disables the account in the proxy*, so no client eats into
the reserve, and re-enables it once the window that hit the floor has reset. It
alerts on both, and it never re-enables an account an operator disabled. With no
reading, or one older than 20 min,
the `gpt` pool falls back to one unit (3). `reviewsys status` prints the effective
`runs: max N (0 = unlimited); lane pools: gpt N (reason), ...` line.

### Parallel reviewer lanes and model slots

Within a phase the general reviewer and the selected specialists run side by side
(`phase_parallelism` per run: default 0 = every lane of the phase at once; 1 restores
the one-after-another flow). They
never read each other's output, so only the verifier after them waits for all of them;
the phases themselves run in order, except on a `single_stage` tier. Measured 2026-09-23..27 (193 two-phase
reviews, 2–6 reviewers per phase), this cuts a full review from ~65 to a projected
~38 min; a cap of 3 already gets ~95% of that.

A run no longer holds one model session at a time, so every lane (reviewers, verifiers,
triage, selector, repair, conversation) first takes a slot in its model family's
machine-wide pool (`lanepool.py`): an exclusive `flock` on one of N files under
`work/lane-slots`, held while the lane runs and dropped by the kernel if the worker dies.
These pools are the only concurrency limit on reviews. `gpt` lanes get the per-account
stream budget (above); the other families take `lane_pools` (`muse = 8, glm = 6,
gemini = 6` by default, from the measured headroom: Muse 100 RPM per team at ~3.7
req/min per lane, peak 11 in flight with no 429s). A lane waiting for a slot simply
starts later.

Every lane waits in its pool's line: a ticket file per waiter under
`lane-slots/<pool>.line`, named `rank-run_id-time-pid-thread`, so the line is ordered by
rank, then the older run, then arrival. Ranks: 1 priority-run side lane, 2 live-run side
lane, 3 priority-run reviewer, 4 live-run reviewer, 5 audit side lane, 6 audit reviewer;
a side lane is anything that is not a parallel reviewer (verifiers, triage, selector,
repair, conversation), short and on some run's critical path ("prioritize verify over
finder jobs"). The head of the line may take any slot its kind allows; the top slot of
a pool with more than one is kept for side lanes, so reviewers never take it. A side lane
with only reviewers ahead of it may take the top slot (and only that one) without being
the head, so a verifier never waits behind a reviewer queue. A ticket left by a dead
worker is pruned by the next waiter. So at the static 2 + 1 budget a Phase-2 run gets two
gpt reviewers at a time; each usable OpenAI account adds three more slots. (The line
replaced the reviewer-only `<pool>.wait` line in a new directory, so workers still on the
old code during a deploy never read the new ticket format; both take the same slot files.)

One lane failing for good (twice, and on every Phase-1 rung) fails the phase; its
siblings are stopped (`lanes.status = cancelled`) instead of spending quota on an
abandoned phase, their `reason` naming the lane that failed and how ("stopped:
phase1/rust-quality output failed the contract (...)"; "stopped: phase2 failed (...)" for the
Phase-1 lanes of a single-stage run). A failed Phase 1 no longer fails the run: whether a reviewer lane died on
every rung, the gate verifier died twice, or either returned output that breaks the
contract, Phase 1 is dropped and the run goes on with Phase 2 alone, exactly like the
backlog rule (the step that failed, `phase1` or `verify1`, is marked `failed` with
`fell_through`; event `phase1.failed_fallthrough`; review titled "Final validation —
Phase 2 only (Phase 1 failed)"). It still fails the run for a `trivial` tier (no Phase 2),
for an audit (it needs the complete finding set and is retried instead), and for a FATAL
error such as a broken prompt template. A failed Phase 2 fails the run as before. A Phase-1 rung that dies moves the whole run down
the ladder once: lanes still on the dead rung follow when they fail, lanes not yet
started go straight to the new rung. Provenance and finding rows keep role order.

## Alerts (Slack via `openclaw message send`)

Only: a head failed after all retries, watchdog (an eligible head whose PR has no run in
flight + `max_runs` not reached + nothing started for 30 min, or ingest stale/erroring),
daemon start/stop.
Everything else is in `events` and `reviewsys status`.

Degraded mode is a page, not an alert: only a person can add or re-enable an
OpenAI account. Entering it posts to `page_target` (#claw) with `page_mentions`
(pasta, latte) @-mentioned, and copies the operator DM. The page lists every
account as usable, out, or parked for its reserve, with its reset time. At most
one @-mention page goes out per hour: it re-pages hourly until the mode clears,
and if the probe flaps back into the mode within the hour, only the DM hears about
it. When the probe sees the models answer again, the all-clear goes to the same
places without mentions. An undelivered page is retried on the next pass. If an
operator forces the mode on or off, only the DM hears about it.

## Development

Reviewsys runs Claude Code in plan mode with the per-invocation setting
`permissions.disableAutoMode="disable"`. This disables Claude Code's separate
LLM permission-classifier requests for review lanes and doctor probes while
retaining plan mode and ordinary permission rules. Shared Claude settings and
other applications are unaffected.

    make sync    # uv sync --group dev
    make check   # ruff + mypy --strict + pytest

Tests are hermetic: fake `gh` (scripted responses), fake `claude` lanes
(canned JSON), real SQLite in tmp. `tests/golden/final-review.md` is the exact
review body for the reference two-phase run.

## Deploy

    deploy/deploy.sh                   # live, latest green main commit
    deploy/deploy.sh main --shadow     # ingest + route only, latest green main commit
    deploy/deploy.sh v0.14.0           # optional immutable tag/ref rollback

Deploys a ref (by default `main`) only when its CI `check` run is green, into
`~/.reviewsys` on the Mac Studio (`claw@100.81.48.28`), and restarts the daemon.
The resolved commit is checked out detached, so each deployment remains
reproducible and can be rolled back to a tag or commit ref.
`claw` has no GUI session so launchd never supervises it; the daemon runs
detached (`deploy/start-daemon.sh`) and a one-line cron watchdog
(`deploy/reviewsys-watchdog.sh`, installed by `deploy.sh`) restarts it within a
minute if it dies. `~/.reviewsys/mode` records live/shadow for the watchdog.

## Operations (on the box)

    R="$HOME/.reviewsys/venv/bin/reviewsys --config $HOME/.reviewsys/config.toml"
    $R status [--json]            # queue counts, active runs, watchdog verdict
    $R doctor                     # gh, claude, skills, proxy models
    $R enqueue OWNER/REPO N       # priority review of a PR's live head
    $R cancel --run-id N          # cooperative cancel of an active run
    $R retry OWNER/REPO N         # re-queue a PR whose head ended `failed`
    $R degraded [on|off|auto]     # show / force the stand-in-models mode (see above)
    $R import-legacy PATH         # one-off: mark legacy queue.json `done` rows as reviewed

Where to look when something is off, in order:

1. `sqlite3 ~/.reviewsys/review.db "select ts,kind,repo,number,run_id,substr(detail,1,160) from events order by id desc limit 40"`
2. `~/.reviewsys/daemon.log` (rotated, 20 MB × 5) and `daemon.stderr.log` (crash output only)
3. `~/.reviewsys/work/runs/run-N/attempts/*/` — `prompt.md`, `stdout.json`, `lane-meta.json`
   (`turns`, `cost_usd`, `subtype`) for every lane of a run; a lane's correction turns in its
   `correction-N/`, the repair lane in `repair/`
4. `~/.reviewsys/watchdog.log` — every restart the cron watchdog performed

Config knobs (`~/.reviewsys/config.toml`, restart the daemon after editing):
`max_runs` (safety cap on runs in flight, live + audit; 0 = none),
`max_concurrent` / `priority_overflow` (their sum is the `gpt` lane streams per usable
OpenAI account), `account_scale_max` (1 = no scaling), `account_reserves`,
`page_target` / `page_mentions` (`[identity]`, degraded-mode pages), `debounce_minutes`,
`phase_parallelism` (reviewer lanes per phase per run, 0 = all), `lane_pools` (lanes in flight per
model family across all runs; `gpt` always follows the per-account budget),
`lane_timeout_minutes` (wall-clock bound per lane), `lane_budget_usd` (runaway
guard passed as `claude --max-budget-usd`; it is the CLI's list-price estimate,
not real spend, so keep it well above a normal $3–10 lane). `[lanes] deny_exec` (see
[Reviews are static](#reviews-are-static-lanes-never-build)), `[lanes] correction_turns`
(default 2, 0..5; see [Output contract](#output-contract-normalization-correction-turns-repair)),
`[lanes] claude_config_dir` (the Claude Code config dir lanes keep their sessions in; every
reviewsys process exports it as `CLAUDE_CONFIG_DIR`, so the session cleanup, gc and `doctor` look
where the lanes write. It must match a launcher that forces its own: the box's
`~/.openclaw/bin/claude` sets `~/.claude-proxy`, so the box config sets that; unset = inherited,
else `~/.claude`).

`[retention] artifact_days` (0 = keep run artifacts forever, the default: they are the
only per-lane record for later analysis and grow ~1 GB per two weeks), `worktree_budget_gb`.

Never edit code under `~/.reviewsys/src` on the box; deploy a tag.

`config.toml` wins over the skills repo's `config.json` for any key it defines, so a
slot change made only in the skills repo has no effect on the box. `doctor` prints the
effective `review slots` line (`runs: max N ...; lane pools: ...`) for exactly this
reason. Do not run `reviewsys tick` while the daemon is live -- it refuses, because two
schedulers could start the same head twice; `tick --no-spawn` never schedules and is
safe any time.

### Notes after the legacy cutover (2026-09)

- Legacy reviews were imported with `import-legacy`, so a later push on a PR
  legacy already reviewed is classified `new_push`. Prior *findings* were not
  imported: the first reviewsys round on such a PR is prompted as a first round,
  but `dedupe.find_duplicate` still matches existing inline comments by the
  byte-compatible `finding_hash` marker, so nothing is re-posted.
- The legacy pipeline (`review-dispatcher`, `review_watcher.py`,
  `review-bot-cron`, `review_status_updater.py`, `reviews/queue.json`) is
  decommissioned. Do not restart or repair it.
