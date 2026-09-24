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
| `scheduler.py` | slots (2 + 1 priority), debounce, single-flight per PR, retries with backoff, supersede/cancel |
| `reaper.py` | heartbeat + deadline enforcement; kills process groups; no slot can be ghosted |
| `worker.py` | one run: worktree → select → triage → context → phase1 → verify1 → gate → phase2 → verify2 → publish (deep backlog: context → phase2 → verify2 → publish) |
| `triage.py` | one cheap lane rates the PR (trivial/low/normal/critical); the tier picks each phase's `--effort` |
| `quota.py` | Phase-1 model ladder: remaining Antigravity / Z.AI quota via the proxy's management `api-call`; first rung with quota runs |
| `lane.py` | runs `claude --bare --permission-mode plan` in the worktree, captures JSON + token usage |
| `prompts.py` | assembles prompts from the `thepastaclaw/skills` repo templates |
| `contract.py` | reviewer/verifier JSON contracts; legacy-compatible `finding_hash` / `dedupe_key` |
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
If the policy has a `triage` block, a `gpt-6-astra --effort low` lane rates each PR
and the tier picks the `--effort` of the reviewer lanes; verifiers keep their fixed
level. Blockers always win: a Phase-1 blocker publishes the preliminary review
regardless of tier. `trivial` with no blockers publishes a *final* review from
Phase 1 only and says so in the provenance block. Triage failure falls back to
`fallback_tier` and is recorded as a `triage.degraded` event. `normal` is the
default tier; `critical` requires both a large or intricate diff *and* a change to
a critical surface (consensus, funds, cryptography, key handling, peer-facing
deserialization, migrations), and the triage must name what meets the bar. Size
alone never qualifies, and the tie-break is the lower tier. This replaced the
"large *or* sensitive, when unsure go higher" wording on 2026-09-10 after 87 % of
runs came out critical.

| tier | Phase 1 (ladder rung, capped by the rung's `reasoning`) | Phase 2 (gpt-6-astra) |
|---|---|---|
| trivial | high | skipped |
| low | high | medium |
| normal | max | high |
| critical | max | xhigh |

Efforts are validated against `low|medium|high|xhigh|max` at load. Every lane's
effort is stored in `lanes.effort`, the tier in `runs.tier`, and both are printed
in the review's provenance block and the gate comment. Without a `triage` block the
policy behaves as a single `normal` tier at the configured reasoning levels.

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
"source: gpt-6-astra (phase2-reviewer: general); coderabbit".

## Backlog mode: Phase 2 only

`backlog_skip_phase1_above` in `config.toml` (default 10, `0` disables) trades depth
for throughput. When a run starts and more heads than that are queued, the worker
records the `phase1` step as `skipped`, emits `phase1.skipped_backlog`, and goes
straight to the Phase-2 gpt-6-astra reviewers and final verifier. The verifier is
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

## Degraded mode: stand-in models when the OpenAI pool is dry

`gpt-6-astra` sits on the critical path of every run (triage, Phase-1 gate verifier,
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
  quota error mid-run (reviewer, verifier, triage or selector) flips the run over on
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
- It answers per thread with `STILL_VALID | FIXED | WITHDRAWN | INTENTIONALLY_DEFERRED |
  NO_REPLY` and free prose. The rules it is given: verify claims against the code, never
  repeat a point already made, evaluate a proposed change on its merits and say whether it
  resolves the concern, concede when wrong, stay silent (`NO_REPLY`) when a reply is
  addressed to someone else or there is nothing new to add. Severity cannot change in a
  reply.
- Posting: the prose is the comment (marker `thepastaclaw-thread-answer v1`, once per
  human reply and head, `@` defused, CodeRabbit retriggers discarded), with a quiet
  trailing note for FIXED / WITHDRAWN / INTENTIONALLY_DEFERRED instead of a bold verdict
  lead; FIXED and WITHDRAWN threads are resolved (where the bot may). `NO_REPLY` posts
  nothing, is recorded as `thread.answered` with `action: no_reply`, and the decision is
  remembered per (finding, reply) in `kv` (`converse.silent:<repo>#<n>:<hash>`), so a later
  conversation on the same PR never posts a late second opinion to that message; the
  thread is re-considered only when someone speaks on it again; markers for PRs with no
  head queued inside the artifact retention window are pruned by `gc`. Replies from other
  bots (`*[bot]` logins, CodeRabbit) are part of the transcript but never count as a human
  waiting for an answer, and the router does not queue a `review_reply` head for them at
  all, so no bot-to-bot loop can start.
- Verdict: a conversation never adds blockers and never approves. When every blocking
  finding on the commit has been withdrawn or resolved and the standing review is
  `CHANGES_REQUESTED`, the same short "Re-review after discussion" follow-up moves it to
  COMMENT, with provenance stating that no code was re-reviewed. A concession is
  persisted as a `conceded`-stage `findings` row for the sha, so blockers conceded in
  earlier conversations stay lifted; the standing set is the latest *final* publication's
  blockers for the sha plus unresolved blocking threads this database has no row for
  (legacy findings). A blocking thread a maintainer resolved by hand, without the bot
  conceding it, still counts. Known gap: if the worker dies between posting a concession
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
estimated start time (slot-pipeline model over the median of recent run durations,
plus any debounce/backoff still to elapse) and an estimated review duration. The
comment carries a checkbox, **Request priority review**; ticking it promotes the
head to the priority lane (front of the queue, extra overflow slot, no debounce),
records `head.priority_requested` with the editor's login from the comment's edit
history, and re-renders the comment without the box. GitHub only renders the box as
clickable to users who can edit the comment, i.e. repository collaborators. The
same comment is reused by the worker for "in progress" / "done" / "failed" states,
so a PR never has more than one bot status comment. Writes are capped at 25 per
pass to respect GitHub's content-creation limits; the rest catch up next pass.

## Liveness model

The daemon spawns `reviewsys worker --run-id N` in its own session. The worker
heartbeats every 15 s and checks for cancellation; the reaper fails any run
whose heartbeat is older than 5 min, whose deadline passed, or whose process is
gone, and requeues the head with backoff (`infra` up to 3 attempts, `contract`
twice, `fatal` never). Slot counts are recomputed from the DB every tick.

Slots scale with the OpenAI (Codex) accounts that can take work, because the
primary models are limited per ChatGPT account (~3 concurrent streams, weekly
quota), not globally. Each usable account adds `max_concurrent` normal slots
plus `priority_overflow` slots a priority head may claim (2 + 1), for at most
`account_scale_max` accounts (3, so 6 + 3). The total is a hard ceiling no head
of any kind crosses. Every 2 min the daemon reads the proxy's `auth-files`
(its record of each credential's cooldowns and `X-Codex-*-Used-Percent` quota
headers; nothing is sent to OpenAI). An account is usable when it is enabled, not
parked, and not cooling down for itself or for the sentinel model: it is spent to
the end. The exception is `account_reserves` in the box's `config.toml`, keyed by account
email (the list lives only on the box, since this repository is public): at
that floor reviewsys *disables the account in the proxy*, so no client eats into
the reserve, and re-enables it once the window that hit the floor has reset. It
alerts on both, and it never re-enables an account an operator disabled. With no
reading, or one older than 20 min,
capacity falls back to one unit (2 + 1). A backlog (>10 queued) lends an overflow
slot to normal work only while one remains free for priority, so the priority lane
stays reserved however long the queue gets. `reviewsys status` prints the
effective `slots:` line.

## Alerts (Slack via `openclaw message send`)

Only: a head failed after all retries, watchdog (eligible work + free slot +
nothing started for 30 min, or ingest stale/erroring), daemon start/stop.
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
   (`turns`, `cost_usd`, `subtype`) for every lane of a run
4. `~/.reviewsys/watchdog.log` — every restart the cron watchdog performed

Config knobs (`~/.reviewsys/config.toml`, restart the daemon after editing):
`max_concurrent` / `priority_overflow` (slots per usable OpenAI account),
`account_scale_max` (1 = static slots), `account_reserves`, `page_target` / `page_mentions`
(`[identity]`, degraded-mode pages), `debounce_minutes`,
`lane_timeout_minutes` (wall-clock bound per lane), `lane_budget_usd` (runaway
guard passed as `claude --max-budget-usd`; it is the CLI's list-price estimate,
not real spend, so keep it well above a normal $3–10 lane).

Never edit code under `~/.reviewsys/src` on the box; deploy a tag.

`config.toml` wins over the skills repo's `config.json` for any key it defines, so a
slot change made only in the skills repo has no effect on the box. `doctor` prints the
effective `review slots` line for exactly this reason. Do not run `reviewsys tick` while
the daemon is live -- it refuses, because two schedulers would each admit up to the
ceiling; `tick --no-spawn` never schedules and is safe any time.

### Notes after the legacy cutover (2026-09)

- Legacy reviews were imported with `import-legacy`, so a later push on a PR
  legacy already reviewed is classified `new_push`. Prior *findings* were not
  imported: the first reviewsys round on such a PR is prompted as a first round,
  but `dedupe.find_duplicate` still matches existing inline comments by the
  byte-compatible `finding_hash` marker, so nothing is re-posted.
- The legacy pipeline (`review-dispatcher`, `review_watcher.py`,
  `review-bot-cron`, `review_status_updater.py`, `reviews/queue.json`) is
  decommissioned. Do not restart or repair it.
