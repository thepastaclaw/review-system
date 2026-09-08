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
| `ingest.py` | GitHub GraphQL poll → `prs`/`heads`; notifications → `inbox` |
| `router.py` | inbox → priority heads (`@thepastaclaw review`, review_requested) or own-PR comment batches → OpenClaw wake |
| `scheduler.py` | slots (2 + 1 priority), debounce, single-flight per PR, retries with backoff, supersede/cancel |
| `reaper.py` | heartbeat + deadline enforcement; kills process groups; no slot can be ghosted |
| `worker.py` | one run: worktree → select → triage → context → phase1 → verify1 → gate → phase2 → verify2 → publish (deep backlog: context → phase2 → verify2 → publish) |
| `triage.py` | one cheap lane rates the PR (trivial/low/normal/critical); the tier picks each phase's `--effort` |
| `lane.py` | runs `claude --bare --permission-mode plan` in the worktree, captures JSON + token usage |
| `prompts.py` | assembles prompts from the `thepastaclaw/skills` repo templates |
| `contract.py` | reviewer/verifier JSON contracts; legacy-compatible `finding_hash` / `dedupe_key` |
| `dedupe.py` | within-batch same-root collapse; cross-round matching against existing inline comments |
| `publish.py` | pure `render(ReviewModel)`; diff position mapping; posting; CodeRabbit reactions |
| `github.py` | PR metadata, review threads, evidence bundle, gate/status comment |
| `status.py` | `reviewsys status --json` + watchdog predicate |
| `doctor.py` | gh auth, claude launcher, skills, proxy single-`stop_sequences` probe per model |

State: `~/.reviewsys/review.db` (SQLite, WAL). Tables: `prs, heads, runs, steps, lanes, findings, posted_findings, reviews, inbox, events, kv`.

## Model policy and effort tiers

Models, agents and default reasoning levels come from `review_model_policy` in the
skills repo's `config.json`, read at every config load (no redeploy to change them).
If the policy has a `triage` block, a `gpt-6-astra --effort low` lane rates each PR
and the tier picks the `--effort` of the reviewer lanes; verifiers keep their fixed
level. Blockers always win: a Phase-1 blocker publishes the preliminary review
regardless of tier. `trivial` with no blockers publishes a *final* review from
Phase 1 only and says so in the provenance block. Triage failure falls back to
`fallback_tier` and is recorded as a `triage.degraded` event.

| tier | Phase 1 (glm-5.3-flash) | Phase 2 (gpt-6-astra) |
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

## Alerts (Slack via `openclaw message send`)

Only: a head failed after all retries, watchdog (eligible work + free slot +
nothing started for 30 min, or ingest stale/erroring), daemon start/stop.
Everything else is in `events` and `reviewsys status`.

## Development

    make sync    # uv sync --group dev
    make check   # ruff + mypy --strict + pytest

Tests are hermetic: fake `gh` (scripted responses), fake `claude` lanes
(canned JSON), real SQLite in tmp. `tests/golden/final-review.md` is the exact
review body for the reference two-phase run.

## Deploy

    deploy/deploy.sh v0.3.0            # live
    deploy/deploy.sh v0.3.0 --shadow   # ingest + route only, never starts reviews or wakes the agent

Deploys only a lightweight tag whose CI `check` run is green, into
`~/.reviewsys` on the Mac Studio (`claw@100.81.48.28`), and restarts the daemon.
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
    $R import-legacy PATH         # one-off: mark legacy queue.json `done` rows as reviewed

Where to look when something is off, in order:

1. `sqlite3 ~/.reviewsys/review.db "select ts,kind,repo,number,run_id,substr(detail,1,160) from events order by id desc limit 40"`
2. `~/.reviewsys/daemon.log` (rotated, 20 MB × 5) and `daemon.stderr.log` (crash output only)
3. `~/.reviewsys/work/runs/run-N/attempts/*/` — `prompt.md`, `stdout.json`, `lane-meta.json`
   (`turns`, `cost_usd`, `subtype`) for every lane of a run
4. `~/.reviewsys/watchdog.log` — every restart the cron watchdog performed

Config knobs (`~/.reviewsys/config.toml`, restart the daemon after editing):
`max_concurrent` / `priority_overflow` (slots), `debounce_minutes`,
`lane_timeout_minutes` (wall-clock bound per lane), `lane_budget_usd` (runaway
guard passed as `claude --max-budget-usd`; it is the CLI's list-price estimate,
not real spend, so keep it well above a normal $3–10 lane).

Never edit code under `~/.reviewsys/src` on the box; deploy a tag.

### Notes after the legacy cutover (2026-09)

- Legacy reviews were imported with `import-legacy`, so a later push on a PR
  legacy already reviewed is classified `new_push`. Prior *findings* were not
  imported: the first reviewsys round on such a PR is prompted as a first round,
  but `dedupe.find_duplicate` still matches existing inline comments by the
  byte-compatible `finding_hash` marker, so nothing is re-posted.
- The legacy pipeline (`review-dispatcher`, `review_watcher.py`,
  `review-bot-cron`, `review_status_updater.py`, `reviews/queue.json`) is
  decommissioned. Do not restart or repair it.
