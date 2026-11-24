# Using PastaClaw reviews

For PR authors, maintainers, and clients following **@thepastaclaw** reviews.
You need only GitHub and, optionally, the public dashboard—not access to the
review service's host. This guide describes the current source's defaults;
repository coverage, permissions, scheduling, models, checks, and labels depend
on the deployment. The [README](../README.md) keeps the deeper policy and
operations reference.

## For people: get a useful review

### Quick start for a PR author

1. Open a PR in a repository covered by the service. Keep it draft while it is
   not ready; mark it **Ready for review** for automatic review.
2. Push a coherent, tested change. Ordinary new PRs and pushes queue themselves;
   **do not also post a review command after every push**. Normal automatic
   reviews wait through a **30-minute debounce by default**.
3. Find the bot's stable status comment on the PR (the **gate comment**).
   Follow its link to the [dashboard](https://thepastaclaw.github.io/review-system/)
   for queue position, live steps, and estimates. Check that its commit is your
   current PR head.
4. Read the review's blocker list and inline threads. Fix real blockers, reply
   where clarification or disagreement is needed, test, then push the fixes
   together. The push normally queues the next review automatically.

Automatic discovery covers open, non-draft PRs in enabled repositories. It skips
selected automation authors (Dependabot, Renovate, and GitHub Actions). An
unlisted repository can receive an ad hoc review when a trusted human asks and
the bot has access; a repository explicitly disabled in configuration stays off.
The bot's own PRs have a separate feedback workflow, not the ordinary
mention/reply-triggered review route described here.

**Drafts:** no automatic review starts. Where the gate comment offers them,
collaborators can use its normal or priority checkbox to request a review of the
current draft commit. That requested review survives subsequent draft polls;
a new push to the draft needs another request or a transition to ready. The
current notification router requires a non-draft PR for `@thepastaclaw review`
and GitHub reviewer requests: use the draft's boxes or mark it ready instead.
Returning a PR to draft stops automatic work, not an explicit checkbox request
for that same head.

**Opt out of automatic review:** apply `tag:skip-review` (default; case-insensitive).
Deployments can configure different `skip_labels` or disable them with an empty
list. This stops automatic review, including in-flight automatic work, but is
not a ban on explicit requests. On a non-draft PR an explicit mention or GitHub
reviewer request can still ask for a review. Existing checked boxes can also be
honoured; a skip-labelled gate does not offer new boxes. Removing the label
queues the current head normally, with debounce. A reply on a skip-labelled PR
can discuss an already-reviewed commit, but does not request review of a new one.

### Keep working while it waits or reads

The debounce is a quiet period for each newly observed head, not an appointment
30 minutes after the first push. A new head replaces the older queued head and
gets its own normal debounce. Configuration can change the interval; configured
priority authors and explicit requests can bypass it. Eligibility is not a
promised start time: capacity, an earlier run of this PR, retries, or a service
pause can still make it wait.

Use that time for your own local review and tests: they can overlap the remote
queue. Remote review lanes are static/read-only reviews, not a substitute for
running your tests. There is no need to freeze all edits or hold tested fixes
merely because a review is reading. Batch related, tested fixes into a useful
push rather than sending a stream of tiny pushes that repeatedly replace heads.
If you want feedback on a particular snapshot before changing it, waiting is a
choice—not a general requirement.

A push supersedes old automatic work, but **does not instantly kill every lane**.
Cancellation is cooperative; the scheduler lets runs already in `verify2` or
`publish` continue rather than cancelling them immediately. Workers also check
the live head before publishing and can retire obsolete work without a report.
Publication races or an earlier preliminary report can leave historical output
for the old SHA. Do not expect every superseded run to publish, or treat an
old-head report as a verdict on the new head. A superseded/cancelled run is not
itself a review failure.

### Read the status and verdict

Start with the commit, then the state—not just a green icon or the newest comment.

| What you see | What it means / what to do |
| --- | --- |
| **Review not started yet** | Draft or normal push debounce. Wait, mark ready, or use an offered request box if appropriate. |
| **Queued** | Waiting for eligibility, capacity, or retry backoff. The dashboard explains the wait; priority is not an immediate-start guarantee. |
| **In progress / running** | A run exists; some lanes may still be waiting for model slots. Use the dashboard for moving progress and the check's title for the step. |
| **Preliminary / Phase 2 deferred** | Phase-1 findings stopped the deeper pass. Read the findings and gate explanation, fix/respond, then push. A points gate can defer Phase 2 even with zero canonical blockers; this is not final approval. |
| **Final review complete** | Read the actual verdict: `APPROVE`, `REQUEST_CHANGES`, or informational `COMMENT`. “No blockers” alone does not guarantee approval. Some configured paths finish with Phase 1 only. |
| **Earlier review stands / unchanged diff** | The service explicitly carried a review over a push whose PR diff did not change. Follow the referenced review; do not infer carry-over merely from similar-looking code. |
| **Degraded / paused** | With outage policy `mode: pause`, queued reviews and lanes waiting for a slot wait for recovery; already-running calls may finish. With `mode: standin`, substitute models can produce a degraded report, but do not grant a new approval. See dashboard health. |
| **Could not complete / failed** | No successful review is implied. Check the reason and retry/backoff state; ask once for a retry if needed after the underlying problem clears. |
| **Automatic re-review paused / stepped back** | The review requested a maintainer decision after repeated rounds; its request for changes still stands. Address that discussion rather than blindly retriggering. |

If enabled for the repository, the **PastaClaw review** commit check (name is
configurable) maps queued/running to pending work; completed `success` means
approval, `failure` means changes requested (including a stopped preliminary
pass), and `neutral` means a comment verdict **or a labelled maintainer override**.
`action_required` means the service gave up or ended without a verdict;
`cancelled`/`skipped` do not mean reviewed. GitHub's check **Re-run** button is not
a supported trigger. Checks are deployment-dependent and may be absent on drafts.
Optional `pastaclaw:approved`, `pastaclaw:changes-requested`, and
`pastaclaw:commented` labels help filtering where enabled, but can lag and do not
replace head-specific evidence (notably after a dismissal).

Read the review's **“Changes requested” / “Nothing blocks”** lead, counts, and
linked carried threads, including findings rendered only in the review body.
Blocking findings count toward the verdict; **suggestions and nitpicks are
optional**, not an instruction to fix everything before another push. A
preliminary points gate is a separate reason a deeper pass might not have run.
A disputed blocker can be left to maintainers without granting approval.

**Resolving a GitHub thread is not approving the PR.** A conversation can withdraw
or defer a finding and update a verdict, but “fixed in this proposed patch” is
not proof that the reviewed code is fixed. A later-round approval may still need
a fresh full-diff final pass. Read the current-head review/check rather than
counting resolved threads. A maintainer dismissal is an override, not technical
verification that the findings were fixed.

### Ask for the right kind of work

Use one supported surface, once; check the result before asking again.

| Action | Who / where | Effect |
| --- | --- | --- |
| Ordinary ready PR or push | PR authors, enabled repo | Normal automatic review after debounce (unless configured priority). No extra command needed. |
| **Request normal review** checkbox | Users GitHub permits to edit the bot's gate comment, normally repository collaborators; only where offered | Requests the current head now, bypassing debounce **without priority**. Configured priority authors are an exception. Works on drafts too. |
| **Request priority review** checkbox | Same edit permissions; only where offered | Bypasses debounce and prioritizes scheduling, including lane-slot ordering. Does not interrupt an already-running model call or bypass an outage pause. May spend quota faster if deployment fast-priority support is enabled. |
| Top-level `@thepastaclaw review` | Open, non-draft PR; see trust rules below | Requests priority review of the live head. A fresh, explicit trusted-human ask can also re-review the **same SHA**, not only a new push. |
| Request **thepastaclaw** as a reviewer in GitHub | GitHub users with permission to request reviewers; watched, open, non-draft PR | Priority review of the live head. A still-pending actual reviewer request can re-review a head already reviewed. Repeated notifications alone are not fresh requests. |
| Reply under an existing bot finding | Trusted humans on a non-draft PR, not another bot's thread | Discuss that finding; routing depends on whether the live head has a standing review (below). |
| Top-level comment quoting `> finding title` for a body-only finding | Trusted human, watched non-draft PR | Full review route reads the discussion; on an unreviewed push it keeps normal automatic timing. Prefer the existing inline thread when there is one. |
| Resolve a thread / dismiss a review | Appropriate GitHub permissions | Neither queues a new review nor proves a fix. Dismissal can make an enabled failing bot check neutral as a maintainer override. |

For same-head mention reruns, **trusted human** means GitHub association
`OWNER`, `MEMBER`, `COLLABORATOR`, or `CONTRIBUTOR`, or a configured trusted
reviewer. Write a new top-level comment in your own words, for example
`@thepastaclaw review` or `@thepastaclaw please re-review`. Quoted text, code
examples, thanks mentioning a review, bot comments, and mentions inside finding
threads do not qualify as that same-head rerun request. Each mention comment is
handled once; same-SHA reruns are capped at **three per trigger type** (mentions
and reviewer requests have separate caps). An initial mention on a watched repo
is not subject to this same-head trust gate; ad hoc requests on unlisted repos
require trust. Do not rely on an automated bot comment to force a same-head rerun.

### Discuss findings where they were raised

Reply in the **existing finding thread** with a concrete question, the reason
you disagree, or fix evidence (commit link, test, relevant code). For example:

> This path is only reachable after the ownership check in `load_record`.
> Does that change the finding? The regression test is in commit `<sha>`.

On an already-reviewed live head, a trusted human reply normally uses the
**conversation lane**: it reads the exchange and code and answers there, rather
than starting the full review pipeline. It can keep or withdraw the finding,
recognize a pending fix, or consider a reasonable deferral. A proposed/unpushed
fix remains **fix pending**, not fixed. Deferrals generally need tracking; for
an accepted maintainer deferral the service may find or create a follow-up issue
where permitted, so asking to defer is not always side-effect-free.

After an unreviewed push, “fixed in `<sha>`” does **not** accelerate or prioritize
the automatic review; that normal review reads the reply. A reply during a run
waits for routing after that run ends. On repositories reviewed only on request,
a reply may instead ask for review. Replies from other bots are context, not
human requests for an answer. A top-level reply about an ordinary inline finding
is less reliable than answering its thread.

### If something seems stuck

- Compare the PR head with the gate/review commit. A push may have replaced the
  run you were watching. Refresh the dashboard and inspect the matching SHA.
- Check draft/skip-label state, repository coverage, debounce/backoff, and the
  dashboard's queue reason before issuing another request.
- Check dashboard health for **pause/degraded mode**, stale ingest, or a stuck
  warning. Progress/ETAs are estimates, not deadlines; a running lane can take
  longer than the historical estimate.
- If a request appears ignored, check edit permission, trust, the draft rules,
  and same-head limits. Do not keep checking boxes or posting commands.
- For a genuine failure, read the reason; a supported request can retry, while
  a service outage needs recovery rather than more requests. Escalate to the
  maintainers with PR URL, current SHA, gate/review link, and observation time.

## For agents and integrations: read status, don't guess

This is an **external-client reference**, not a service administration guide.
The [public dashboard](https://thepastaclaw.github.io/review-system/) and
[JSON snapshot](https://thepastaclaw.github.io/review-system/data/status.json)
are unauthenticated, read-only GitHub Pages exports. They are **not an
authenticated REST mutation API**; there are no review-service enqueue/cancel
endpoints documented here. Clients need no host CLI, database, or queue access.
Use the GitHub surfaces above for intentional control, with ordinary GitHub
permissions and the stated actor restrictions.

### Snapshot schema: useful fields

The export is built by [exporter.py](../reviewsys/exporter.py) and
[status.py](../reviewsys/status.py). Schema version **1** currently exports:

| Location | Useful fields and interpretation |
| --- | --- |
| Top level | `schema_version`, `data_as_of` (observation time, from `live.ts`), `generated_at` (export construction time), `live`, `history`, `rounds`; optional `shadow` comparison data. Times are UTC ISO-8601 strings. |
| `live.active[]` | `repo`, numeric `number`, full `sha`, `id` (run ID), `queue` (`live` versus post-merge `audit`), `status` (`spawned`/`running`), `phase`, `trigger`, `priority`, `attempt`, `tier`, `started_at`, `queued_at`, `pr_url`. Filter `queue == "live"` for PR review work. |
| Active timing/progress | `heartbeat_at`, `heartbeat_age_seconds`, `deadline_at`, `deadline_in_seconds`, `elapsed_seconds`, `wait_seconds`; `progress` (fraction, not percent; nullable), `remaining_seconds` (nullable), `overdue`, `upcoming`. `steps`, `lanes`, and `live_lanes` give more detail; a live lane may be waiting rather than running. |
| Active results so far | `findings` (nullable until a verification finishes), `tokens_in`, `tokens_out`, `degraded`. Intermediate findings and lane failures are not the final run verdict. |
| `live.queued[]` | `repo`, `number`, `sha`, `position`, `priority`, `trigger`, `queued_at`, `eligible_at`, `eligible`, `eligible_in_seconds`, `age_seconds`, `attempts`, `reason`, `pr_url`, `prioritize_url`. No active-run `phase`, `started_at`, or `progress` yet. `prioritize_url` is a GitHub comment link, not an API operation. |
| Queue reason | Currently `debounce (more pushes may follow)`, `waiting for a review slot`, or `retry backoff after attempt N`. Interpret with global health: this reason alone does not describe an outage pause. `eligible_at` is an earliest eligibility time, not an ETA. |
| `live` health | `ingest_last_at`, `notify_last_at`, `ingest_error_streak`, `failed_heads_24h`; `watchdog.ingest_stale`, `.stuck`, `.eligible`, `.active`, `.idle_minutes`. No top-level `health` object. |
| `live.degraded` | `configured`, `active`; when configured: `mode` (`pause`/`standin`), `forced`, `since`, `sentinel`, `last_probe` (`at`, `quota_exhausted`, `reason`); stand-in mode also supplies `substitutes` and `phase1_effort_cap`. Missing mode on an unconfigured deployment is normal. |
| Other summaries | `live.capacity`, `lane_pools` (running/waiting/slots), `heads`, `priority_queued`, `last_run_started_at`, `last_run_done_at`, `median_run_minutes`, `typical_minutes_by_tier`, `today`, `ci_watches`, `audit`. These are aggregate hints, not proof about one head. |
| History | `history.daily`, `totals`, `recent_events`, token/finding summaries, `path_timing`, and `rounds` support dashboard history. They are not a complete authoritative per-head completion API. |

Fields can be absent/null where no observation or estimate exists. Support the
schema version explicitly and tolerate extra fields; reject malformed required
identity/timestamp data rather than interpreting it as an empty queue.

### Read-only example: look up one PR's live head

Requires standard `gh`, `curl`, and `jq` clients, not a review-host installation.
`gh api` below **only reads GitHub**; authenticate normally if the repository
requires it. Change the repository/number. This prints evidence, not a verdict:

```sh
repo=dashpay/platform
number=5536
head=$(gh api "repos/$repo/pulls/$number" --jq .head.sha) || exit 1
snapshot=$(mktemp) || exit 1
trap 'rm -f "$snapshot"' EXIT
curl --fail --silent --show-error --connect-timeout 10 --max-time 30 \
  https://thepastaclaw.github.io/review-system/data/status.json \
  -o "$snapshot" || exit 1
jq -e '.schema_version == 1 and
  (.data_as_of | type == "string") and
  (.live.active | type == "array") and (.live.queued | type == "array")' \
  "$snapshot" >/dev/null || exit 1
jq --arg repo "$repo" --argjson number "$number" --arg sha "$head" '
  def matches: (.repo | ascii_downcase) == ($repo | ascii_downcase)
    and .number == $number and .sha == $sha;
  {schema_version, data_as_of, generated_at,
   health: (.live | {ingest_last_at, ingest_error_streak, watchdog, degraded}),
   matching_active: [.live.active[] | select(matches and .queue == "live") |
     {repo, number, sha, id, phase, started_at, progress, remaining_seconds}],
   matching_queued: [.live.queued[] | select(matches) |
     {repo, number, sha, eligible_at, reason, position}]}
' "$snapshot"
```

The public snapshot fetched on **2026-11-24**, at
`data_as_of: 2026-11-24T10:36:00Z`, contained a queued entry for
`dashpay/platform#5536`, SHA `97875d14db9d952738cc0a246806d2e259df9cf1`,
with `eligible_at: 2026-11-24T10:37:57Z` and
`reason: "debounce (more pushes may follow)"`. This is a historical example of
real output, **not** a claim that this PR is still queued. The same snapshot had
`live.degraded.mode: "pause"`, `active: false`.

### Freshness and evidence rules

- Publishing is intended roughly once a minute; Pages deployment and edge/client
  caches add delay. The fetched response advertised `Cache-Control: max-age=600`
  and an `ETag`. A successful HTTP response or new fetch time does **not** make
  the embedded data current. Inspect both `data_as_of` and `generated_at`, then
  ingest/probe/heartbeat timestamps. Check clock skew and parse errors too.
- Poll conservatively (for example once per minute at most, with jitter), respect
  cache headers, and use conditional requests (`ETag`/`Last-Modified`) where
  available. A `304` reuses the old snapshot and its old observation time. Back
  off exponentially on errors/rate limits, honour `Retry-After`, and stop or slow
  down when no longer following a PR. Do not cache-bust in a tight loop.
- Match **repository + PR number + full SHA** to GitHub's live PR head. Re-read
  that head before making a completion claim: it may have moved since the first
  lookup. Never join on PR number or a short SHA alone; exclude audit work.
- Empty matching arrays mean **no matching live evidence in this snapshot**,
  not “verified idle”, “review completed”, or “approved”. Missing, stale, or
  conflicting evidence is **unknown**. The export can omit work not represented
  by those lists (drafts, skipped heads, completed work, CI follow-ups), and
  deployment/data failures can leave an old or minimal snapshot.
- `progress`, remaining time, queue position, medians, and tier timings are
  historical estimates. They may change, saturate below completion, or be null;
  `overdue` is not failure. Neither 100% nor disappearance from `active` proves
  publication. Confirm a head-specific GitHub review/check, or explicit
  unchanged-diff carry-over, before reporting a verdict.

### GitHub evidence and controls

The stable gate issue comment contains **`<!-- thepastaclaw-gate v1 -->`**.
Validate the comment's **author login** as `thepastaclaw` (or the deployment's
configured bot), as well as the marker; another user's copied marker is not
service evidence. Read-only lookup:

```sh
gh api --paginate "repos/$repo/issues/$number/comments?per_page=100" --jq '
  .[] | select(.user.login == "thepastaclaw" and
    (.body | contains("<!-- thepastaclaw-gate v1 -->"))) |
  {html_url, updated_at, body}'
```

Then validate the commit mentioned in the body against the live head. The gate
is one mutable comment per PR, not an append-only completion event stream. For
reviews, validate author and reviewed `commit_id`; for checks, match `head_sha`
and the expected check name/App identity, and read conclusion **and summary**.
An explicit carried review is different from merely finding an older approval.
When GitHub and the export disagree, do not invent a composite “complete” state.

Mutations are the supported **GitHub actions in the human table above**: pushing
or marking ready can schedule automatic work; checking a gate box, posting an
explicit qualifying mention, or requesting the reviewer can schedule new work
and consume quota; replies can start conversation/full review, change a verdict,
resolve threads, or file accepted deferrals' tracking issues. Observe rather
than issue redundant commands after ordinary automatic pushes. Agents do not
inherit trusted-human status by quoting a human command. None of this requires
editing service queues/DBs or running daemon/admin commands locally.

### Source and further detail

Behavior reference: [ingest](../reviewsys/ingest.py),
[request routing](../reviewsys/router.py),
[gate checkboxes](../reviewsys/queue_status.py),
[scheduling](../reviewsys/scheduler.py),
[worker publication/conversation guards](../reviewsys/worker.py),
[GitHub rendering](../reviewsys/github.py),
[check meanings](../reviewsys/checks.py), and
[outage state](../reviewsys/degraded.py).
Defaults live in [configuration](../reviewsys/config.py); these are not proof
of any deployment's private settings. Existing extended references remain in
[Replies to findings](../README.md#replies-to-findings),
[Queue comment and priority requests](../README.md#queue-comment-and-priority-requests),
[Skip label](../README.md#skip-label-prs-that-must-not-be-reviewed), and
[Check run](../README.md#check-run-review-progress-for-tools-and-the-merge-box).
