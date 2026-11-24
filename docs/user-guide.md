# PastaClaw review reference

For PR authors and clients using **@thepastaclaw**. Defaults below are configurable; repository coverage, trust, checks, and labels depend on deployment. No review-host access needed. [Detailed policy/operations reference](../README.md).

## Comment commands and discussion

**`/skip-bots` has no built-in effect on Reviewsys.** Another repository automation may interpret it separately. Use draft state or the default `tag:skip-review` label to opt out of automatic review (explicit-request exceptions below). Reviewsys has no pause/resume/cancel or normal/priority slash commands.

| Exact comment text / placement | Effect and restrictions |
| --- | --- |
| `@thepastaclaw review` — top-level PR comment | Requests **priority** review of the live head on an open, non-draft PR. Works in enabled repos; an unlisted repo requires a trusted requester and bot access. Explicitly disabled repos stay off. |
| `@thepastaclaw please re-review` or `@thepastaclaw, could you please re-review?` — new top-level comment | Explicit same-head rerun forms; `@thepastaclaw review` also qualifies. Requires a **trusted human**, asking in their own words. Once per comment; at most **3 same-SHA reruns per trigger type** (mentions and reviewer requests have separate caps). |
| Plain reply in the **original bot finding thread**; no command needed | Trusted humans can ask questions, disagree, or give fix evidence. On an already-reviewed live head, normally starts **conversation**, not full review. After an unreviewed automatic push, the normal review reads the reply without bypassing debounce. During a run, routing waits for it to end. Non-draft PRs only; other bots' replies do not trigger discussion. |
| `> exact finding title` followed by your response — top-level comment | For a finding published only in the review body. Trusted human, enabled non-draft PR: requests a full same-head review, or leaves an unreviewed push's normal timing intact. Prefer the original thread when available. |

**Trusted human:** GitHub association `OWNER`, `MEMBER`, `COLLABORATOR`, or `CONTRIBUTOR`, or a configured trusted reviewer. Initial mentions on watched repos do not use the same-head trust gate. Bot comments, quoted/code examples, thanks mentioning a review, and finding-thread mentions are **not** explicit same-head full-review asks. The bot's own PRs use a separate feedback route.

## Other controls (not textual comment commands)

| GitHub action | Effect and restrictions |
| --- | --- |
| Tick **Request normal review** in the bot's status comment | Where offered, requests the current head **without debounce or priority** (configured priority authors excepted). Requires permission to edit that comment, normally repository collaborators. Works on drafts. |
| Tick **Request priority review** in that comment | Same permissions; bypasses debounce and prioritizes scheduling/lane slots. May spend quota faster with configured fast-priority support. Does not interrupt running model calls or bypass an outage pause. |
| Request **thepastaclaw** as a GitHub reviewer | Users with GitHub permission to request reviewers; enabled, open, non-draft PR. Priority review; an actual pending request can rerun the same head. Repeated notifications alone are not new requests. |
| Keep/return PR to **Draft** | No automatic review. A checkbox-requested review of the current draft head survives; a new draft push needs another request or marking ready. Mention/reviewer-request routing requires a non-draft PR. |
| Add **`tag:skip-review`** label | Default automatic opt-out, case-insensitive; `skip_labels` is configurable/disableable. Stops automatic work, but non-draft explicit mentions/reviewer requests still work. Already-checked boxes may be honoured; labelled gates offer no new boxes. Replies can discuss reviewed commits, not request review of new ones. Removing the label queues normally. |
| Mark ready / push | Eligible PRs review automatically; **do not add a redundant trigger after an ordinary push**. |

## Timing, results, and follow-up

- **Eligibility:** open, non-draft PR in an enabled repo, without an opt-out label. Selected automation authors (Dependabot/Renovate/GitHub Actions) are skipped.
- **Normal debounce: 30 minutes** by default, from each newly observed head. New pushes replace older queued heads; configured priority authors and explicit requests bypass debounce. Eligibility is not a promised start time.
- **Keep working:** local review/tests can overlap debounce and static/read-only remote review. Push batched, tested fixes; no blanket edit/push freeze is needed.
- **Superseded runs:** cancellation is not instant; late-stage work may continue, but publication guards can retire an obsolete head without a report. Never expect a stale report or use an old-head verdict for the new head.

| Status | Interpretation |
| --- | --- |
| Not started / queued | Draft, debounce, capacity wait, or retry backoff; inspect the dashboard reason. |
| Running | A run exists; individual lanes may still be waiting for model slots. |
| Preliminary / Phase 2 deferred | Deeper review stopped at the findings/points gate, possibly with zero canonical blockers. **Not final approval.** |
| Final | Read the actual verdict: `APPROVE`, `REQUEST_CHANGES`, or `COMMENT`. “No blockers” alone is not approval. |
| Earlier review stands | Explicit unchanged-diff carry-over; follow the referenced review. |
| Degraded / paused | `pause`: queued work/waiting lanes wait for recovery. `standin`: substitute-model reports do not grant a new approval. Running calls may finish. |
| Failed / could not complete | No successful verdict implied; read the reason/backoff, then request a retry if appropriate after recovery. |
| Automatic re-review paused / stepped back | A maintainer decision is needed; its request for changes stands. Address the discussion rather than retriggering blindly. |

- **Read the current-head** gate, review lead/counts, and linked findings (including body-only findings). Blockers count; suggestions/nitpicks are optional. Resolving a thread **is not approval**; proposed/unpushed fixes remain pending. Accepted deferrals may create tracking issues; discussion can update a verdict, but later-round approval may still need a fresh full-diff pass.
- **Optional commit check** (default name `PastaClaw review`): `success` = approval; `failure` = changes requested/stopped preliminary pass; `neutral` = comment or explicitly identified maintainer override, not proof of a fix. `action_required` = service gave up/no verdict; `cancelled`/`skipped` ≠ reviewed. GitHub's **Re-run** button is not supported.
- **If stuck:** check SHA, draft/label, coverage/trust, debounce/backoff, and dashboard health. Escalate with PR URL, SHA, status link, and time; don't spam requests. Optional verdict labels can lag and are not head-specific proof.

## Agent / integration reference (read-only)

- Dashboard: **https://thepastaclaw.github.io/review-system/**
- JSON snapshot: **https://thepastaclaw.github.io/review-system/data/status.json**
- Public GitHub Pages export, **not an authenticated REST mutation API**. Controls are the GitHub actions above; clients need no service CLI, database, or queue access. Automated actors do not inherit human trust by copying commands.

| Export fields (schema version 1) | Meaning |
| --- | --- |
| `schema_version`, `data_as_of`, `generated_at` | Version, observation time, export construction time; UTC ISO-8601. Other top-level summaries: `history`, `rounds`, optional `shadow`. |
| `live.active[]` | `repo`, `number`, `sha`, `id`, `queue`, `status`, `phase`, `started_at`, `priority`, `heartbeat_at`, `progress`, `remaining_seconds`, `overdue`, `steps`, `live_lanes`, `degraded`. Filter `queue == "live"`, excluding audits. Progress is a nullable fraction, not completion proof. |
| `live.queued[]` | `repo`, `number`, `sha`, `position`, `priority`, `trigger`, `queued_at`, `eligible_at`, `eligible`, `attempts`, `reason`. Eligibility is not an ETA; queued entries have no run phase/progress yet. |
| Health under `live` | `ingest_last_at`, `notify_last_at`, `ingest_error_streak`, `watchdog.ingest_stale`/`.stuck`, `degraded.active`/`.mode`/`.last_probe`, `capacity`, `lane_pools`. Degraded mode can be absent when unconfigured. |

**Evidence rules:** match **repo + PR number + full SHA** to GitHub's live PR head, and recheck that head before claiming completion. Missing/stale/conflicting data or empty matches mean **unknown**, not verified idle/completed. Confirm a matching GitHub review/check or explicit carry-over; estimates and disappearance from `active` are not verdicts.

**Freshness:** inspect `data_as_of`/`generated_at` and ingest/heartbeat times. Publication is roughly minute-based, but Pages/caches delay it (observed `max-age=600`). Poll at most once/minute with jitter, respect caching/conditional responses, and back off on errors/`Retry-After`; a `304` does not refresh observation time.

Read-only lookup (requires `gh`, `curl`, `jq`; prints evidence, **not a verdict**):

```sh
repo=dashpay/platform; number=5536
head=$(gh api "repos/$repo/pulls/$number" --jq .head.sha) || exit 1
curl -fsS --connect-timeout 10 --max-time 30 https://thepastaclaw.github.io/review-system/data/status.json |
  jq --arg repo "$repo" --argjson n "$number" --arg sha "$head" '
    {schema_version, data_as_of, generated_at, matches:
      [.live.active[], .live.queued[] | select(
        (.repo | ascii_downcase) == ($repo | ascii_downcase) and .number == $n and .sha == $sha
        and ((.queue // "live") == "live"))]} '
```

**Gate identification:** require marker `<!-- thepastaclaw-gate v1 -->` **and author** `thepastaclaw` (or configured bot); a copied marker is not evidence. This is one mutable status comment per PR. Validate its commit; for reviews validate author/`commit_id`, for checks `head_sha` and expected name/App identity.

Sources: [command routing](../reviewsys/router.py), [checkboxes](../reviewsys/queue_status.py), [eligibility/defaults](../reviewsys/ingest.py), [export schema](../reviewsys/exporter.py). More detail: [finding replies](../README.md#replies-to-findings), [skip label](../README.md#skip-label-prs-that-must-not-be-reviewed), [checks](../README.md#check-run-review-progress-for-tools-and-the-merge-box).
