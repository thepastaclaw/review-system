"""The watch: a review-now label or a ticked gate box runs its pass within seconds."""

import json
import subprocess

from reviewsys import watch
from reviewsys.daemon import Daemon
from reviewsys.gh import Gh
from reviewsys.github import GATE_MARKER
from reviewsys.ingest import REVIEW_NOW_LABEL

REPO = "dashpay/platform"
EVENTS = f"repos/{REPO}/issues/events?per_page=30"
COMMENTS = f"repos/{REPO}/issues/comments?sort=updated&direction=desc&per_page=30"
A = "a" * 40
TICKED = f"{GATE_MARKER}\n- [x] **Request normal review**\n"
UNTICKED = f"{GATE_MARKER}\n- [ ] **Request normal review**\n"


class Api:
    """`gh api -i` replies per endpoint: a 304 unless the listing changed since its ETag."""

    def __init__(self, remaining=4000):
        self.listings = {EVENTS: [], COMMENTS: []}
        self.versions = {EVENTS: 0, COMMENTS: 0}
        self.remaining = remaining
        self.calls = []

    def set(self, endpoint, body):
        self.listings[endpoint] = body
        self.versions[endpoint] += 1

    def __call__(self, argv, stdin, timeout):
        args = list(argv[1:])
        endpoint = args[2]
        etag = args[args.index("-H") + 1].split(": ", 1)[1] if "-H" in args else ""
        self.calls.append((endpoint, etag))
        tag = f'"{self.versions.get(endpoint, 0)}"'
        head = (
            f"x-ratelimit-limit: 5000\nx-ratelimit-remaining: {self.remaining}\n"
            f"x-ratelimit-reset: 1000000000\netag: {tag}"
        )
        if etag == tag:
            return subprocess.CompletedProcess(
                argv, 1, f"HTTP/2.0 304 Not Modified\n{head}\n\n", ""
            )
        body = json.dumps(self.listings.get(endpoint, []))
        return subprocess.CompletedProcess(argv, 0, f"HTTP/2.0 200 OK\n{head}\n\n{body}", "")


def labelled(eid, number=7, name=REVIEW_NOW_LABEL, pr=True):
    issue = {"number": number, **({"pull_request": {"url": "u"}} if pr else {})}
    return {"id": eid, "event": "labeled", "label": {"name": name}, "issue": issue}


def gate(at, body, login="thepastaclaw"):
    return {"updated_at": at, "user": {"login": login}, "body": body, "issue_url": "x/issues/7"}


def watcher(api, clock=lambda: 0.0):
    return watch.Watcher(Gh("gh", api), "thepastaclaw", [REPO], clock=clock)


def test_parse_reads_status_headers_and_body():
    res = watch.parse('HTTP/2.0 200 OK\r\nETag: "x"\r\nX-Ratelimit-Remaining: 9\r\n\r\n[1]')
    assert (res.status, res.headers["etag"], res.headers["x-ratelimit-remaining"]) == (
        200,
        '"x"',
        "9",
    )
    assert res.body == [1]
    assert watch.parse("HTTP/2.0 304 Not Modified\netag: x\n\n").status == 304


def test_a_new_review_now_label_on_a_pr_runs_ingest():
    api = Api()
    api.set(EVENTS, [labelled(10)])  # before the watch started: only sets where new ones start
    w = watcher(api)
    assert w.poll() == set()
    assert w.poll() == set()  # unchanged: 304s, sent with the ETag
    assert set(api.calls[-2:]) == {(EVENTS, '"1"'), (COMMENTS, '"0"')}
    api.set(EVENTS, [labelled(12, name="bug"), labelled(11, pr=False), labelled(10)])
    assert w.poll() == set()  # another label, an issue's label
    api.set(EVENTS, [labelled(13), labelled(12, name="bug")])
    assert w.poll() == {watch.INGEST}
    assert w.spent == [0.0, 0.0, 0.0, 0.0]  # the 200s only


def test_a_ticked_gate_box_runs_the_queue_pass():
    api = Api()
    api.set(COMMENTS, [gate("2026-11-27T10:00:00Z", TICKED)])
    w = watcher(api)
    assert w.poll() == set()
    api.set(COMMENTS, [gate("2026-11-27T10:01:00Z", UNTICKED)])  # the bot's own rewrite
    assert w.poll() == set()
    api.set(COMMENTS, [gate("2026-11-27T10:02:00Z", TICKED, login="knst")])  # not a gate
    assert w.poll() == set()
    api.set(COMMENTS, [gate("2026-11-27T10:03:00Z", TICKED)])
    assert w.poll() == {watch.QUEUE}


def test_it_spends_little_and_never_the_reviews_share():
    api = Api(remaining=499)  # under 10% of 5000 left
    wall = [0.0]
    w = watch.Watcher(Gh("gh", api), "thepastaclaw", [REPO], wall=lambda: wall[0])
    w.poll()
    seen = len(api.calls)  # the reads that saw it, then nothing
    w.poll()
    assert len(api.calls) == seen
    api.remaining = 4000
    wall[0] = 2e9  # past x-ratelimit-reset: a new window, the same watcher reads again
    w.poll()
    assert len(api.calls) == seen + 2
    now = [0.0]
    w = watcher(api, clock=lambda: now[0])
    w.spent = [0.0] * 250  # 5% of the limit in the last hour
    calls = len(api.calls)
    assert w.poll() == set() and len(api.calls) == calls
    now[0] = watch.HOUR_S
    w.poll()
    assert len(api.calls) == calls + 2


def test_a_failed_read_keeps_its_place_and_the_others_their_hits():
    api = Api()
    w = watcher(api)
    w.poll()
    real = api.__call__

    def broken(argv, stdin, timeout):
        if "comments" in argv[3]:
            raise subprocess.TimeoutExpired(argv, timeout)
        return real(argv, stdin, timeout)

    w.gh = Gh("gh", broken)
    api.set(EVENTS, [labelled(5)])
    api.set(COMMENTS, [gate("2026-11-27T10:03:00Z", TICKED)])
    assert w.poll() == {watch.INGEST}
    w.gh = Gh("gh", api)
    assert w.poll() == {watch.QUEUE}  # its ETag and mark were not moved by the failure


def test_ingest_runs_again_after_a_label_hit_for_a_lagging_graphql():
    api = Api()
    now = [0.0]
    w = watcher(api, clock=lambda: now[0])
    w.poll()
    api.set(EVENTS, [labelled(5)])
    assert w.poll() == {watch.INGEST}
    now[0] = watch.FOLLOW_UP_S - 1
    assert w.poll() == set()
    now[0] = watch.FOLLOW_UP_S
    assert w.poll() == {watch.INGEST}
    now[0] = 2 * watch.FOLLOW_UP_S
    assert w.poll() == set()


def test_only_a_scheduling_daemon_watches(cfg, conn, gh, notifier):
    import dataclasses

    assert Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=False).watcher is None
    off = dataclasses.replace(cfg, watch_interval_seconds=0)
    assert Daemon(off, conn, gh=gh, notifier=notifier).watcher is None
    w = Daemon(cfg, conn, gh=gh, notifier=notifier).watcher
    assert w is not None and tuple(w.repos) == cfg.enabled_repos


def test_a_watch_hit_runs_ingest_now_and_the_poll_adds_no_second_review(cfg, conn, gh, notifier):
    gh.open_prs[REPO] = [
        {
            "number": 7,
            "headRefOid": A,
            "isDraft": False,
            "title": "PR 7",
            "updatedAt": "2026-11-27T00:00:00Z",
            "author": {"login": "lklimek"},
            "labels": {"nodes": [{"name": REVIEW_NOW_LABEL}]},
        }
    ]
    d = Daemon(cfg, conn, gh=gh, notifier=notifier, spawn=False)
    api = Api()
    d.watcher = watcher(api)
    d.watch_round()
    d.tick()  # the first ingest pass
    api.set(EVENTS, [labelled(1)])
    assert d.watch_round() == {watch.INGEST} and d.wake.is_set()
    results = d.tick()  # the hit runs before the tick's first task, not 180 s later
    assert "ingest" in results
    d.tick(force=True)  # the regular pass sees the same label
    heads = conn.execute("SELECT trigger FROM heads WHERE number=7").fetchall()
    assert [h["trigger"] for h in heads] == ["normal_request"]
    n = conn.execute("SELECT COUNT(*) FROM events WHERE kind='head.review_now'").fetchone()[0]
    assert n == 1
    assert (
        conn.execute("SELECT detail FROM events WHERE kind='watch.hit'").fetchone()[0]
        == "ingest via watch"
    )


def test_a_hit_during_a_tick_runs_before_the_next_task(cfg, conn, gh, notifier):
    """With the daemon's real task order: the hit's pass, then the scheduler, then the tick
    goes on, its own supersede, reap and schedule passes still in their order."""
    d = Daemon(cfg, conn, gh=gh, notifier=notifier)
    order = []
    names = [t.name for t in d.tasks]
    assert names.index("schedule") < names.index("queue_comments")  # what the hit reorders

    def run(name):
        def fn():
            order.append(name)
            if name == "supersede":  # what the watch thread does while this task runs
                with d._due_lock:
                    d._due.add(watch.QUEUE)

        return fn

    for task in d.tasks:
        task.fn = run(task.name)
        task.last = 1e12 if task.interval_s else 0.0  # every interval task ran just now
    d.tick()
    assert order[:3] == ["supersede", "queue_comments", "schedule"]
    rest = order[3:]
    assert rest.index("reap") < rest.index("schedule")  # after the hit, as every tick
    assert "queue_comments" not in rest  # its interval runs from the hit


def test_an_ingest_hit_runs_ingest_then_the_scheduler(cfg, conn, gh, notifier):
    d = Daemon(cfg, conn, gh=gh, notifier=notifier)
    order = []
    for task in d.tasks:
        task.fn = (lambda n: lambda: order.append(n))(task.name)
        task.last = 1e12 if task.interval_s else 0.0
    d._due.add(watch.INGEST)
    d.tick()
    assert order[:2] == ["ingest", "schedule"]


def test_the_queue_pass_gives_way_to_a_hit_and_runs_again(cfg, conn, gh, notifier, monkeypatch):
    """The gate-comment pass reads one comment per PR and can take minutes (134 s on Brody,
    2026-11-27): a hit arriving meanwhile ends it before its next read, the hit runs, and the
    pass is due again at once."""
    from reviewsys import daemon as daemon_mod

    d = Daemon(cfg, conn, gh=gh, notifier=notifier)
    order = []
    reads = iter(range(100))
    raised = []

    def queue_pass(conn, cfg, gh, *, should_yield):
        order.append("queue_comments")
        for _ in reads:
            if should_yield():
                return {"yielded": 1}
            if not raised:  # the watch thread, mid-pass
                raised.append(1)
                with d._due_lock:
                    d._due.add(watch.INGEST)
        return {"yielded": 0}

    monkeypatch.setattr(daemon_mod, "update_queue_comments", queue_pass)
    for task in d.tasks:
        if task.name not in ("queue_comments",):
            task.fn = (lambda n: lambda: order.append(n))(task.name)
        task.last = 1e12 if task.interval_s else 0.0
    [q] = [t for t in d.tasks if t.name == "queue_comments"]
    q.last = 0.0
    d.tick()
    i = order.index("queue_comments")
    assert order[i : i + 3] == ["queue_comments", "ingest", "schedule"]
    assert q.last == 0.0  # due again
    order.clear()
    d.tick()
    assert "queue_comments" in order
    assert q.last > 0
