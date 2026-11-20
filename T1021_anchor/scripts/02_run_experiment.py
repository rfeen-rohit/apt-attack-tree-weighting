"""What does anchoring login paths at a stage-2 origin actually cost, and buy?

Hopper falls back on anomaly scoring under a daily alert budget when it cannot tell
which earlier login caused a later one. Nine of the eighteen attacks it missed were
stealthy paths of that kind. The design sends any path that starts at the host
stage 2 compromised to Hopper's rules instead of the budgeted score.

Three settings are compared on the same log:

  none            the budget alone (the baseline)
  true_origin     the anchor is the host the attack really started from.
                  This is a CEILING, not a result: the evaluator is handing the
                  detector the answer. It says what the rule could buy at best.
  wrong_origin    the anchor is a server that was never compromised.
                  This is the honest cost measurement: how many extra alerts a day
                  does anchoring produce when the anchor is merely suspected?

Anchors are typed records from common/anchors.py, not bare hostnames, and their
VALIDITY WINDOW is enforced. The earlier version applied the anchor to the whole
log, including logons that happened before the foothold existed, which charged the
rule for alerts it could never have raised and inflated the measured cost.

The number that does not depend on any attack label at all - how often a
server-class host initiates a logon - is reported first, because that is the
assumption the design rests on.
"""
import argparse, json, random, signal, statistics, sys, time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, RESULTS, ROOT, JsonlWriter, log, read_jsonl, save_json,
                    write_jsonl)
import progress
# The shared anchor module sits beside the kits. APPENDED, not inserted, so that this
# kit's own scripts/common.py keeps shadowing the Implementation/common package.
sys.path.append(str(ROOT.parent / "common"))
import anchors
import bridge

SECONDS_PER_DAY = 86400

# The length of the loaded window in days, set once in main(). The alert budget is
# enforced per day BUCKET (the ranking resets each bucket), but a RATE has to be
# divided by the window's real length. A 1.82-day window holds 2 buckets, so dividing
# by buckets understated every per-day figure by 10% there - and by 6.7x on the
# earlier sub-day window, where one bucket made "9.00/day" true by construction.
_WINDOW_DAYS = None


def _set_window_days(d):
    global _WINDOW_DAYS
    _WINDOW_DAYS = float(d) if (d and d > 0) else None
LINK = "A2"          # stage 2 (foothold, T1190) hands this to stage 3 (T1021)
LINK_OUT = "A3"      # and stage 3 hands this to stage 4 (impact, T1486)


# The per-logon writer, so an interruption flushes and names the partial file instead
# of losing whatever is still in the 1 MB buffer.
_ROWS_FILE = {"w": None}


def _check_stop(pr):
    """Called at every progress boundary. On SIGINT or SIGTERM the partial numbers are
    written out, what was saved is said plainly, and the process exits - rather than
    dying with a traceback and leaving a half-written results file behind."""
    sig = progress.stop_requested()
    if sig is None:
        return
    name = progress.stop_name()
    log(f"\n{name}: stopping cleanly.")
    if _ROWS_FILE["w"] is not None:
        w = _ROWS_FILE["w"]
        _ROWS_FILE["w"] = None
        w.close()
        log(f"  SAVED the per-logon rows computed so far: {w.path} ({w.rows:,} rows)")
    if pr is not None:
        pr.finish("interrupted", f"caught {name}")
        log(f"  SAVED the numbers computed so far:")
        log(f"    {pr.json_path}")
        log(f"    {pr.md_path}")
        log(f"  {pr.banner()}")
    log("  NOTE this script has no resume. The experiment is a set of whole-log passes,")
    log("  not a stream, so there is no partial pass to restart from - the PROGRESS")
    log("  files above are the salvage, and a re-run starts at pass 1. To finish sooner,")
    log("  rebuild a smaller log (01_make_auth_data.py --max-events N) or cut the number")
    log("  of passes with --seeds 1 --n-anchors 1.")
    sys.exit(130 if sig == signal.SIGINT else 143)


def _zeros(typecode, n):
    """A zero-filled typed array of n items, built from bytes rather than a list.

    array("i", [0] * n) materialises an n-element Python list first, which for six
    million rows is 240 MB of pointers to build a 24 MB array.
    """
    a = array(typecode)
    a.frombytes(bytes(a.itemsize * int(n)))
    return a


if hasattr(int, "bit_count"):
    def _popcount(x):
        return x.bit_count()
else:                                                    # Python < 3.10
    def _popcount(x):
        return bin(x).count("1")


class LogonLog:
    """The authentication log as typed parallel arrays over interned identifiers.

    WHY THIS EXISTS. The kit used to hold one dict per logon, and then a SECOND dict
    per logon for the scored copy, and then a third for each collected setting. Measured
    on a LANL slice: 826 MB of resident memory for 255,952 logons, 3,384 bytes per
    logon. The LANL red-team labels start at t=150885 - about 15 million input rows, or
    6.4 million logons - so reaching a single labelled event that way needs 20-25 GB and
    the experiment simply cannot be run on a laptop. It is not a tuning problem; the
    representation is wrong.

    So: six columns, no per-logon objects.

      time        array('q')   8 bytes
      user        array('i')   4 bytes   an integer id, not a string
      src, dst    array('i')   4 bytes each, from one host namespace
      label       bytearray    1 byte    an id into label_names
      day         array('i')   4 bytes   filled in by score()
      score_half  bytearray    1 byte    twice the score; every score is a multiple
                                         of 0.5, so this is exact and reversible

    26 bytes a logon, plus the interning tables. LANL has ~17,684 hosts and ~12,425
    users in the whole 58 days, so every host and user is a small int and each name is
    stored once rather than once per logon that mentions it.

    What is NOT stored: logon_type and success, which no part of this experiment reads,
    and prior_edge_days, which was carried on every scored row and used nowhere outside
    the scorer. redteam_kind is held in a dict keyed on row index, because there are 749
    labelled rows in the entire corpus.
    """
    __slots__ = ("time", "user", "src", "dst", "label", "day", "score_half",
                 "host_names", "host_ids", "user_names", "user_ids",
                 "label_names", "label_ids", "red_kind", "red_indices", "red_id",
                 "n", "t0", "n_distinct_days", "days_seen",
                 "as_src", "as_dst", "server_flag", "server_names",
                 "logons_from_server_like", "scored")

    RED = "redteam"

    def __init__(self):
        self.time = array("q")
        self.user = array("i")
        self.src = array("i")
        self.dst = array("i")
        self.label = bytearray()
        self.day = array("i")
        self.score_half = bytearray()
        self.host_names, self.host_ids = [], {}
        self.user_names, self.user_ids = [], {}
        self.label_names, self.label_ids = [], {}
        self.red_kind = {}
        self.red_indices = array("i")
        self.red_id = -1
        self.n = 0
        self.t0 = None
        self.n_distinct_days = 0
        self.days_seen = set()
        self.as_src = self.as_dst = None
        self.server_flag = None
        self.server_names = frozenset()
        self.logons_from_server_like = 0
        self.scored = False

    # ------------------------------------------------------------- building
    @staticmethod
    def _id(names, ids, s):
        i = ids.get(s)
        if i is None:
            i = len(names)
            ids[s] = i
            names.append(s)
        return i

    def append(self, t, user, src, dst, label, red_kind=None):
        i = self.n
        self.time.append(int(t))
        self.user.append(self._id(self.user_names, self.user_ids, user))
        self.src.append(self._id(self.host_names, self.host_ids, src))
        self.dst.append(self._id(self.host_names, self.host_ids, dst))
        li = self._id(self.label_names, self.label_ids, label)
        if li > 255:
            raise SystemExit(f"more than 256 distinct labels in the log; the label "
                             f"column is one byte wide ({label!r} is the 257th)")
        self.label.append(li)
        if label == self.RED:
            self.red_id = li
            self.red_indices.append(i)
            if red_kind is not None:
                self.red_kind[i] = red_kind
        self.n = i + 1

    def finalize(self, min_events=5, ratio=3.0):
        """Per-host in/out counts and the server-like class, over integer ids.

        server_like_hosts() used to walk every row building two defaultdicts keyed on
        host NAME, and it was called once per scoring pass. Here the counts live in two
        typed arrays indexed by host id and are computed once.
        """
        nh = len(self.host_names)
        self.as_src = _zeros("q", nh)
        self.as_dst = _zeros("q", nh)
        for h in self.src:
            self.as_src[h] += 1
        for h in self.dst:
            self.as_dst[h] += 1
        flag = bytearray(nh)
        names, initiated = [], 0
        for h in range(nh):
            d, s = self.as_dst[h], self.as_src[h]
            if d >= min_events and d >= ratio * max(s, 1):
                flag[h] = 1
                names.append(self.host_names[h])
                initiated += s
        self.server_flag = flag
        self.server_names = frozenset(names)
        self.logons_from_server_like = initiated
        return self

    # ------------------------------------------------------------- scoring
    def score(self, t0, on_row=None, every=250_000):
        """A stand-in for Hopper's budgeted anomaly score, computed ONCE.

        Each logon is scored on how unusual it is for that user: an edge never used
        before, a destination never reached before, and an origin that normally only
        receives logons. Hopper's real scorer is richer; what matters here is that it
        ranks, so that a daily budget can be applied to the top of the ranking.

        The score does not depend on any anchor, any store, any threshold or any
        setting, so computing it inside run() meant computing the same numbers again
        for every one of the run's fourteen-to-nineteen passes. It is computed here
        once and every pass reads the same two arrays.

        The "days this edge was seen on" set is a BITMASK, one int per distinct edge,
        instead of a Python set per distinct edge: the scorer only ever asks how many
        distinct days other than today the edge appeared on, which is a popcount. For a
        58-day corpus the mask is a single machine word. The edge key is likewise packed
        into one int - user, src and dst are already integers - rather than a 3-tuple of
        strings, so a distinct edge costs about a hundred bytes instead of four hundred.
        """
        self.t0 = int(t0)
        n = self.n
        self.day = _zeros("i", n)
        self.score_half = bytearray(n)
        day, sh = self.day, self.score_half
        time_a, user_a, src_a, dst_a = self.time, self.user, self.src, self.dst
        server = self.server_flag
        hb = max(1, len(self.host_names).bit_length())
        ushift = hb * 2
        edge_mask, dst_mask = {}, {}
        eget, dget = edge_mask.get, dst_mask.get
        pc = _popcount
        days = self.days_seen
        for i in range(n):
            u = user_a[i]
            s = src_a[i]
            dd = dst_a[i]
            d = (time_a[i] - self.t0) // SECONDS_PER_DAY
            day[i] = d
            days.add(d)
            bit = 1 << d
            ek = (u << ushift) | (s << hb) | dd
            dk = (u << hb) | dd
            me = eget(ek, 0)
            md = dget(dk, 0)
            prior_edge_days = pc(me & ~bit)
            prior_dst_days = pc(md & ~bit)
            v = 0
            if prior_edge_days == 0:
                v += 4                       # 2.0
            elif prior_edge_days < 3:
                v += 1                       # 0.5
            if prior_dst_days == 0:
                v += 3                       # 1.5
            if server[s]:
                v += 2                       # 1.0
            sh[i] = v
            edge_mask[ek] = me | bit
            dst_mask[dk] = md | bit
            if on_row is not None and (i + 1) % every == 0:
                on_row(i + 1)
        self.n_distinct_days = len(days)
        self.scored = True
        return self

    # ------------------------------------------------------------- reading
    def key(self, i):
        """The (time, user, src, dst) tuple the red-team labels are matched on."""
        return (self.time[i], self.user_names[self.user[i]],
                self.host_names[self.src[i]], self.host_names[self.dst[i]])

    def red_rows(self):
        """The red-team rows as dicts. There are 749 of them in all of LANL, so the
        convenience is free; every other row stays in the columns."""
        out = []
        for i in self.red_indices:
            out.append({"time": self.time[i],
                        "user": self.user_names[self.user[i]],
                        "src": self.host_names[self.src[i]],
                        "dst": self.host_names[self.dst[i]],
                        "label": self.RED,
                        "day": self.day[i] if self.scored else None,
                        "score": self.score_half[i] * 0.5 if self.scored else None,
                        "index": i,
                        "redteam_kind": self.red_kind.get(i)})
        return out

    def bytes_per_logon(self):
        """What the columns actually cost, for the record in the results file."""
        if not self.n:
            return None
        b = (self.time.buffer_info()[1] * self.time.itemsize
             + self.user.buffer_info()[1] * self.user.itemsize
             + self.src.buffer_info()[1] * self.src.itemsize
             + self.dst.buffer_info()[1] * self.dst.itemsize
             + len(self.label) + len(self.score_half)
             + (self.day.buffer_info()[1] * self.day.itemsize))
        return round(b / self.n, 2)


def load_logons(path, pr=None, every_rows=250_000, heartbeat=30,
                from_time=None, to_time=None):
    """Read auth.jsonl one line at a time into the column store, in bytes of progress.

    read_jsonl() slurps the file with read_text().splitlines(), which holds the whole
    text AND a list of every line AND the parsed rows at the same time - three copies
    of a log that, on the full LANL pass, is over a hundred gigabytes. This keeps ONE
    column store, not even one dict per row: the dict json.loads returns is read into
    the columns and dropped on the next iteration.

    --from-time / --to-time drop rows outside the window here as well as in 01, so a
    window can be narrowed without rebuilding auth.jsonl.
    """
    p = Path(path)
    lg = LogonLog()
    win = {"from_time": from_time, "to_time": to_time,
           "rows_before_window": 0, "rows_after_window": 0}
    if not p.exists():
        return lg, win
    total = p.stat().st_size
    if pr is not None:
        pr.stage = "loading"
        pr.log_file = {"name": p.name, "path": str(p), "bytes": total}
        pr.load = {"rows": 0, "bytes": 0, "pct": 0.0}
        pr.tick(force=True)
    seen, k = 0, 0
    t0 = time.time()
    append = lg.append
    with p.open("rb") as f:
        for raw in f:
            seen += len(raw)
            k += 1
            if raw.strip():
                r = json.loads(raw)
                t = r["time"]
                if from_time is not None and t < from_time:
                    win["rows_before_window"] += 1
                elif to_time is not None and t > to_time:
                    win["rows_after_window"] += 1
                else:
                    append(t, r["user"], r["src"], r["dst"], r["label"],
                           r.get("redteam_kind"))
            if k % every_rows == 0 and pr is not None:
                el = time.time() - t0
                pct = 100.0 * seen / max(1, total)
                pr.load = {"rows": lg.n, "lines": k, "bytes": seen,
                           "pct": round(pct, 3),
                           "rows_per_second": round(k / max(1e-9, el), 1),
                           "est_remaining_human": progress.fmt_dur(
                               el * (100.0 - pct) / pct if pct > 0 else None)}
                wrote = pr.tick()
                if wrote or pr.beat(heartbeat):
                    log(f"  [{progress.fmt_dur(el):>7}] loading {p.name}: "
                        f"{k:,} rows, {seen:,}/{total:,} bytes ({pct:.2f}%), "
                        f"{k/max(1e-9, el):,.0f} rows/s, ETA "
                        f"{pr.load['est_remaining_human']}")
                _check_stop(pr)
    if pr is not None:
        pr.load = {"rows": lg.n, "lines": k, "bytes": seen, "pct": 100.0,
                   "seconds": round(time.time() - t0, 1)}
        pr.stage = "experiment"
        pr.tick(force=True)
    log(f"loaded {lg.n:,} logons from {p.name} ({seen:,} bytes) in "
        f"{progress.fmt_dur(time.time() - t0)}")
    if win["rows_before_window"] or win["rows_after_window"]:
        log(f"  --from-time/--to-time dropped {win['rows_before_window']:,} row(s) "
            f"before the window and {win['rows_after_window']:,} after it")
    log(f"  column store: {lg.bytes_per_logon() or 0:.1f} bytes per logon in the "
        f"loaded columns ({len(lg.host_names):,} distinct hosts, "
        f"{len(lg.user_names):,} distinct users, interned to integer ids)")
    return lg, win


def estimate_phases(a, loaded_mode, has_entry, has_wrong_pool):
    """How many full passes over the log this run will make.

    This is the number that decides whether the run takes an hour or a week, because
    the cost is (rows) x (passes): --seeds and --n-anchors each multiply it, and the
    confidence sweep adds one pass per grid point per store. It is printed up front so
    the choice is made before the days are spent rather than after.

    A pass is now routing, ranking and the rules check. SCORING IS NOT IN IT: it does
    not depend on the setting, so it happens once before the first pass instead of
    being recomputed in all of them.
    """
    n = 1                                              # the baseline
    if loaded_mode:
        n += 3                                         # loaded_all / true / wrong
    if has_entry:
        n += 1                                         # true_origin
    n_wrong = a.seeds if has_wrong_pool else 0
    n += n_wrong                                       # the wrong-origin draws
    k = max(1, a.n_anchors)
    n += (k if has_entry else 0) + (k if has_wrong_pool else 0)      # the cost curve
    grid = len(anchors.confidence_grid())
    n += grid * ((1 if (has_entry or loaded_mode) else 0)
                 + (1 if (n_wrong or loaded_mode) else 0))           # the sweep
    return n


def day_of(t, t0):
    return (t - t0) // SECONDS_PER_DAY


def server_like_hosts(lg, min_events=5, ratio=3.0):
    """Hosts that mostly receive logons and rarely start one.

    Derived from behaviour, not from names, so it works on real data where the
    naming means nothing. Returns host NAMES, as the callers that report them expect;
    the hot path uses lg.server_flag, which is the same classification by id.
    """
    if lg.server_flag is None or (min_events, ratio) != (5, 3.0):
        nh = len(lg.host_names)
        out = set()
        for h in range(nh):
            d, s = lg.as_dst[h], lg.as_src[h]
            if d >= min_events and d >= ratio * max(s, 1):
                out.add(lg.host_names[h])
        return out
    return set(lg.server_names)


def class_of(host, meta, _web=None, _srv=None):
    """SRV / WEB / other, from the generator's own host lists where available."""
    if host in (_web if _web is not None else set(meta.get("web_hosts") or ())):
        return "WEB"
    if host in (_srv if _srv is not None else set(meta.get("server_hosts") or ())):
        return "SRV"
    if host.startswith("WEB"):
        return "WEB"
    if host.startswith("SRV"):
        return "SRV"
    return "other"


def base_rate_by_class(lg, servers, meta, n_days):
    """The assumption check, split by host class.

    The pooled figure hid the thing that mattered: the anchored host is a WEB host,
    and the rule's cost on WEB hosts is not the rule's cost on SRV hosts. Reporting
    them together let one class's base rate stand in for the other's.
    """
    web = set(meta.get("web_hosts") or ())
    srv = set(meta.get("server_hosts") or ())
    as_src, as_dst, ids = lg.as_src, lg.as_dst, lg.host_ids
    out = {}
    for cls in ("WEB", "SRV", "other"):
        members = sorted(h for h in lg.host_names
                         if class_of(h, meta, web, srv) == cls)
        if not members:
            continue
        initiated = sum(as_src[ids[h]] for h in members)
        out[cls] = {
            "hosts": len(members),
            "in_server_like_class": sum(1 for h in members if h in servers),
            "logons_initiated": initiated,
            "logons_received": sum(as_dst[ids[h]] for h in members),
            "share_of_all_logons": round(initiated / lg.n, 5) if lg.n else None,
            "per_day": round(initiated / n_days, 2),
        }
    return out


def destination_becomes_active(lg, idx, window=3600):
    """The design alerts on an anchored path only if the destination then does
    something. Without process events, 'does something' is approximated by the
    destination appearing as the source of a later logon inside the window."""
    limit = lg.time[idx] + window
    want = lg.dst[idx]
    time_a, src_a, n = lg.time, lg.src, lg.n
    for j in range(idx + 1, n):
        if time_a[j] > limit:
            return False
        if src_a[j] == want:
            return True
    return False


def attack_paths(red, gap_seconds):
    """Group red-team events into attack paths.

    Hopper reports recall per attack PATH; this kit reported it per logon EVENT, so
    the two numbers could not be put side by side. A path here is a run of red-team
    events in time order, each starting from a host the path has already reached,
    with no silence longer than `gap_seconds`.
    """
    paths, cur = [], None
    for r in sorted(red, key=lambda x: x["time"]):
        key = (r["time"], r["user"], r["src"], r["dst"])
        if (cur is not None and r["src"] in cur["hosts"]
                and r["time"] - cur["last_time"] <= gap_seconds):
            cur["keys"].add(key)
            cur["hosts"].add(r["dst"])
            cur["last_time"] = r["time"]
            cur["events"] += 1
        else:
            cur = {"keys": {key}, "hosts": {r["src"], r["dst"]}, "events": 1,
                   "entry": r["src"], "first_time": r["time"], "last_time": r["time"]}
            paths.append(cur)
    return paths


def build_store(path, hosts, emitted_at, ttl, confidence, note, evidence_extra=None,
                operating_point=None, prior=None):
    """Emit one A2 host anchor per host, appending to the run's anchors JSONL.

    With `operating_point` and `prior` the confidence is COMPUTED by the shared rule
    in common/anchors.py from stage 2's own measured TPR and FPR, and the inputs are
    recorded in the anchor's evidence so a reader can recompute it. `confidence` is
    then ignored, and is only used in the fallback case where no operating point was
    supplied - which is the asserted number this wiring exists to retire.
    """
    store = anchors.AnchorStore(path)
    for h in hosts:
        ev = {"emitting_stage": "stage 2 foothold (T1190)",
              "emitted_at_basis": "first red-team event in this log",
              "subject_role": "claimed entry host"}
        ev.update(evidence_extra or {})
        store.emit(anchors.Anchor.create(
            LINK, "host", h, emitted_at=emitted_at, ttl_seconds=ttl,
            confidence=None if operating_point is not None else confidence,
            operating_point=operating_point, prior=prior,
            evidence=ev, note=note))
    return store


def run(lg, t0, budget_per_day, store=None, min_confidence=0.0,
        activity_window=3600, path_gap=SECONDS_PER_DAY, setting="none",
        collect_rows=False, route_at=0.5, pr=None, progress_rows=250_000,
        track_partial=False, heartbeat=30, rows_out=None, collect_alerting=False):
    """One setting, evaluated against the scores computed once in lg.score().

    Nothing in here allocates per logon. The routing decision writes into a typed
    array; the daily rankings are typed arrays per day; the rank of each logon is one
    int in an array indexed by row; whether a logon alerted, and by which path, is one
    byte in a bytearray. The per-logon records go straight out to disk through
    `rows_out` as they are formed rather than being accumulated into a list that is
    written at the end.
    """
    n_rows = lg.n
    if not lg.scored:
        raise RuntimeError("lg.score() must be called before run(): the scores are "
                           "computed once for the whole run, not once per pass")
    # The partial tracker shadows the budgeted path for the days that have FINISHED,
    # so the mid-run file carries real alert and recall numbers rather than estimates.
    # It is deliberately separate from the final computation below: the published
    # numbers come out of the same code they always did.
    tracker = progress.DayBudgetTracker(budget_per_day) if track_partial else None

    def _tick(done, what):
        if pr is None:
            return
        pr.phase_rows = done
        if tracker is not None:
            pr.partial = tracker.snapshot(attack_paths, path_gap)
        wrote = pr.tick()
        if wrote or pr.beat(heartbeat):
            d = pr.as_dict()
            log(f"  [{d['elapsed_human']:>7}] pass {pr.phase_done + 1}/"
                f"{pr.phases_total} {setting}: {what} {done:,}/{n_rows:,} rows "
                f"({100.0 * done / max(1, n_rows):.1f}%)  "
                f"ETA {d['est_remaining_human']}"
                + (f"  [partial: {pr.partial['alerts_per_day_so_far']} alerts/day over "
                   f"{pr.partial['days_complete']} finished day(s), red team "
                   f"{pr.partial['redteam_events_caught']}/"
                   f"{pr.partial['redteam_events_seen']} events]"
                   if (pr.partial and tracker is not None) else ""))
        _check_stop(pr)

    time_a, src_a, dst_a = lg.time, lg.src, lg.dst
    user_a, label_a, day_a, score_a = lg.user, lg.label, lg.day, lg.score_half
    host_names, user_names = lg.host_names, lg.user_names
    red_id = lg.red_id

    # Which host ids the store could possibly match. Every anchor in the store names a
    # subject; a logon whose src is not one of them cannot match, so it never reaches
    # store.match() - which walked every anchor in the store, for every logon, in every
    # pass. The set is built from the same filters match() applies, so the outcome is
    # identical: a subject absent from this log is absent from the set.
    subject_ids = set()
    if store is not None:
        for x in store:
            if x.subject_kind == "host" and x.link == LINK:
                h = lg.host_ids.get(x.subject)
                if h is not None:
                    subject_ids.add(h)

    anchored_idx = array("i")
    by_day = {}
    anchor_of, relax_of = {}, {}
    for i in range(n_rows):
        a = None
        if subject_ids and src_a[i] in subject_ids:
            # at=r["time"] is the fix: an anchor applies only inside its validity
            # window, so logons that predate the foothold are no longer charged to the
            # rule.
            a = store.match("host", host_names[src_a[i]], at=time_a[i],
                            min_confidence=min_confidence, link=LINK)
        rel = 0.0
        if a is not None:
            # The policy table names the identifier this link's consumer is ALLOWED to
            # match on. Matching an A2 anchor on anything but a host would be the silent
            # kind-confusion the anchor module exists to remove, so it is an error here
            # rather than a quiet mismatch discovered in the results.
            requires = anchors.policy_for(a)["requires"]
            if requires != "host":
                raise anchors.AnchorError(
                    f"link {a.link} licenses relaxation only against subject_kind "
                    f"{requires!r}, but this kit matches on 'host'")
            # Graded, decayed relaxation instead of a bare confidence floor: an anchor is
            # worth less the longer ago it was emitted. Routing to the rules path is
            # binary by nature - a path either gets the rules or the budget - so the
            # relaxation is what the routing decision is taken on.
            rel = anchors.relaxation(a, at=time_a[i])
            anchor_of[i] = a
            relax_of[i] = rel
        if a is not None and rel >= route_at:
            anchored_idx.append(i)
        else:
            d = day_a[i]
            arr = by_day.get(d)
            if arr is None:
                arr = by_day[d] = array("i")
            arr.append(i)
            if tracker is not None:
                tracker.add_row(d, score_a[i] * 0.5, i, time_a[i],
                                user_names[user_a[i]], host_names[src_a[i]],
                                host_names[dst_a[i]], label_a[i] == red_id)
        if pr is not None and (i + 1) % progress_rows == 0:
            _tick(i + 1, "routing")

    if pr is not None:
        _tick(n_rows, "ranking")

    # One byte per logon says whether it alerted and which path raised it:
    # 0 no alert, 1 "anchored path to an active destination", 2 "top of the daily
    # anomaly ranking". The two paths are disjoint by construction, so this replaces
    # both the list of alert dicts and the set of alerted indices.
    alert_code = bytearray(n_rows)
    for i in anchored_idx:
        if destination_becomes_active(lg, i, activity_window):
            alert_code[i] = 1

    # budgeted path: the top of each day's ranking
    rank = _zeros("i", n_rows)
    for d, idxs in by_day.items():
        k = 0
        for i in sorted(idxs, key=lambda j: -score_a[j]):
            k += 1
            rank[i] = k
            if k <= budget_per_day and score_a[i] > 0:
                alert_code[i] = 2

    n_alerts = n_rows - alert_code.count(0)
    anchored_alerts = alert_code.count(1)
    n_days_buckets = max(1, lg.n_distinct_days)
    n_days = _WINDOW_DAYS if _WINDOW_DAYS else n_days_buckets

    red = lg.red_rows()
    red_keys = {(r["time"], r["user"], r["src"], r["dst"]) for r in red}
    caught = {(r["time"], r["user"], r["src"], r["dst"]) for r in red
              if alert_code[r["index"]]}
    stealth = [r for r in red if r.get("redteam_kind") == "stealthy"]
    stealth_keys = {(r["time"], r["user"], r["src"], r["dst"]) for r in stealth}

    paths = attack_paths(red, path_gap)
    paths_detected = sum(1 for p in paths if p["keys"] & caught)

    # The false-positive rate this kit hands to the next stage: an alert on a logon that
    # was not red team, over every non-red-team logon the detector saw. That denominator
    # is the honest one - it is what the detector had the opportunity to get wrong.
    n_neg_logons = n_rows - len(red)
    n_fp_alerts = n_alerts - sum(1 for r in red if alert_code[r["index"]])

    subjects = (sorted({a.subject for a in store if a.subject_kind == "host"})
                if store is not None else [])
    servers = lg.server_names

    out = {
        "setting": setting,
        "alerts": n_alerts,
        "alerts_per_day": round(n_alerts / n_days, 2),
        "days": round(n_days, 4),
        "days_buckets": n_days_buckets,
        "days_denominator": ("the window's length in days" if _WINDOW_DAYS else
                             "the count of day buckets - no window was recorded"),
        "budget_basis": ("the budget is enforced per day BUCKET; this rate is over the "
                         "window's length, so the two differ when the window is not a "
                         "whole number of buckets"),
        "anchor_hosts": subjects,
        "anchor_hosts_n": len(subjects),
        # Whether the anchored host is in the class the assumption check measures. If
        # it is not, the base rate reported above is not this host's base rate.
        "anchor_hosts_server_like": {h: (h in servers) for h in subjects},
        "anchor_hosts_all_server_like": all(h in servers for h in subjects) if subjects else None,
        "anchor_min_confidence": min_confidence,
        "anchor_route_at": route_at,
        "anchor_matched_logons": len(anchor_of),
        "anchor_relaxation_mean_over_matched": (round(statistics.mean(relax_of.values()), 4)
                                                if relax_of else None),
        "anchor_relaxation_max": (round(max(relax_of.values()), 4) if relax_of else None),
        "anchored_paths": len(anchored_idx),
        "anchored_alerts": anchored_alerts,
        "alerts_not_redteam": n_fp_alerts,
        "non_redteam_logons": n_neg_logons,
        "alert_fpr_over_logons": (round(n_fp_alerts / n_neg_logons, 6)
                                  if n_neg_logons else None),
        "redteam_total_events": len(red),
        "redteam_detected_events": len(caught & red_keys),
        "redteam_recall_over_events": (round(len(caught & red_keys) / len(red), 4)
                                       if red else None),
        "redteam_total_paths": len(paths),
        "redteam_detected_paths": paths_detected,
        "redteam_recall_over_paths": (round(paths_detected / len(paths), 4)
                                      if paths else None),
        "redteam_path_sizes": [p["events"] for p in paths],
        "stealthy_total": len(stealth),
        "stealthy_detected": len(caught & stealth_keys),
        "stealthy_recall": (round(len(caught & stealth_keys) / len(stealth), 4)
                            if stealth else None),
        "server_like_hosts": len(servers),
        # kept under the old name so earlier notes resolve; it is the per-EVENT figure
        "redteam_recall": (round(len(caught & red_keys) / len(red), 4) if red else None),
        "redteam_total": len(red),
        "redteam_detected": len(caught & red_keys),
    }

    alerting = []
    if collect_rows and rows_out is not None:
        routed = set(anchored_idx)
        label_names = lg.label_names
        for i in range(n_rows):
            a = anchor_of.get(i)
            k = rank[i]
            row = {
                "time": time_a[i], "user": user_names[user_a[i]],
                "src": host_names[src_a[i]], "dst": host_names[dst_a[i]],
                "label": label_names[label_a[i]], "score": score_a[i] * 0.5,
                "rank": (k if k else None),
                "alerted": alert_code[i] != 0, "anchored": a is not None,
                "anchor_id": a.anchor_id if a else None,
                "anchor_confidence": a.confidence if a else None,
                # the graded relaxation this anchor was worth AT THIS LOGON'S TIME, and
                # whether it was enough to divert the path to the rules
                "relaxation": round(relax_of.get(i, 0.0), 6),
                "routed_to_rules": i in routed,
                "setting": setting,
            }
            rows_out.write(row)
            # Only the ALERTING rows are kept in memory, because link A3 emits one
            # anchor per alerting path and nothing downstream reads the rest.
            if collect_alerting and row["alerted"]:
                alerting.append(row)
    return out, alerting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=9,
                    help="alerts a day the budgeted path may raise (Hopper reports under nine)")
    ap.add_argument("--activity-window", type=int, default=3600)
    ap.add_argument("--seeds", type=int, default=5, help="repeats, for the wrong-anchor draw")
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--n-anchors", type=int, default=1,
                    help="hosts to anchor. Sect. 7's claim is about a host CLASS, so the "
                         "cost is reported as a curve over this, not a single number.")
    ap.add_argument("--anchor-ttl", type=int, default=SECONDS_PER_DAY,
                    help="seconds an anchor stays valid after the foothold (its window)")
    ap.add_argument("--anchor-confidence", type=float, default=None,
                    help="EXPLICIT OVERRIDE: assert the emitting detector's belief in "
                         "[0,1] rather than deriving it. Defaults to the kit's historical "
                         "0.8 when no operating point is given. An asserted confidence is "
                         "not a measured one and is logged as such.")
    ap.add_argument("--emitter-operating-point",
                    help="path to the operating point the EMITTING stage measured for "
                         "itself (link A2, so stage 2 / T1190), e.g. "
                         "../T1190_aggregation/results/operating_point_linear.json. Given "
                         "this, the anchor confidence is COMPUTED from the emitter's own "
                         "TPR and FPR and --prior by the shared rule in common/anchors.py.")
    ap.add_argument("--prior", type=float, default=0.01,
                    help="base rate of compromise among subjects of this kind. Turns an "
                         "operating point into a confidence, and is also used to report "
                         "what THIS kit's operating point is worth to the next stage.")
    ap.add_argument("--route-at", type=float, default=0.5,
                    help="relaxation at which a path is routed to the rules instead of the "
                         "budget. Routing is binary by nature, so the graded relaxation "
                         "(policy max_drop x decayed confidence) is what the decision is "
                         "taken on, rather than the raw confidence against a floor.")
    ap.add_argument("--path-gap", type=int, default=SECONDS_PER_DAY,
                    help="silence after which a red-team path is treated as a new path")
    ap.add_argument("--allow-cold-window", action="store_true",
                    help="run even when the anomaly score is degenerate on this window "
                         "(fewer than three distinct values, or fewer than three day "
                         "buckets). The recall it reports is not a measurement; the flag "
                         "exists so that fact is a choice on the command line rather than "
                         "a zero in a results file that looks like a finding.")
    ap.add_argument("--tag", default="default",
                    help="names the output files, so a second run with different flags "
                         "does not overwrite the first")
    ap.add_argument("--anchor-file",
                    help="link A2: a T1190 anchors JSONL. Given this, the true-origin and "
                         "wrong-origin settings are replaced by the hosts T1190's detector "
                         "ACTUALLY FLAGGED, mapped to hosts by common/bridge.json, instead "
                         "of by build_store() inventing an anchor from this kit's own "
                         "entry_host label. This is the link.")
    ap.add_argument("--synthesise-anchors", action="store_true",
                    help="THE OLD FABRICATED BEHAVIOUR, kept reachable and logged as such: "
                         "build the A2 anchors here from meta['entry_host'], i.e. from this "
                         "kit's own ground truth, rather than reading what stage 2 "
                         "detected. Needed to reproduce every result published before the "
                         "link existed. Without --anchor-file this is the default, because "
                         "there is nothing else to do.")
    ap.add_argument("--anchors-out", default=None,
                    help="where the A3 anchors this kit EMITS go "
                         "(default results/anchors_a3_<tag>.jsonl)")
    ap.add_argument("--no-anchors-out", action="store_true",
                    help="emit no A3 anchor file")
    ap.add_argument("--progress-seconds", type=int, default=180,
                    help="seconds between writes of results/PROGRESS_<tag>.json and "
                         ".md - the mid-run snapshot that is readable on its own and "
                         "can be sent to someone for a sanity check")
    ap.add_argument("--progress-rows", type=int, default=250_000,
                    help="rows between progress checks inside a pass")
    ap.add_argument("--heartbeat", type=int, default=30,
                    help="seconds between progress lines on stdout")
    ap.add_argument("--logons-file", choices=("full", "none"), default="full",
                    help="'full' (default, unchanged) writes one row per logon per "
                         "collected setting to results/logons_<tag>.jsonl - about 34 GB "
                         "on the 8-day LANL window. 'none' counts the rows but writes "
                         "nothing; every number in experiment_<tag>.json is unaffected.")
    ap.add_argument("--no-progress-file", action="store_true",
                    help="print progress but write no PROGRESS_<tag> files")
    bridge.add_arguments(ap, link="A3")
    ap.add_argument("--from-time", type=int, default=None,
                    help="evaluate only logons at or after this time, in the unix "
                         "seconds LANL numbers time in. 01 can already cut the window "
                         "while building auth.jsonl; this narrows it further without "
                         "rebuilding the file. Every number the run reports is then a "
                         "number for the window, and the window is written into "
                         "results/experiment_<tag>.json and the PROGRESS files so it "
                         "cannot be quoted without it.")
    ap.add_argument("--to-time", type=int, default=None,
                    help="evaluate only logons at or before this time. See --from-time.")
    a = ap.parse_args()

    # ---- progress, partial results, and a clean exit on SIGINT/SIGTERM.
    progress.install_signal_handlers()
    pr = progress.ProgressReport(a.tag, RESULTS, every_seconds=a.progress_seconds,
                                 enabled=not a.no_progress_file)
    pr.config = {k: vars(a).get(k) for k in (
        "budget", "activity_window", "seeds", "seed_base", "n_anchors", "anchor_ttl",
        "route_at", "path_gap", "prior", "tag", "anchor_file",
        "emitter_operating_point", "anchor_confidence")}
    pr.set_stage("starting")
    _SUMMARY_KEYS = ("alerts", "alerts_per_day", "days", "anchored_paths",
                     "anchor_matched_logons", "alerts_not_redteam",
                     "non_redteam_logons", "alert_fpr_over_logons",
                     "redteam_total_events", "redteam_detected_events",
                     "redteam_recall_over_events", "redteam_total_paths",
                     "redteam_detected_paths", "redteam_recall_over_paths",
                     "stealthy_total", "stealthy_detected", "server_like_hosts")

    # The per-logon file is opened once and written as each pass produces its rows.
    # It used to be a list of one dict per logon per collected setting, held until the
    # very end and then serialised - three to five whole copies of the log in RAM for
    # the sake of a file that is append-only by nature.
    rows_file = {"w": None, "settings": []}

    def do_run(*args, **kw):
        """One full pass over the log, announced and accounted for.

        Every pass is counted so the progress line can say "pass 7 of 19" and the ETA
        has something to extrapolate from. The baseline pass also carries the partial
        tracker, because the baseline is the number worth watching: it is the one the
        operating point handed to stage 4 is measured on.
        """
        name = kw.get("setting", "run")
        pr.start_phase(name, lg.n)
        log(f"\npass {pr.phase_done + 1}/{pr.phases_total}: {name} "
            f"({lg.n:,} logons)")
        kw.setdefault("pr", pr)
        kw.setdefault("progress_rows", a.progress_rows)
        kw.setdefault("heartbeat", a.heartbeat)
        kw.setdefault("track_partial", name == "none")
        if kw.get("collect_rows"):
            kw.setdefault("rows_out", rows_file["w"])
            kw.setdefault("collect_alerting", name == "none")
            if lg.n:
                rows_file["settings"].append(name)
        r, rr = run(*args, **kw)
        pr.finish_phase(name, {k: r.get(k) for k in _SUMMARY_KEYS})
        _check_stop(pr)
        return r, rr

    br = bridge.load(a.bridge, no_bridge=a.no_bridge)
    in_store, consumed = None, {"link": LINK, "anchor_file": a.anchor_file,
                                "mode": None, "hosts": [], "n_anchors": 0,
                                "n_malformed": 0, "note": ""}
    if a.anchor_file and a.synthesise_anchors:
        raise SystemExit(
            "--anchor-file and --synthesise-anchors are mutually exclusive. One reads the "
            "anchors stage 2 actually emitted; the other fabricates them from this kit's "
            "own entry_host label. Running both would make it impossible to say which "
            "produced the numbers.")
    if a.anchor_file:
        in_store = anchors.AnchorStore.load(a.anchor_file)
        consumed.update(mode="loaded", n_anchors=len(in_store),
                        n_malformed=in_store.malformed_lines,
                        hosts=sorted({x.subject for x in in_store
                                      if x.link == LINK and x.subject_kind == "host"}))
        log(f"link A2: loaded {len(in_store)} anchor(s) from {a.anchor_file} "
            f"({in_store.malformed_lines} malformed line(s)); hosts named: "
            f"{consumed['hosts']}")
        log(f"  bridge: {br.status}")
        if not consumed["hosts"]:
            log("  WARNING the file holds no A2 host anchor, so every anchored setting "
                "will behave exactly like the baseline")
        if not br.enabled:
            # Every subject in an incoming anchor file is bridge-derived: stage 2 has no
            # hosts of its own, so the host named here exists only because a binding said
            # a provenance group stands for it. Switching the binding off downstream has
            # to discard the file, or the ablation would silently keep acting on the
            # assumption it is meant to remove.
            consumed.update(mode="ignored_no_bridge", note=(
                "--no-bridge: every host named in this file exists only because of a "
                "binding, so the file is discarded and this run is unanchored. That is "
                "the ablation."))
            log("  --no-bridge: the host subjects in this file are bridge-derived, so the "
                "file is DISCARDED and this run is unanchored. This is the ablation.")
            in_store = None
    else:
        consumed.update(mode="synthesised", note=(
            "no --anchor-file: the A2 anchors below were FABRICATED in this kit from "
            "meta['entry_host'], which is this kit's own ground truth. That is the defect "
            "the link exists to remove; it is the only behaviour available when stage 2 "
            "has not run."))
        log("WARNING: no --anchor-file. The A2 anchors are FABRICATED here from "
            "meta['entry_host'] - this kit's own ground truth - which is the old "
            "behaviour. Pass --anchor-file "
            "../T1190_aggregation/results/anchors_<tag>.jsonl to consume what stage 2 "
            "actually detected.")

    # ---- where this run's anchor confidence comes from.
    # The point of the contract in common/anchors.py is that stage N+1's confidence is
    # the posterior implied by stage N's MEASURED operating point, not a number the
    # author picked. The flat 0.8 is still reachable, but it is now the exception and
    # the log says so.
    emitter_op, anchor_prior = None, a.prior
    if a.emitter_operating_point:
        emitter_op = anchors.OperatingPoint.load(a.emitter_operating_point)
    if emitter_op is not None and a.anchor_confidence is not None:
        log("WARNING: --anchor-confidence was given together with "
            "--emitter-operating-point. The ASSERTED number wins and the measurement is "
            "discarded, so nobody can recompute this run's confidence from the emitter's "
            "performance. This is the defect the operating-point hand-off exists to "
            "remove; drop --anchor-confidence to derive it instead.")
        emitter_op = None
    if emitter_op is not None:
        anchor_confidence = anchors.confidence_from_operating_point(emitter_op, anchor_prior)
        confidence_source = (f"derived from {a.emitter_operating_point} at prior "
                             f"{anchor_prior}")
        log(f"anchor confidence DERIVED from the emitter's operating point: tpr "
            f"{emitter_op.tpr}, fpr {emitter_op.fpr} (effective "
            f"{emitter_op.effective_fpr():.6g}; n_pos {emitter_op.n_pos}, n_neg "
            f"{emitter_op.n_neg}) -> LR+ {emitter_op.lr_plus():.6g}, prior {anchor_prior} "
            f"-> confidence {anchor_confidence:.4f}"
            + (f"   [{emitter_op.detector}]" if emitter_op.detector else ""))
    elif in_store is not None:
        # In loaded mode nothing here sets a confidence: every anchor carries the one
        # stage 2 derived from its own measured operating point, and build_store() is
        # never called. Saying "ASSERTED 0.8" would be false.
        anchor_confidence = None
        confidence_source = (f"carried by each anchor in {a.anchor_file}, derived by "
                             f"stage 2 from its own measured operating point; nothing is "
                             f"asserted here and build_store() is not called")
        got = sorted({round(x.confidence, 6) for x in in_store})
        log(f"anchor confidence comes from the consumed file, not from this kit: "
            f"{got} (derived by stage 2 from its own tpr/fpr)")
    else:
        anchor_confidence = 0.8 if a.anchor_confidence is None else a.anchor_confidence
        confidence_source = ("asserted on the command line"
                             if a.anchor_confidence is not None else
                             "asserted: the kit's historical default of 0.8")
        log(f"WARNING: anchor confidence {anchor_confidence} is ASSERTED, not measured "
            f"({confidence_source}). It is not comparable with any other stage's number. "
            f"Pass --emitter-operating-point "
            f"../T1190_aggregation/results/operating_point_<tag>.json to derive it from "
            f"stage 2's own measured TPR and FPR.")
    # written back so config, the sweep note and the bundle all report the value used
    a.anchor_confidence = anchor_confidence

    lg, loaded_window = load_logons(DATA / "auth.jsonl", pr,
                                    every_rows=a.progress_rows,
                                    heartbeat=a.heartbeat,
                                    from_time=a.from_time, to_time=a.to_time)
    if not lg.n:
        raise SystemExit("data/auth.jsonl is missing - run 01_make_auth_data.py first")
    lg.finalize()
    meta_path = DATA / "auth_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} is missing - run 01_make_auth_data.py first "
                         "(auth.jsonl alone is not enough: the entry host, the host "
                         "classes and the red-team timing all come from the metadata)")
    meta = json.loads(meta_path.read_text())
    t0 = min(lg.time)
    _span_days = (max(lg.time) - t0) / SECONDS_PER_DAY
    _set_window_days(_span_days)
    n_days = _span_days if _span_days > 0 else 1.0

    # ---- the window every number below belongs to. 01 records the window it built
    # auth.jsonl with; --from-time/--to-time here can narrow it further. Either way it
    # is carried into the results and the PROGRESS files rather than being remembered
    # separately, because "9.4 alerts a day" means nothing without the days it is over.
    _w_from = [x for x in (meta.get("window_from_time"), a.from_time) if x is not None]
    _w_to = [x for x in (meta.get("window_to_time"), a.to_time) if x is not None]
    window = {
        "applied": bool(_w_from or _w_to),
        "from_time": (max(_w_from) if _w_from else None),
        "to_time": (min(_w_to) if _w_to else None),
        "source": ", ".join(
            [x for x in (("01 --from-time/--to-time, recorded in auth_meta.json"
                          if meta.get("window_applied") else None),
                         ("02 --from-time/--to-time"
                          if (a.from_time is not None or a.to_time is not None)
                          else None)) if x] ) or "none: the whole of auth.jsonl",
        "rows_kept": lg.n,
        "rows_dropped_here_before_window": loaded_window["rows_before_window"],
        "rows_dropped_here_after_window": loaded_window["rows_after_window"],
        "first_time": int(t0), "last_time": int(max(lg.time)),
        "days_covered": int(n_days),
        "redteam_rows": len(lg.red_indices),
        "log_file": "auth.jsonl",
        "auth_meta_window": {k: meta.get(k) for k in (
            "window_applied", "window_from_time", "window_to_time", "window_days",
            "window_rows_before", "window_stopped_at_to_time")},
        "note": ("Every figure in this file is a figure FOR THIS WINDOW. The LANL "
                 "corpus is 58 days and 1.05e9 rows; the red-team labels start at "
                 "t=150885, so a slice taken from day 0 contains none of them and "
                 "every recall number on it is a recall over zero positives. "
                 "scripts/redteam_window.py says which window to ask for."),
    }
    window["days"] = (None if (window["from_time"] is None or window["to_time"] is None)
                      else round((window["to_time"] - window["from_time"])
                                 / SECONDS_PER_DAY, 4))
    pr.window = window
    pr.tick(force=True)
    if window["applied"]:
        log(f"\nWINDOW: t={window['from_time']}..{window['to_time']} "
            f"({window['days']} day(s)) via {window['source']}")
    log(f"  the log loaded spans t={window['first_time']}..{window['last_time']} "
        f"({n_days} day(s)), {lg.n:,} logons, {window['redteam_rows']} of them "
        f"red-team labelled")
    if meta.get("source") == "lanl" and not window["redteam_rows"]:
        log("  WARNING this slice contains NO red-team labelled logon, so every recall "
            "figure below is a recall over zero positives. Run "
            "scripts/redteam_window.py and rebuild with its --from-time/--to-time.")

    # ---- the assumption, measured without any attack label
    servers = server_like_hosts(lg)
    initiated = lg.logons_from_server_like
    assumption = {
        "server_like_hosts": len(servers),
        "logons_initiated_by_them": initiated,
        "share_of_all_logons": round(initiated / lg.n, 5),
        "per_day": round(initiated / n_days, 2),
        "by_class": base_rate_by_class(lg, servers, meta, n_days),
        "note": ("This is the volume the anchor rule diverts from the budget to the rules. "
                 "It does not depend on any attack label. Read the per-class split: the "
                 "anchored host is a WEB host, and the pooled figure is dominated by SRV."),
    }
    pr.assumption = dict(assumption, total_logons=lg.n, days=n_days,
                         hosts_total=sum(c["hosts"] for c in
                                         assumption["by_class"].values()))
    pr.tick(force=True)
    log(f"assumption check: {len(servers)} server-like hosts initiate {initiated} logons "
        f"({assumption['share_of_all_logons']*100:.2f}% of all, {assumption['per_day']}/day)")
    for cls, c in assumption["by_class"].items():
        log(f"  {cls:5s} {c['hosts']:4d} hosts, {c['in_server_like_class']:4d} of them "
            f"server-like; they initiate {c['logons_initiated']} logons "
            f"({c['per_day']}/day) and receive {c['logons_received']}")

    # ---- anchors. The emission time is the foothold moment, so the validity window
    # cannot reach back over logons that happened before the intrusion existed.
    emitted_at = meta.get("redteam_first_time")
    if emitted_at is None:
        emitted_at = t0
        emit_basis = "no red-team label; anchor emitted at the start of the log"
    else:
        emit_basis = "the first red-team event in the log"
    anchors_path = RESULTS / f"anchors_{a.tag}.jsonl"
    anchors_path.parent.mkdir(parents=True, exist_ok=True)
    anchors_path.write_text("")          # one emission log per tagged run

    entry = meta.get("entry_host")
    web_hosts = [h for h in (meta.get("web_hosts") or []) if h != entry]
    srv_sorted = sorted(servers - {entry}) if entry else sorted(servers)
    # The pool extends within the host class first, then into the server-like class, so
    # the curve is "anchoring more of the same kind of host", not an arbitrary mixture.
    true_pool = ([entry] if entry else []) + web_hosts + [h for h in srv_sorted
                                                          if h not in web_hosts]
    # The wrong-origin curve anchors a random SAMPLE of never-compromised server-like
    # hosts rather than the alphabetical prefix, so each step adds a representative host
    # and the k=1 set nests inside k=2. An alphabetical prefix measures whichever hosts
    # happen to sort first, which is not a property of the class.
    wrong_pool = list(srv_sorted)
    random.Random(a.seed_base + 7).shuffle(wrong_pool)

    # ---------------------------------------------------------- link A2, consumed
    # With --anchor-file the host pools above are not used at all. The anchored
    # settings are the hosts stage 2's detector ACTUALLY FLAGGED, carried here through
    # the declared binding, so build_store() - which invents an anchor from this kit's
    # own entry_host label - is not called.
    #
    # They are still SPLIT into a true-origin and a wrong-origin group for reporting,
    # using this kit's red-team label. That is evaluation, not input: the detector sees
    # the same anchors either way and the split only decides which column the numbers
    # are printed in.
    loaded_mode = in_store is not None
    red_hosts = set()
    for _i in lg.red_indices:
        red_hosts.add(lg.host_names[lg.src[_i]])
        red_hosts.add(lg.host_names[lg.dst[_i]])
    if loaded_mode:
        incoming = [x for x in in_store if x.link == LINK and x.subject_kind == "host"]
        hosts_seen = set(lg.host_names)
        unknown = sorted({x.subject for x in incoming if x.subject not in hosts_seen})
        consumed.update(
            n_a2_host_anchors=len(incoming),
            hosts_in_this_log=sorted({x.subject for x in incoming} - set(unknown)),
            hosts_not_in_this_log=unknown,
            hosts_that_were_really_compromised=sorted({x.subject for x in incoming}
                                                      & red_hosts),
            hosts_never_compromised=sorted({x.subject for x in incoming} - red_hosts),
            split_note=("the true/wrong split below uses this kit's red-team label for "
                        "REPORTING only; the detector is handed the same anchors either way"),
            confidences=sorted({round(x.confidence, 6) for x in incoming}),
        )
        if unknown:
            log(f"  NOTE {len(unknown)} anchored host(s) do not appear in this log at "
                f"all and can never match: {unknown}. That is the binding being wrong "
                f"about which host a corpus group stands for, and it is visible rather "
                f"than silent.")
        log(f"  of the hosts named, {consumed['hosts_that_were_really_compromised']} "
            f"really were compromised and {consumed['hosts_never_compromised']} never were")
        # the synthetic pools are switched off, so build_store() cannot run below
        entry, true_pool, srv_sorted, wrong_pool = None, [], [], []

    pr.phases_total = estimate_phases(a, loaded_mode, bool(entry), bool(wrong_pool))
    log(f"\nthis run makes {pr.phases_total} full pass(es) over {lg.n:,} logons "
        f"({pr.phases_total * lg.n:,} row-visits in total). --seeds "
        f"{a.seeds} and --n-anchors {a.n_anchors} multiply that; the confidence sweep "
        f"adds {len(anchors.confidence_grid())} pass(es) per store.")
    log(f"  the anomaly score does not depend on the setting, so it is computed ONCE "
        f"below and every pass reads the same arrays; a pass is routing, ranking and "
        f"the rules check.")
    log(f"progress: results/PROGRESS_{a.tag}.json and .md every {a.progress_seconds}s; "
        f"scripts/status.py reads them live; Ctrl-C or SIGTERM saves and exits cleanly.")
    pr.tick(force=True)

    # ---- score once, for every pass.
    pr.set_stage("scoring")
    _t_score = time.time()
    log(f"\nscoring {lg.n:,} logons once (this used to happen inside every one of the "
        f"{pr.phases_total} passes)")

    def _score_tick(done):
        pr.phase_rows = done
        pr.phase_rows_total = lg.n
        wrote = pr.tick()
        if wrote or pr.beat(a.heartbeat):
            el = time.time() - _t_score
            log(f"  [{progress.fmt_dur(el):>7}] scoring {done:,}/{lg.n:,} rows "
                f"({100.0 * done / max(1, lg.n):.1f}%), "
                f"{done / max(1e-9, el):,.0f} rows/s")
        _check_stop(pr)

    lg.score(t0, on_row=_score_tick, every=a.progress_rows)
    log(f"  scored in {progress.fmt_dur(time.time() - _t_score)}; "
        f"{lg.n_distinct_days} distinct day(s); columns now cost "
        f"{lg.bytes_per_logon():.1f} bytes per logon")

    # ---- is this window warm enough to rank at all?
    # The score asks whether this user has used this exact (user, src, dst) edge, and
    # reached this destination, on an EARLIER day - and a day is
    # (t - window_start) // 86400, so a short window has one or two buckets, almost no
    # edge recurs across them, every logon scores novel, and the ranking carries no
    # information. Two runs were spent discovering that by hand. Now it is printed and,
    # unless the author insists, refused.
    _b = bytes(lg.score_half)
    _spread = [(v * 0.5, _b.count(v)) for v in sorted(set(_b), reverse=True)]
    log("  score spread: " + ", ".join(f"{s}={c:,}" for s, c in _spread)
        + f"  ({len(_spread)} distinct value(s))")
    if len(_spread) < 3 or lg.n_distinct_days < 3:
        _msg = (
            f"COLD WINDOW: the anomaly score took only {len(_spread)} distinct value(s) "
            f"across {lg.n:,} logons, over {lg.n_distinct_days} day bucket(s).\n"
            f"  The score is novelty over EARLIER days, and a day here is "
            f"(t - window_start) // {SECONDS_PER_DAY}, so a window under three days has "
            f"at most two buckets and almost no (user, src, dst) edge recurs across "
            f"them. Every logon then scores novel, nothing separates them, and a budget "
            f"of {a.budget}/day picks {a.budget} of the tied top by index order. Any "
            f"recall measured here is arithmetic, not a property of the detector.\n"
            f"  Widen the lead:  python3 scripts/redteam_window.py "
            f"<redteam.txt.gz> --lead-hours 168\n"
            f"  then rebuild with 01 --from-time/--to-time and run this again.\n"
            f"  --allow-cold-window runs it anyway; the result is void and must be "
            f"labelled so.")
        if not a.allow_cold_window:
            raise SystemExit("\n" + _msg)
        log("\nWARNING " + _msg)
    pr.set_stage("experiment")
    pr.phase_rows = 0

    results = {}
    true_store = wrong_store = None
    if a.logons_file == "none":
        class _CountOnly:                     # same interface, no bytes on disk
            path = None
            def __init__(self): self.rows = 0
            def write(self, row): self.rows += 1
            def close(self): pass
        rows_file["w"] = _CountOnly()
    else:
        rows_file["w"] = JsonlWriter(RESULTS / f"logons_{a.tag}.jsonl")
    _ROWS_FILE["w"] = rows_file["w"]
    base, alerting = do_run(lg, t0, a.budget, path_gap=a.path_gap,
                          setting="none", collect_rows=True)
    results["none"] = base

    if loaded_mode:
        true_store = anchors.AnchorStore(None, [x for x in incoming if x.subject in red_hosts])
        wrong_store = anchors.AnchorStore(None, [x for x in incoming
                                                 if x.subject not in red_hosts])
        for name, st in (("loaded_all", anchors.AnchorStore(None, incoming)),
                         ("loaded_true_origin", true_store),
                         ("loaded_wrong_origin", wrong_store)):
            if not len(st):
                results[name] = {"setting": name, "skipped":
                                 "no anchor of this kind in the consumed file"}
                continue
            r, _rr = do_run(lg, t0, a.budget, st, activity_window=a.activity_window,
                        path_gap=a.path_gap, setting=name, collect_rows=True,
                        route_at=a.route_at)
            results[name] = r
        # the two names the rest of the script and the bundler expect
        if "loaded_true_origin" in results and "skipped" not in results["loaded_true_origin"]:
            results["true_origin"] = dict(results["loaded_true_origin"],
                                          setting="true_origin (from REAL stage-2 detections)")

    if entry:
        hosts = true_pool[:max(1, a.n_anchors)]
        true_store = build_store(anchors_path, hosts, emitted_at, a.anchor_ttl,
                                 anchor_confidence,
                                 note="true origin: the host the attack really started from",
                                 evidence_extra={"control": "true_origin"},
                                 operating_point=emitter_op, prior=anchor_prior)
        r, _rr = do_run(lg, t0, a.budget, true_store, activity_window=a.activity_window,
                    path_gap=a.path_gap, setting="true_origin", collect_rows=True,
                    route_at=a.route_at)
        results["true_origin"] = r

    wrong = []
    rng = random.Random(a.seed_base)
    candidates = srv_sorted
    for k in range(a.seeds):
        if not candidates:
            break
        hosts = [candidates[rng.randrange(len(candidates))]
                 for _ in range(max(1, a.n_anchors))]
        hosts = sorted(set(hosts))
        store = build_store(anchors_path, hosts, emitted_at, a.anchor_ttl,
                            anchor_confidence,
                            note="wrong origin: never compromised, the honest cost control",
                            evidence_extra={"control": "wrong_origin", "draw": k},
                            operating_point=emitter_op, prior=anchor_prior)
        r, _rr = do_run(lg, t0, a.budget, store, activity_window=a.activity_window,
                    path_gap=a.path_gap, setting=f"wrong_origin_{k}",
                    collect_rows=(k == 0), route_at=a.route_at)
        wrong.append(r)
        if k == 0:
            wrong_store = store
    if wrong:
        def m(key):
            vals = [w[key] for w in wrong if w.get(key) is not None]
            return round(statistics.mean(vals), 4) if vals else None
        results["wrong_origin"] = {
            "runs": wrong,
            "alerts_per_day_mean": round(statistics.mean(w["alerts_per_day"] for w in wrong), 2),
            "alerts_per_day_sd": (round(statistics.stdev(w["alerts_per_day"] for w in wrong), 2)
                                  if len(wrong) > 1 else 0.0),
            "redteam_recall_over_events_mean": m("redteam_recall_over_events"),
            "redteam_recall_over_paths_mean": m("redteam_recall_over_paths"),
            "redteam_recall_mean": m("redteam_recall_over_events"),
        }

    # ---- cost as a function of how many hosts are anchored
    curve = {"true_origin": [], "wrong_origin": []}
    for k in range(1, max(1, a.n_anchors) + 1):
        if entry:
            hosts = true_pool[:k]
            st = build_store(anchors_path, hosts, emitted_at, a.anchor_ttl,
                             anchor_confidence, note=f"curve true_origin n={k}",
                             evidence_extra={"control": "true_origin", "curve_n": k},
                             operating_point=emitter_op, prior=anchor_prior)
            r, _ = do_run(lg, t0, a.budget, st, activity_window=a.activity_window,
                       path_gap=a.path_gap, setting=f"true_origin_n{k}",
                       route_at=a.route_at)
            curve["true_origin"].append({
                "n_anchors": k, "hosts": hosts, "alerts_per_day": r["alerts_per_day"],
                "extra_alerts_per_day": round(r["alerts_per_day"] - base["alerts_per_day"], 2),
                "anchored_paths": r["anchored_paths"],
                "all_server_like": r["anchor_hosts_all_server_like"],
                "redteam_recall_over_events": r["redteam_recall_over_events"],
                "redteam_recall_over_paths": r["redteam_recall_over_paths"],
                "stealthy_detected": r["stealthy_detected"]})
        if wrong_pool:
            hosts = wrong_pool[:k]
            st = build_store(anchors_path, hosts, emitted_at, a.anchor_ttl,
                             anchor_confidence, note=f"curve wrong_origin n={k}",
                             evidence_extra={"control": "wrong_origin", "curve_n": k},
                             operating_point=emitter_op, prior=anchor_prior)
            r, _ = do_run(lg, t0, a.budget, st, activity_window=a.activity_window,
                       path_gap=a.path_gap, setting=f"wrong_origin_n{k}",
                       route_at=a.route_at)
            curve["wrong_origin"].append({
                "n_anchors": k, "hosts": hosts, "alerts_per_day": r["alerts_per_day"],
                "extra_alerts_per_day": round(r["alerts_per_day"] - base["alerts_per_day"], 2),
                "anchored_paths": r["anchored_paths"],
                "all_server_like": r["anchor_hosts_all_server_like"],
                "redteam_recall_over_events": r["redteam_recall_over_events"],
                "redteam_recall_over_paths": r["redteam_recall_over_paths"],
                "stealthy_detected": r["stealthy_detected"]})

    # ---- cost at each confidence threshold a consumer might set
    grid = anchors.confidence_grid()
    sweep = {"grid": grid, "anchor_confidence": anchor_confidence,
             "true_origin": [], "wrong_origin": [],
             "note": ("Anchors here all carry the same confidence, so the curve is a step "
                      "at that value: above it the anchor is ignored and the setting "
                      "collapses to the baseline. The step's position is the point - a "
                      "consumer choosing a threshold above the emitter's confidence buys "
                      "and pays nothing.")}
    # Each control is swept against its OWN anchors. Sweeping the whole emission log
    # would mix the true and wrong origins into one store and measure neither.
    for name, st in (("true_origin", true_store), ("wrong_origin", wrong_store)):
        if st is None:
            continue
        for thr in grid:
            r, _ = do_run(lg, t0, a.budget, st, min_confidence=thr,
                       activity_window=a.activity_window, path_gap=a.path_gap,
                       setting=f"sweep_{name}_{thr}", route_at=a.route_at)
            sweep[name].append({
                "min_confidence": thr, "alerts_per_day": r["alerts_per_day"],
                "extra_alerts_per_day": round(r["alerts_per_day"] - base["alerts_per_day"], 2),
                "anchored_paths": r["anchored_paths"],
                "redteam_recall_over_events": r["redteam_recall_over_events"],
                "redteam_recall_over_paths": r["redteam_recall_over_paths"]})
    # Reloading the emission log proves the anchors survive the round trip to JSONL,
    # which is the whole point of writing them out for a consuming kit to read.
    reloaded = anchors.AnchorStore.load(anchors_path)
    sweep["loaded_anchor_summary"] = reloaded.summary()

    # ---- what THIS kit hands the next stage (link A3, consumed by T1486).
    # The primary operating point is the BASELINE run - the budget alone, with no anchor
    # from stage 2 - because that is this detector's performance on its own merits; an
    # anchored run's performance is partly the emitter's and would double-count it.
    # Recall is the PATH-level figure, which is the unit Hopper reports, and the
    # false-positive rate is alerts that were not red team over every non-red-team logon.
    if base.get("redteam_total_paths") == 1:
        log(f"\nWARNING: this window holds exactly ONE red-team attack path "
            f"({base.get('redteam_total_events')} labelled event(s) chained into it by "
            f"--path-gap {a.path_gap}s), so the path-level tpr has a DENOMINATOR OF 1 "
            f"and can only come out 0.0 or 1.0. Either way it is not an estimate, and "
            f"the operating point handed to stage 4 inherits that: 0.0 is REFUSED as "
            f"anti-evidence, 1.0 gives LR+ = 1/fpr and saturates the confidence cap.\n"
            f"  Sweep --path-gap (3600 splits a path on an hour of silence) and report "
            f"the path count with the recall, or lead with the event-level recall over "
            f"{base.get('redteam_total_events')} positives and say which denominator "
            f"each number uses.")

    op_path = RESULTS / f"operating_point_{a.tag}.json"
    op_block, own_op, implied, op_refused = None, None, None, None
    if base["redteam_recall_over_paths"] is None or not base["non_redteam_logons"]:
        log(f"\nNOT writing {op_path.name}: this log has no red-team path or no benign "
            f"logon, so there is no measured tpr/fpr pair to hand to stage 4. The next "
            f"kit will have to be run without an emitter operating point.")
    else:
        own_op = anchors.OperatingPoint(
            tpr=float(base["redteam_recall_over_paths"]),
            fpr=float(base["alert_fpr_over_logons"]),
            n_pos=int(base["redteam_total_paths"]),
            n_neg=int(base["non_redteam_logons"]),
            threshold=float(a.budget),
            detector=f"T1021 budgeted anomaly @ {a.budget}/day")
        own_op.save(op_path)
        op_block = own_op.as_dict()
        # The operating point can be REFUSED as evidence (LR+ at or below 1, or a
        # measured FPR of zero with no negatives). common/anchors.py raises AnchorError
        # for both; letting it propagate would lose a completed run, so it is caught,
        # the reason is recorded, and link A3 emits nothing.
        try:
            implied = anchors.confidence_from_operating_point(own_op, a.prior)
        except anchors.AnchorError as e:
            op_refused = str(e)
            log(f"\nREFUSED: this kit's own operating point is not evidence, so link A3 "
                f"emits nothing. {op_refused}")
        log(f"\n--- the operating point this kit hands to stage 4 (link A3) ---")
        # lr_plus() RAISES when LR+ is at or below 1, and that case is already handled
        # above - op_refused is set and link A3 emits nothing. Calling it again inside
        # this log line killed the whole run after every pass had finished, which is
        # exactly what happens on a window that contains red-team labels the baseline
        # does not catch: tpr 0 over 1 path. The refusal is reported, not re-raised.
        try:
            lr_text = f"{own_op.lr_plus():.6g}"
        except anchors.AnchorError:
            lr_text = "REFUSED (at or below 1; see operating_point_refused)"
        log(f"  wrote {op_path.name}: tpr {own_op.tpr} over {own_op.n_pos} attack path(s), "
            f"fpr {own_op.fpr} over {own_op.n_neg} benign logon(s) "
            f"(effective fpr {own_op.effective_fpr():.6g}), LR+ {lr_text}")
        if implied is not None:
            log(f"  at prior {a.prior} an A3 anchor built from it would carry confidence "
                f"{implied if implied is not None else float('nan'):.4f}; that is what this "
                f"kit's detection is worth to the next stage")
        else:
            log(f"  no A3 confidence can be derived from it, so link A3 emits nothing: "
                f"{op_refused}")

    # ================================================== link A3, emitted
    # One anchor per path THIS detector alerted on, in the BASELINE setting: the budget
    # alone, no anchor from stage 2. The baseline is used for the same reason the
    # operating point is - an anchored run's alerts are partly stage 2's and emitting
    # from them would count the upstream detection twice.
    #
    # The subject is a SESSION, which link A3's policy requires and which T1486 keys on.
    # T1021 has no sessions; it has a host and an account. The binding in
    # common/bridge.json identifies one with the other, and the candidate order is
    # declared here: the DESTINATION host with the account first, because that is the
    # host the attacker reached and so the one whose subsequent impact stage 4 would
    # see; then the source with the account; then the bare hosts.
    a3_path = (Path(a.anchors_out) if a.anchors_out
               else RESULTS / f"anchors_a3_{a.tag}.jsonl")
    a3 = {"link": LINK_OUT, "emitted": 0, "alerting_logons": 0, "unbound": 0,
          "capped": 0, "refused": op_refused, "anchors_file": None, "by_subject": {},
          "unbound_subjects": [], "note": "",
          "candidate_order": ["session dst\\user", "session src\\user", "host dst",
                              "host src"]}
    # `alerting` is the baseline pass's alerting rows, the only per-logon records kept
    # in memory; every row (alerting or not) already went to logons_<tag>.jsonl.
    a3["alerting_logons"] = len(alerting)
    if a.no_anchors_out:
        a3["note"] = "--no-anchors-out: nothing emitted"
    elif own_op is None or op_refused:
        a3["note"] = ("no usable operating point, so there is no measured confidence to "
                      "put on an A3 anchor: " + (op_refused or "none was written"))
    elif not br.enabled:
        a3["note"] = ("--no-bridge: no host or account is bound to a session, so no "
                      "alerting path can be given a session subject. This is the ablation.")
        log("\nlink A3: --no-bridge, so nothing is emitted")
    elif not alerting:
        a3["note"] = "the baseline raised no alert, so there is nothing to anchor"
    else:
        a3_path.parent.mkdir(parents=True, exist_ok=True)
        a3_path.write_text("")
        out_store = anchors.AnchorStore(a3_path)
        per_subject = {}
        for r in alerting:
            cands = [("session", f"{r['dst']}\\{r['user']}"),
                     ("session", f"{r['src']}\\{r['user']}"),
                     ("host", r["dst"]), ("host", r["src"])]
            b = br.resolve_any("T1021", cands, "T1486")
            if b is None:
                a3["unbound"] += 1
                lab = f"{r['src']}->{r['dst']} as {r['user']}"
                if len(a3["unbound_subjects"]) < 20:
                    a3["unbound_subjects"].append(lab)
                continue
            per_subject.setdefault(b.to_subject, []).append((r, b))
        for subj, items in sorted(per_subject.items()):
            items.sort(key=lambda t: -t[0]["score"])
            keep, drop = items[:50], items[50:]
            a3["capped"] += len(drop)
            for r, b in keep:
                vf = b.to_valid_from if b.to_valid_from is not None else r["time"]
                ttl = b.to_ttl_seconds if b.to_ttl_seconds is not None else a.anchor_ttl
                try:
                    out_store.emit(anchors.Anchor.create(
                        LINK_OUT, "session", subj, emitted_at=float(vf),
                        ttl_seconds=float(ttl), operating_point=own_op, prior=a.prior,
                        evidence={
                            "emitting_stage": "stage 3 lateral movement (T1021)",
                            "detection": "the baseline detector alerted on this path",
                            "src": r["src"], "dst": r["dst"], "user": r["user"],
                            "logon_time": r["time"], "score": r["score"],
                            "rank_in_day": r["rank"],
                            "label_for_bookkeeping_only": r["label"],
                            "bridge_binding": b.as_dict(),
                            "bridge_status": br.status,
                            "window_basis": b.to_time_basis,
                        },
                        note="real detector output: an alerting path in the unanchored baseline"))
                    a3["emitted"] += 1
                    a3["by_subject"][subj] = a3["by_subject"].get(subj, 0) + 1
                except anchors.AnchorError as e:
                    a3["refused"] = str(e)
                    log(f"  REFUSED one A3 anchor: {e}")
        a3["anchors_file"] = a3_path.name
        a3["loaded_anchor_summary"] = anchors.AnchorStore.load(a3_path).summary()
        log(f"\n--- link A3: anchors emitted for stage 4 (T1486) ---")
        log(f"  {a3['alerting_logons']} alerting path(s) in the unanchored baseline; "
            f"wrote {a3_path.name} with {a3['emitted']} anchor(s), subject_kind "
            f"'session', confidence {implied:.4f} at prior {a.prior}")
        for subj, n in sorted(a3["by_subject"].items()):
            log(f"    session {subj!r:28s} {n:4d} anchor(s)")
        if a3["unbound"]:
            log(f"  {a3['unbound']} alerting path(s) have no binding to any session and "
                f"so cannot reach stage 4 (first few: {a3['unbound_subjects'][:5]})")
        if a3["capped"]:
            log(f"  {a3['capped']} further alerting path(s) named a session that already "
                f"had 50 anchors and were not emitted")
        log(f"  next: python3 ../T1486_sessions/scripts/01_run_experiment.py "
            f"--anchor-file {a3_path} --tag {a.tag}")
    a3.update(br.result_block(LINK_OUT, anchor_file=a3["anchors_file"], n_matched=0))
    # How many A2 anchors actually matched something here, which is the number
    # requirement 5 asks for on the consuming side.
    _m = results.get("loaded_all", {}).get("anchor_matched_logons")
    consumed["matched_logons"] = _m
    consumed.update(br.result_block(LINK, anchor_file=a.anchor_file, n_matched=_m))

    rows_file["w"].close()
    _ROWS_FILE["w"] = None
    out = {"config": vars(a), "data": meta, "window": window,
           "assumption_check": assumption,
           "anchor_setup": {
               "module": "common/anchors.py", "link": LINK,
               "emitted_at": emitted_at, "emitted_at_basis": emit_basis,
               "ttl_seconds": a.anchor_ttl, "confidence": anchor_confidence,
               "confidence_source": confidence_source,
               "emitter_operating_point_path": a.emitter_operating_point,
               "emitter_operating_point": (emitter_op.as_dict() if emitter_op else None),
               "prior": anchor_prior,
               "route_at": a.route_at,
               "routing_note": ("A matched anchor no longer routes a path by itself. The "
                                "graded relaxation from common/anchors.py (the link's "
                                "max_drop times the confidence decayed to the logon's own "
                                "time) is compared against --route-at, so an anchor that "
                                "has aged out of usefulness stops diverting traffic even "
                                "while it is still inside its validity window."),
               "anchors_file": anchors_path.name,
               "window_note": ("The validity window is now enforced at match time "
                               "(at=logon time). The earlier code applied the anchor to "
                               "the entire log, including logons before the foothold, "
                               "which inflated the measured cost."),
               "wrong_origin_note": ("The wrong-origin control is emitted at the same "
                                     "moment with the same TTL, so the two settings differ "
                                     "only in which host is named."),
           },
           "results": results, "anchor_cost_curve": curve, "confidence_sweep": sweep,
           "anchor_a2_consumed": consumed,
           "anchor_a3_emitted": a3,
           "anchor_source": ("REAL stage-2 detections read from " + str(a.anchor_file)
                             if loaded_mode else
                             "FABRICATED in this kit from meta['entry_host']"),
           "build_store_called": not loaded_mode,
           "operating_point_emitted": op_block,
           "operating_point_refused": op_refused,
           "operating_point_file": (op_path.name if op_block else None),
           "operating_point_prior_for_report": a.prior,
           "operating_point_note": ("Measured on the BASELINE setting (budget only). tpr "
                                    "is path-level red-team recall; fpr is alerts that "
                                    "were not red team over all non-red-team logons. The "
                                    "next kit turns this into its anchor confidence, so "
                                    "stage 4 never has to assert a number."),
           "per_logon_file": (f"logons_{a.tag}.jsonl" if a.logons_file == "full"
                              else None),
           "per_logon_file_note": ("not written (--logons-file none); rows counted only"
                                   if a.logons_file == "none" else None),
           "per_logon_rows_written": rows_file["w"].rows,
           "per_logon_settings": sorted(set(rows_file["settings"]))}
    save_json(RESULTS / f"experiment_{a.tag}.json", out)

    log("\n--- alerts a day, and what was caught ---")
    log(f"  recall is reported BOTH over logon events and over attack paths "
        f"({base['redteam_total_paths']} path(s) in this log); Hopper reports paths")
    log(f"  baseline (budget only)      {base['alerts_per_day']:6.2f}/day   "
        f"events {base['redteam_detected_events']}/{base['redteam_total_events']}   "
        f"paths {base['redteam_detected_paths']}/{base['redteam_total_paths']}   "
        f"stealthy {base['stealthy_detected']}/{base['stealthy_total']}")
    if "true_origin" in results:
        r = results["true_origin"]
        log(f"  anchored at the true origin {r['alerts_per_day']:6.2f}/day   "
            f"events {r['redteam_detected_events']}/{r['redteam_total_events']}   "
            f"paths {r['redteam_detected_paths']}/{r['redteam_total_paths']}   "
            f"stealthy {r['stealthy_detected']}/{r['stealthy_total']}   [CEILING]")
        log(f"    anchored hosts server-like? {r['anchor_hosts_server_like']}")
    if wrong:
        w = results["wrong_origin"]
        log(f"  anchored at a wrong origin  {w['alerts_per_day_mean']:6.2f}/day "
            f"(sd {w['alerts_per_day_sd']})   event recall "
            f"{w['redteam_recall_over_events_mean']}   path recall "
            f"{w['redteam_recall_over_paths_mean']}")
    for name in ("true_origin", "wrong_origin"):
        if not curve[name]:
            continue
        log(f"\n--- cost as a function of how many hosts are anchored ({name}) ---")
        for c in curve[name]:
            log(f"  n={c['n_anchors']:3d}  {c['alerts_per_day']:6.2f}/day  "
                f"(+{c['extra_alerts_per_day']} over baseline)  "
                f"{c['anchored_paths']:4d} anchored logons   "
                f"event recall {c['redteam_recall_over_events']}   "
                f"path recall {c['redteam_recall_over_paths']}")
    if sweep["wrong_origin"]:
        log("\n--- cost at each confidence threshold (wrong origin; anchors carry "
            f"{a.anchor_confidence if a.anchor_confidence is not None else sorted({round(x.confidence, 6) for x in in_store})}) ---")
        for c in sweep["wrong_origin"]:
            log(f"  >= {c['min_confidence']:.1f}  {c['alerts_per_day']:6.2f}/day  "
                f"(+{c['extra_alerts_per_day']})  {c['anchored_paths']} anchored logons")
    log(f"\nwrote {anchors_path.name} ({len(reloaded)} anchors)"
        + (f" and logons_{a.tag}.jsonl" if a.logons_file == "full" else
           " (per-logon file not written: --logons-file none)"))
    log(f"\nnext: python3 scripts/03_bundle_results.py --tag {a.tag}")
    pr.finish("complete", f"experiment_{a.tag}.json is the result; this file is "
                          f"superseded by it")
    log(f"progress files left for the record: {pr.json_path.name}, {pr.md_path.name}")


if __name__ == "__main__":
    main()
