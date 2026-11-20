"""Build an authentication log to run the experiment on.

Two sources:

  --synthetic   generate an enterprise-shaped log here, with a red-team campaign in it.
                Runs immediately, no download.
  --lanl PATH   read the LANL 2015 'auth.txt.gz' that Hopper, Bai et al. and the graph
                detectors all use, streaming the first N rows. Real data, ~7.6 GB
                compressed, 1.05e9 rows, 58 days: https://csr.lanl.gov/data/cyber1/

Both produce the same shape, so the rest of the kit does not care which was used.
Every row carries a `success` flag: a failed logon is part of the lateral-movement
signal and part of the red-team labels, so it is kept rather than filtered away.

THE LANL PASS IS CHECKPOINTED AND RESUMABLE.
-------------------------------------------
A full pass over auth.txt.gz is 1-3 hours of decompression on a laptop, and the kit
is run against it for days. Losing that to Ctrl-C, a closed lid or a flat battery is
not acceptable, so this script:

  * writes data/auth.jsonl incrementally, never holding the rows in memory (the old
    version built a list of every row and then wrote it, which cannot survive a
    billion rows on any laptop);
  * writes data/.lanl_progress.json every --checkpoint-every input rows (default 5M),
    atomically, recording input rows read, compressed bytes consumed, the last
    timestamp and day reached, wall clock, an estimate of what is left, and the
    config the numbers belong to;
  * on restart, resumes from the checkpoint instead of starting over;
  * on SIGINT or SIGTERM, writes the checkpoint, says what was saved and prints the
    exact command to resume, then exits.

HOW RESUME WORKS, AND WHAT IT COSTS. A gzip stream is not seekable: there is no way
to jump to a byte offset in the decompressed data, and the standard library cannot
serialise a zlib decompressor's state to disk. So resume re-streams auth.txt.gz from
the beginning and SKIPS the input rows already written, counting newlines in raw
binary chunks without decoding, parsing or writing them. The skip is real work -
the gzip still has to be decompressed - but it runs at roughly the raw decompression
rate with none of the per-row cost, and the script says how long it took.

A TIME WINDOW, WHICH IS WHY THE LANL PASS IS RUNNABLE AT ALL.
--------------------------------------------------------------
--from-time and --to-time select the slice of the corpus to KEEP, in the seconds
LANL numbers time in. This is not a convenience, it is what makes the red-team
labels reachable: the first red-team event is at t=150885, about 15 million input
rows in, so streaming from day 0 under a --max-events a laptop can hold produces a
slice with ZERO of the 749 labels in it and every recall figure is a recall over
nothing. scripts/redteam_window.py reads redteam.txt.gz and says which window to
ask for; --redteam-window prints the same report from here.

Rows outside the window are never parsed into a record and never written, so
auth.jsonl - and therefore 02's memory - is the size of the window rather than the
size of the prefix of the corpus it sits in. The gzip is still not seekable, so
reaching --from-time costs the decompression of everything before it; but --to-time
STOPS the pass, because auth.txt.gz is in time order, so nothing past the window is
read at all. The window is recorded in auth_meta.json and in the checkpoint's config,
so a resume cannot splice two different windows together and no number can be quoted
without the window it belongs to.

What makes it CORRECT rather than approximate: the checkpoint records the input row
counter and the output file's byte offset captured at the same instant, after the
output was flushed and fsynced. On resume the output is truncated back to that
offset, which discards any row written after the last durable checkpoint, and the
input is skipped to the matching row. The two counters cannot drift apart, so the
resumed file is the file an uninterrupted run would have written.
"""
import argparse, gzip, json, os, random, signal, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, log, save_json, write_jsonl

SERVER_PREFIX = "SRV"
WEB_PREFIX = "WEB"

PROGRESS_NAME = ".lanl_progress.json"
PROGRESS_FORMAT = 3
CHUNK_BYTES = 1 << 20
SECONDS_PER_DAY = 86400

# ---------------------------------------------------------------- interruption
# A handler that only sets a flag. Writing the checkpoint from inside a signal
# handler would risk interleaving with the write already in progress, so the flag is
# checked at a chunk boundary instead - about a second apart at worst - where the
# output file and the counters are in a consistent state.
_STOP = {"signum": None}


def _on_signal(signum, frame):
    _STOP["signum"] = signum


def install_signal_handlers():
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _on_signal)
        except (ValueError, OSError):
            pass


def fmt_dur(seconds):
    if seconds is None:
        return "unknown"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def atomic_write_json(path, obj):
    """Write JSON via a temp file and rename, so a kill mid-write cannot leave a
    half-written checkpoint that the next run would then try to resume from."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def read_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def synthetic(seed=0, days=7, n_users=180, n_workstations=220, n_servers=25,
              n_web=3, logons_per_user_day=9, redteam_paths=8, stealth_fraction=0.5,
              web_inbound_per_day=(3, 6)):
    """An enterprise-shaped login log.

    Most logons run workstation -> server. Servers rarely initiate an interactive
    logon, which is the assumption the design rests on and which this data lets you
    measure rather than assume. The red team starts at a web server, which is the
    origin anchor A2 supplies.
    """
    rng = random.Random(seed)
    users = [f"u{i:04d}" for i in range(n_users)]
    wks = [f"WKS{i:04d}" for i in range(n_workstations)]
    srv = [f"{SERVER_PREFIX}{i:03d}" for i in range(n_servers)]
    web = [f"{WEB_PREFIX}{i:02d}" for i in range(n_web)]
    hosts = wks + srv + web

    # each user has a small habitual set of destinations
    habit = {u: rng.sample(srv, k=rng.randint(2, 5)) for u in users}
    home = {u: rng.choice(wks) for u in users}

    rows = []
    t = 0
    for day in range(days):
        for u in users:
            for _ in range(rng.randint(max(1, logons_per_user_day - 4), logons_per_user_day + 4)):
                t += rng.randint(5, 120)
                src = home[u] if rng.random() < 0.8 else rng.choice(wks)
                dst = rng.choice(habit[u]) if rng.random() < 0.85 else rng.choice(srv)
                rows.append(dict(time=t, user=u, src=src, dst=dst, logon_type="Network",
                                 success=True, label="benign"))
        # service accounts and admins do a little server-to-server work: the honest
        # part of the base rate that makes the anchor rule cost something
        for _ in range(rng.randint(2, 6)):
            t += rng.randint(50, 400)
            rows.append(dict(time=t, user="svc_monitor", src=rng.choice(srv),
                             dst=rng.choice(srv), logon_type="Network",
                             success=True, label="benign"))
        # Administrators and deploy tooling log in TO the web servers. Without this a
        # WEB host is never a destination, so it fails server_like_hosts' as_dst test and
        # the anchored entry host sits OUTSIDE the very class the assumption check
        # measures - which made the three result rows measure different rules.
        for w in web:
            for _ in range(rng.randint(*web_inbound_per_day)):
                t += rng.randint(40, 300)
                rows.append(dict(time=t, user=rng.choice(["svc_deploy", "admin_ops"]),
                                 src=rng.choice(wks + srv), dst=w, logon_type="Network",
                                 success=True, label="benign"))
        # and, rarely, a web server initiates one - the false-positive source for A2
        if rng.random() < 0.35:
            t += rng.randint(100, 500)
            rows.append(dict(time=t, user="svc_deploy", src=rng.choice(web),
                             dst=rng.choice(srv), logon_type="Network",
                             success=True, label="benign"))

    # --- red team: starts on a web server (the stage-2 foothold), then moves
    entry = rng.choice(web)
    compromised = rng.sample(users, k=3)
    rt_start = int(len(rows) * 0.55)
    t = rows[rt_start]["time"]
    attack = []
    cur_hosts = [entry]
    for i in range(redteam_paths):
        t += rng.randint(30, 900)
        u = rng.choice(compromised)
        src = rng.choice(cur_hosts)
        # a stealthy hop reuses an edge the legitimate user already uses, which is the
        # class Hopper missed; a loud hop goes somewhere the user never goes
        if rng.random() < stealth_fraction and habit[u]:
            dst = rng.choice(habit[u])
            kind = "stealthy"
        else:
            dst = rng.choice([h for h in srv if h not in habit[u]] or srv)
            kind = "loud"
        attack.append(dict(time=t, user=u, src=src, dst=dst, logon_type="Network",
                           success=True, label="redteam", redteam_kind=kind))
        cur_hosts.append(dst)

    rows += attack
    rows.sort(key=lambda r: r["time"])
    meta = dict(source="synthetic", seed=seed, days=days, n_rows=len(rows),
                web_hosts=web, server_hosts=srv, entry_host=entry,
                redteam_events=len(attack),
                redteam_first_time=min(a["time"] for a in attack) if attack else None,
                redteam_stealthy=sum(1 for a in attack if a["redteam_kind"] == "stealthy"),
                failed_logons=0, successful_logons=len(rows))
    return rows, meta


def _load_redteam_labels(path):
    """The red-team label file, read BEFORE the auth stream.

    It has to be first: the src == dst filter below drops rows, and the number of
    red-team rows it drops is a figure that has to be reported rather than lost.

    Reads .gz as well as plain text, and streams rather than slurping: LANL ships
    redteam.txt gzipped alongside auth.txt.gz, and the old read_text() on a gzip
    file produced binary noise that matched nothing, silently losing every label.
    """
    red = set()
    n = 0
    if path and Path(path).exists():
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", errors="replace") as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 4:
                    try:
                        red.add((int(p[0]), p[1], p[2], p[3]))
                    except ValueError:
                        continue
                    n += 1
    return red, n


class GzipLineReader:
    """Chunked line reader over a gzip (or plain) file, with a cheap skip.

    Lines are produced a chunk at a time rather than one at a time so that the
    interruption flag, the checkpoint and the progress line are all checked at a
    natural boundary, and so that skipping on resume can count newlines in raw bytes
    without building a str per line.
    """

    def __init__(self, path, chunk_bytes=CHUNK_BYTES):
        self.path = str(path)
        self.chunk_bytes = chunk_bytes
        self.compressed_total = os.path.getsize(self.path)
        self._raw = open(self.path, "rb")
        self._gz = (gzip.GzipFile(fileobj=self._raw, mode="rb")
                    if self.path.endswith(".gz") else self._raw)
        self._buf = b""

    def compressed_consumed(self):
        """Bytes of the compressed file read so far. Slightly ahead of the logical
        position because the gzip reader buffers, which is fine for a percentage."""
        try:
            return self._raw.tell()
        except (OSError, ValueError):
            return None

    def close(self):
        for f in (self._gz, self._raw):
            try:
                f.close()
            except Exception:
                pass

    def skip_lines(self, n):
        """Consume exactly n lines without decoding or parsing them."""
        left = int(n)
        while left > 0:
            if self._buf:
                c = self._buf.count(b"\n")
                if c >= left:
                    pos = -1
                    for _ in range(left):
                        pos = self._buf.index(b"\n", pos + 1)
                    self._buf = self._buf[pos + 1:]
                    return n
                left -= c
                self._buf = self._buf[self._buf.rfind(b"\n") + 1:]
            data = self._gz.read(self.chunk_bytes)
            if not data:
                raise EOFError(
                    f"{self.path} ran out after skipping {n - left:,} of {n:,} lines. "
                    f"The file is shorter than the checkpoint says it was, so the "
                    f"checkpoint does not belong to this file.")
            self._buf += data
        return n

    def chunks(self):
        """Yield lists of complete lines (str, newline stripped), in file order."""
        while True:
            data = self._gz.read(self.chunk_bytes)
            if not data:
                if self._buf:
                    tail, self._buf = self._buf, b""
                    yield [tail.decode("utf-8", "replace")]
                return
            self._buf += data
            nl = self._buf.rfind(b"\n")
            if nl < 0:
                continue
            block, self._buf = self._buf[:nl], self._buf[nl + 1:]
            yield block.decode("utf-8", "replace").split("\n")


def lanl_config(a, out_path):
    """The identity of a LANL pass. A checkpoint is only resumable into a run whose
    config is identical: a different --max-events or a different (or re-downloaded)
    auth.txt.gz means the partial auth.jsonl is not a prefix of what this run would
    write, and resuming would splice two different datasets together."""
    src = Path(a.lanl)
    st = src.stat()
    rt = Path(a.lanl_redteam) if a.lanl_redteam else None
    return {
        "format": PROGRESS_FORMAT,
        "lanl_path": str(src.resolve()),
        "lanl_size": st.st_size,
        "lanl_mtime_ns": st.st_mtime_ns,
        "max_events": int(a.max_events),
        # The window is part of the identity of the pass: a partial auth.jsonl written
        # under one window is not a prefix of what another window would write.
        "from_time": (None if a.from_time is None else int(a.from_time)),
        "to_time": (None if a.to_time is None else int(a.to_time)),
        "redteam_path": str(rt.resolve()) if rt else None,
        "redteam_size": (rt.stat().st_size if rt and rt.exists() else None),
        "out_path": str(Path(out_path).resolve()),
    }


def config_diff(old, new):
    if not isinstance(old, dict):
        return ["the checkpoint carries no config block at all"]
    out = []
    for k in sorted(set(old) | set(new)):
        if old.get(k) != new.get(k):
            out.append(f"{k}: checkpoint {old.get(k)!r} != this run {new.get(k)!r}")
    return out


def resume_command(a):
    bits = [f"python3 scripts/01_make_auth_data.py --lanl {a.lanl}"]
    if a.lanl_redteam:
        bits.append(f"--lanl-redteam {a.lanl_redteam}")
    if a.from_time is not None:
        bits.append(f"--from-time {int(a.from_time)}")
    if a.to_time is not None:
        bits.append(f"--to-time {int(a.to_time)}")
    bits.append(f"--max-events {a.max_events}")
    if a.checkpoint_every != 5_000_000:
        bits.append(f"--checkpoint-every {a.checkpoint_every}")
    return " ".join(bits)


def from_lanl_streaming(a, out_path, progress_path, config, resume=None):
    """Stream LANL auth.txt.gz: time,user@dom,user@dom,src,dst,authtype,logontype,orient,success.

    Failed logons are KEPT, with success=False. Filtering them out discarded part of
    the lateral-movement signal - a failed attempt is how a wrong credential looks,
    and brute-forcing is exactly what a hop looks like before it works - and it also
    discarded red-team labelled rows, because the red team failed too.

    Rows go to out_path as they are parsed. Nothing but the red-team label set and a
    handful of counters is held in memory, so the pass is flat in RAM whether it
    reads two million rows or all 1.05 billion.
    """
    red, n_label_rows = _load_redteam_labels(a.lanl_redteam)
    log(f"red-team labels: {n_label_rows} row(s) from {a.lanl_redteam}"
        if a.lanl_redteam else "red-team labels: none given, every row will be benign")

    stats = dict(lines_read=0, malformed=0, self_logons_dropped=0,
                 redteam_rows_dropped_self_logon=0, failed_logons=0, successful_logons=0,
                 rows_before_window=0, rows_after_window=0)
    st = dict(rows_written=0, redteam_matched=0, redteam_first_time=None,
              first_time=None, last_time=None, prior_elapsed=0.0, resumes=0,
              stopped_at_to_time=False, last_time_seen=None)
    from_time = None if a.from_time is None else int(a.from_time)
    to_time = None if a.to_time is None else int(a.to_time)
    if from_time is not None and to_time is not None and to_time < from_time:
        raise SystemExit(f"--to-time {to_time} is before --from-time {from_time}; "
                         f"that window holds nothing.")
    if from_time is not None or to_time is not None:
        lo = f"t >= {from_time}" if from_time is not None else "no lower bound"
        hi = f"t <= {to_time}" if to_time is not None else "no upper bound"
        span = ("" if (from_time is None or to_time is None) else
                f"  ({(to_time - from_time) / SECONDS_PER_DAY:.2f} day(s) of log)")
        log(f"window: keeping rows with {lo} and {hi}{span}")
        log("  rows outside it are skipped without being parsed or written, so "
            "auth.jsonl is the size of the window.")
        if to_time is not None:
            log("  the pass STOPS at the first timestamp past --to-time: auth.txt.gz is "
                "in time order, so nothing after the window is read.")

    if resume is not None:
        stats.update({k: int(v) for k, v in resume["stats"].items() if k in stats})
        st.update(rows_written=int(resume["rows_written"]),
                  redteam_matched=int(resume["redteam_matched"]),
                  redteam_first_time=resume["redteam_first_time"],
                  first_time=resume["first_time"], last_time=resume["last_time"],
                  prior_elapsed=float(resume.get("elapsed_seconds") or 0.0),
                  resumes=int(resume.get("resumes") or 0) + 1)
        # Truncate the output back to the offset the checkpoint vouches for. Rows
        # after it were written after the last durable checkpoint, so their input
        # position is unknown and they cannot be kept.
        with open(out_path, "r+b") as f:
            f.truncate(int(resume["out_bytes"]))
        out = open(out_path, "ab", buffering=1 << 20)
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out = open(out_path, "wb", buffering=1 << 20)

    reader = GzipLineReader(a.lanl)
    started = time.time()

    if resume is not None and stats["lines_read"]:
        n = stats["lines_read"]
        gb = (resume.get("compressed_consumed") or 0) / 1e9
        log(f"resume: SKIPPING the first {n:,} input rows ({gb:.2f} GB of gzip). "
            f"gzip is not seekable, so this re-decompresses that prefix without "
            f"parsing or writing it; expect roughly "
            f"{fmt_dur(gb * 60)} at a typical laptop's ~17 MB/s of gzip.")
        t = time.time()
        reader.skip_lines(n)
        log(f"resume: skipped {n:,} rows in {fmt_dur(time.time() - t)}. Continuing at "
            f"input row {n:,}, output row {st['rows_written']:,}, "
            f"byte {int(resume['out_bytes']):,} of {Path(out_path).name}.")

    def snapshot(status, note=""):
        out.flush()
        os.fsync(out.fileno())
        now = time.time()
        elapsed = st["prior_elapsed"] + (now - started)
        done_rows = stats["lines_read"]
        cc = reader.compressed_consumed()
        # rows/s over THIS process's share of the work, which is the rate that
        # predicts what is left
        this_rows = done_rows - (int(resume["stats"]["lines_read"]) if resume else 0)
        rate = this_rows / max(1e-9, now - started)
        by_rows = (a.max_events - done_rows) / rate if rate > 0 else None
        by_bytes = None
        if cc and rate > 0 and reader.compressed_total:
            brate = (cc - (resume.get("compressed_consumed") or 0 if resume else 0)) \
                    / max(1e-9, now - started)
            if brate > 0:
                by_bytes = (reader.compressed_total - cc) / brate
        remaining = min([x for x in (by_rows, by_bytes) if x is not None], default=None)
        day = None
        if st["last_time"] is not None and st["first_time"] is not None:
            day = (int(st["last_time"]) - int(st["first_time"])) // SECONDS_PER_DAY + 1
        payload = {
            "format": PROGRESS_FORMAT,
            "stage": "01_make_auth_data --lanl",
            "status": status,
            "note": note,
            "pid": os.getpid(),
            "config": config,
            "cli": {k: v for k, v in vars(a).items()},
            "lines_read": done_rows,
            "rows_written": st["rows_written"],
            "out_bytes": out.tell(),
            "out_path": str(Path(out_path).resolve()),
            "max_events": a.max_events,
            "pct_of_max_events": round(100.0 * done_rows / max(1, a.max_events), 3),
            "compressed_consumed": cc,
            "compressed_total": reader.compressed_total,
            "pct_of_gzip": (round(100.0 * cc / reader.compressed_total, 4)
                            if cc else None),
            "first_time": st["first_time"],
            "last_time": st["last_time"],
            # the timestamp the INPUT has reached, which is the only progress there is
            # while the pass is still skipping its way to --from-time
            "last_time_seen": st["last_time_seen"],
            "window": {"from_time": from_time, "to_time": to_time,
                       "rows_before_window": stats["rows_before_window"],
                       "stopped_at_to_time": st["stopped_at_to_time"]},
            "day": day,
            "days_covered": day,
            "rows_per_second": round(rate, 1),
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_human": fmt_dur(elapsed),
            "est_remaining_seconds": (round(remaining, 1) if remaining is not None else None),
            "est_remaining_human": fmt_dur(remaining),
            "est_remaining_basis": ("rows to --max-events" if remaining == by_rows
                                    else "compressed bytes to end of file"),
            "redteam_matched": st["redteam_matched"],
            "redteam_first_time": st["redteam_first_time"],
            "redteam_label_rows_in_file": n_label_rows,
            "stats": dict(stats),
            "resumes": st["resumes"],
            "started_at": round(started, 1),
            "updated_at": round(now, 1),
            "resume_command": resume_command(a),
        }
        atomic_write_json(progress_path, payload)
        return payload

    def heartbeat(p):
        log(f"  [{p['elapsed_human']:>7}] {p['lines_read']:>12,} rows read "
            f"({p['pct_of_max_events']:.2f}% of --max-events, "
            f"{p['pct_of_gzip'] if p['pct_of_gzip'] is not None else float('nan'):.3f}% of the gzip)  "
            f"{p['rows_written']:,} written  "
        + (f"input at t={p['last_time_seen']}"
           + (f", window starts at {from_time}" if (from_time is not None
                                                    and not p['rows_written']) else "")
           if p['last_time_seen'] is not None else f"day {p['day']}")
        + f"  "
            f"{p['rows_per_second']:,.0f} rows/s  ETA {p['est_remaining_human']}")

    last_ckpt = stats["lines_read"]
    last_beat = time.time()
    max_events = int(a.max_events)
    done = False
    try:
        for lines in reader.chunks():
            for line in lines:
                if stats["lines_read"] >= max_events:
                    done = True
                    break
                stats["lines_read"] += 1
                # The timestamp is read on its own, BEFORE the row is split into nine
                # fields, so a row outside the window costs one find() and one int()
                # rather than a nine-way split and a dict.
                c = line.find(",")
                if c < 0:
                    stats["malformed"] += 1
                    continue
                try:
                    ts = int(line[:c])
                except ValueError:
                    stats["malformed"] += 1
                    continue
                st["last_time_seen"] = ts
                if from_time is not None and ts < from_time:
                    stats["rows_before_window"] += 1
                    continue
                if to_time is not None and ts > to_time:
                    # auth.txt.gz is in time order, so this is the end of the window and
                    # there is nothing further to read.
                    stats["rows_after_window"] += 1
                    st["stopped_at_to_time"] = True
                    done = True
                    break
                p = line.rstrip("\r").split(",")
                if len(p) < 9:
                    stats["malformed"] += 1
                    continue
                src, dst = p[3], p[4]
                key = (ts, p[1], src, dst)
                if src == dst:
                    # A host authenticating to itself is not a hop, so it is still
                    # dropped - but count it, and count how many red-team rows go with
                    # it, because those are labels the evaluation can never score.
                    stats["self_logons_dropped"] += 1
                    if key in red:
                        stats["redteam_rows_dropped_self_logon"] += 1
                    continue
                success = (p[8] == "Success")
                stats["successful_logons" if success else "failed_logons"] += 1
                r = {"time": ts, "user": p[1], "src": src, "dst": dst,
                     "logon_type": p[6], "success": success, "label": "benign"}
                if key in red:
                    r["label"] = "redteam"
                    r["redteam_kind"] = "unknown"
                    st["redteam_matched"] += 1
                    if st["redteam_first_time"] is None or ts < st["redteam_first_time"]:
                        st["redteam_first_time"] = ts
                out.write((json.dumps(r) + "\n").encode())
                st["rows_written"] += 1
                if st["first_time"] is None:
                    st["first_time"] = ts
                st["last_time"] = ts
            if done:
                break
            if _STOP["signum"] is not None:
                name = signal.Signals(_STOP["signum"]).name
                p = snapshot("interrupted", f"caught {name} at a chunk boundary")
                out.close()
                reader.close()
                log(f"\n{name}: stopping cleanly.")
                log(f"  SAVED: {p['rows_written']:,} rows in {Path(out_path).name} "
                    f"({p['out_bytes']:,} bytes), {p['lines_read']:,} input rows read "
                    f"({p['pct_of_max_events']:.2f}% of --max-events, "
                    f"{p['pct_of_gzip']}% of the gzip), last timestamp "
                    f"{p['last_time']} (day {p['day']}), {p['elapsed_human']} elapsed.")
                log(f"  checkpoint: {progress_path}")
                log(f"  RESUME WITH:\n    {p['resume_command']}")
                log(f"  (it will skip the {p['lines_read']:,} rows already written "
                    f"before doing any new work)")
                sys.exit(130 if _STOP["signum"] == signal.SIGINT else 143)
            now = time.time()
            if stats["lines_read"] - last_ckpt >= a.checkpoint_every:
                heartbeat(snapshot("running"))
                last_ckpt = stats["lines_read"]
                last_beat = now
            elif now - last_beat >= a.heartbeat:
                heartbeat(snapshot("running"))
                last_beat = now
    finally:
        try:
            out.flush()
            os.fsync(out.fileno())
        except Exception:
            pass

    final = snapshot("complete", (
        "the pass stopped at --to-time; auth.jsonl holds the whole window"
        if st["stopped_at_to_time"] else "the pass finished; auth.jsonl is whole"))
    out.close()
    reader.close()

    days = final["day"]
    meta = dict(source="lanl", path=str(a.lanl), n_rows=st["rows_written"],
                redteam_events=st["redteam_matched"],
                redteam_first_time=st["redteam_first_time"],
                redteam_label_rows_in_file=n_label_rows,
                redteam_label_rows_matched=st["redteam_matched"],
                redteam_label_rows_unmatched=max(0, n_label_rows - st["redteam_matched"]
                                                 - stats["redteam_rows_dropped_self_logon"]),
                first_time=st["first_time"], last_time=st["last_time"],
                days_covered=days,
                streamed=True, checkpointed=True, resumes=st["resumes"],
                checkpoint_file=PROGRESS_NAME,
                compressed_bytes_consumed=final["compressed_consumed"],
                compressed_bytes_total=final["compressed_total"],
                wall_clock_seconds=final["elapsed_seconds"],
                **stats)
    meta["last_time_seen"] = st["last_time_seen"]
    if from_time is not None or to_time is not None:
        # Recorded on the data itself, so 02 and the bundler can quote the window
        # beside every number instead of relying on anyone remembering it.
        meta.update(
            window_applied=True,
            window_from_time=from_time,
            window_to_time=to_time,
            window_days=(None if (from_time is None or to_time is None) else
                         round((to_time - from_time) / SECONDS_PER_DAY, 4)),
            window_rows_before=stats["rows_before_window"],
            window_stopped_at_to_time=st["stopped_at_to_time"],
            window_note=("every figure computed from this file is a figure FOR THIS "
                         "WINDOW, not for the 58-day corpus"))
    else:
        meta.update(window_applied=False)
    return meta


def summarise_checkpoint(p):
    log(f"  status           {p.get('status')}")
    log(f"  input rows read  {p.get('lines_read'):,} of --max-events "
        f"{p.get('max_events'):,}  ({p.get('pct_of_max_events')}%)")
    log(f"  rows written     {p.get('rows_written'):,} -> {p.get('out_bytes'):,} bytes")
    log(f"  gzip consumed    {p.get('compressed_consumed')} of "
        f"{p.get('compressed_total')} bytes ({p.get('pct_of_gzip')}%)")
    log(f"  last timestamp   {p.get('last_time')}  (day {p.get('day')})")
    log(f"  wall clock       {p.get('elapsed_human')}   ETA {p.get('est_remaining_human')}")
    log(f"  red team matched {p.get('redteam_matched')} of "
        f"{p.get('redteam_label_rows_in_file')} label rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lanl", help="path to auth.txt.gz")
    ap.add_argument("--lanl-redteam", help="path to redteam.txt (or redteam.txt.gz)")
    ap.add_argument("--max-events", type=int, default=3_000_000)
    ap.add_argument("--from-time", type=int, default=None,
                    help="keep only rows at or after this time, in the unix seconds LANL "
                         "numbers time in (t=1 is the first row of the corpus). The point "
                         "is the red-team labels: the first one is at t=150885, so a slice "
                         "taken from day 0 under any --max-events a laptop can hold "
                         "contains none of them. scripts/redteam_window.py says which "
                         "window to ask for.")
    ap.add_argument("--to-time", type=int, default=None,
                    help="keep only rows at or before this time. The pass STOPS here "
                         "rather than reading on: auth.txt.gz is in time order, so the "
                         "rest of the 7.6 GB is never decompressed.")
    ap.add_argument("--redteam-window", action="store_true",
                    help="read the red-team label file, report where the labelled events "
                         "are and what window to take, and exit without touching "
                         "auth.txt.gz. The same report as scripts/redteam_window.py.")
    ap.add_argument("--margin-hours", type=float, default=6.0,
                    help="context either side of the labelled events in the window "
                         "--redteam-window suggests")
    ap.add_argument("--allow-zero-redteam", action="store_true",
                    help="permit a LANL slice in which NONE of the red-team labels match "
                         "a kept logon. Without this the pass is a hard error, because a "
                         "slice with no labels in it silently turns every recall figure "
                         "into a recall over nothing - which is what happens to any slice "
                         "taken from day 0.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--checkpoint-every", type=int, default=5_000_000,
                    help="input rows between checkpoints on the LANL pass. Smaller "
                         "loses less work to a kill; larger costs fewer fsyncs.")
    ap.add_argument("--heartbeat", type=int, default=30,
                    help="seconds between progress lines on the LANL pass")
    ap.add_argument("--restart", action="store_true",
                    help="discard any checkpoint and start the LANL pass over")
    ap.add_argument("--status", action="store_true",
                    help="print the LANL checkpoint and exit, changing nothing")
    a = ap.parse_args()

    out_path = DATA / "auth.jsonl"
    progress_path = DATA / PROGRESS_NAME

    if a.redteam_window:
        import redteam_window
        path = a.lanl_redteam or "../_shared_data/lanl/redteam.txt.gz"
        redteam_window.report(path, margin_hours=a.margin_hours)
        return

    if a.status:
        p = read_json(progress_path)
        if p is None:
            log(f"no checkpoint at {progress_path}")
            return
        log(f"checkpoint {progress_path}")
        summarise_checkpoint(p)
        log(f"  resume with:   {p.get('resume_command')}")
        return

    if a.lanl:
        config = lanl_config(a, out_path)
        prev = read_json(progress_path)
        if a.restart and prev is not None:
            log(f"--restart: discarding the checkpoint at {progress_path} and the "
                f"partial {out_path.name}; this pass starts from row 0.")
            prev = None
            progress_path.unlink(missing_ok=True)
        resume = None
        if prev is not None:
            diff = config_diff(prev.get("config"), config)
            log(f"found a checkpoint at {progress_path}:")
            summarise_checkpoint(prev)
            if prev.get("status") == "complete" and not diff:
                log(f"\nThis pass is ALREADY COMPLETE and the config is unchanged, so "
                    f"there is nothing to do: {out_path.name} holds "
                    f"{prev.get('rows_written'):,} rows.")
                log(f"  re-run it anyway with --restart")
                log("next: python3 scripts/02_run_experiment.py")
                return
            if not out_path.exists():
                raise SystemExit(
                    f"\nREFUSING to resume: the checkpoint says {prev.get('rows_written'):,} "
                    f"rows were written but {out_path} does not exist. The partial output "
                    f"was deleted or moved, so there is nothing to resume onto.\n"
                    f"  start over with: {resume_command(a)} --restart")
            if diff:
                raise SystemExit(
                    "\nREFUSING to resume: the checkpoint belongs to a different run.\n  "
                    + "\n  ".join(diff)
                    + f"\nResuming would splice two different datasets into one "
                      f"{out_path.name}.\n  start over with: {resume_command(a)} --restart")
            have = out_path.stat().st_size
            if have < int(prev.get("out_bytes", 0)):
                raise SystemExit(
                    f"\nREFUSING to resume: {out_path.name} is {have:,} bytes but the "
                    f"checkpoint vouches for {int(prev['out_bytes']):,}. The output was "
                    f"truncated behind the checkpoint, so the two no longer agree.\n"
                    f"  start over with: {resume_command(a)} --restart")
            extra = have - int(prev["out_bytes"])
            log(f"\nRESUMING. {out_path.name} is {have:,} bytes; the checkpoint vouches "
                f"for {int(prev['out_bytes']):,}" + (
                    f", so the last {extra:,} byte(s) written after it will be dropped "
                    f"and re-read." if extra else " exactly."))
            resume = prev
        install_signal_handlers()
        meta = from_lanl_streaming(a, out_path, progress_path, config, resume=resume)
        save_json(DATA / "auth_meta.json", meta)
        rows = None
        # A red-team file that matches nothing is not a result, it is a broken slice.
        # The meta is saved first so the pass is not lost, and then this is an error
        # rather than a line of output nobody reads: every recall number computed on a
        # label-free slice is a recall over zero positives, and the old behaviour was to
        # print "0 matched" and carry on to produce exactly those numbers.
        if (a.lanl_redteam and meta["redteam_label_rows_in_file"]
                and meta["redteam_label_rows_matched"] == 0
                and not a.allow_zero_redteam):
            if (meta.get("window_applied") and meta["n_rows"] == 0
                    and meta["lines_read"] >= int(a.max_events)):
                raise SystemExit(
                    f"\nHARD ERROR: --max-events {a.max_events:,} stopped the pass after "
                    f"{meta['lines_read']:,} input rows, and the window "
                    f"t={meta['window_from_time']}..{meta['window_to_time']} had not been "
                    f"reached yet - the input was still at t="
                    f"{meta.get('last_time_seen') or 'unknown'} when the pass was cut "
                    f"off, so NOT ONE row was written.\n"
                    f"  --max-events counts INPUT rows, and LANL's rows per second of log "
                    f"time vary: t=147285 is about 29 million rows in, not 15 million.\n"
                    f"  --to-time already stops the pass at the end of the window, so "
                    f"--max-events only needs to be larger than the window's end:\n"
                    f"    {resume_command(a).replace('--max-events ' + str(a.max_events), '--max-events 1000000000')} --restart")
            w = (f"the window t={meta.get('window_from_time')}..{meta.get('window_to_time')}"
                 if meta.get("window_applied") else
                 f"the first {meta['lines_read']:,} input rows "
                 f"(t={meta['first_time']}..{meta['last_time']}, "
                 f"day 1..{meta['days_covered']})")
            raise SystemExit(
                f"\nHARD ERROR: {meta['redteam_label_rows_in_file']} red-team label row(s) "
                f"were read from {a.lanl_redteam} and NONE of them matched a kept logon.\n"
                f"  {w} contains no labelled red-team activity at all.\n"
                f"  Every recall figure computed on this slice would be a recall over zero "
                f"positives.\n"
                f"  {meta['n_rows']:,} logons were still written to {out_path.name}, so "
                f"nothing is lost - but choose a window that contains labels:\n"
                f"    python3 scripts/redteam_window.py {a.lanl_redteam}\n"
                f"  and re-run with the --from-time/--to-time it prints.\n"
                f"  If a label-free slice is what you actually want (a pure "
                f"false-positive-rate measurement), say so with --allow-zero-redteam.")
    else:
        rows, meta = synthetic(seed=a.seed, days=a.days)
        write_jsonl(out_path, rows)
        save_json(DATA / "auth_meta.json", meta)

    log(f"\n{meta['n_rows']} logons, {meta.get('redteam_events',0)} of them red team")
    log(f"  {meta.get('successful_logons', 0)} successful, "
        f"{meta.get('failed_logons', 0)} failed (failures are kept, not filtered)")
    if meta["source"] == "lanl":
        log(f"  {meta['self_logons_dropped']} src==dst rows dropped, of which "
            f"{meta['redteam_rows_dropped_self_logon']} carried a red-team label")
        log(f"  red-team label file had {meta['redteam_label_rows_in_file']} rows; "
            f"{meta['redteam_label_rows_matched']} matched a kept logon, "
            f"{meta['redteam_label_rows_unmatched']} matched nothing in this slice")
        if meta.get("window_applied"):
            log(f"  WINDOW t={meta['window_from_time']}..{meta['window_to_time']}: "
                f"{meta['window_rows_before']:,} input rows before it were skipped "
                f"without being written"
                + ("; the pass STOPPED at --to-time rather than reading on"
                   if meta.get("window_stopped_at_to_time") else ""))
            log(f"  every number from this file is a number FOR THIS WINDOW")
        log(f"  {meta['lines_read']:,} input rows read in "
            f"{fmt_dur(meta['wall_clock_seconds'])}"
            + (f" across {meta['resumes']} resume(s)" if meta['resumes'] else "")
            + f"; days 1..{meta['days_covered']}")
    log("next: python3 scripts/02_run_experiment.py")


if __name__ == "__main__":
    main()
