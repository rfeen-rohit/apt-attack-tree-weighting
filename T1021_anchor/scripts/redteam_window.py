"""Which slice of LANL actually contains red-team activity, and how to ask for it.

The problem this exists to solve: auth.txt.gz is 58 days and 1.05e9 rows, and the
FIRST red-team label is at t=150885 - about 15 million input rows in. Streaming from
day 0 with a --max-events small enough to fit a laptop therefore produces a slice with
ZERO red-team labels in it, and every recall number computed on that slice is a recall
over nothing. A 2-million-row slice matches 0 of the 749 labels.

So before building anything, read the label file and ask it where to look. This module
reports the first and last red-team timestamp, the day each falls in, the per-day event
counts, and a suggested --from-time/--to-time window with a stated margin either side.

Run it directly, or via 01_make_auth_data.py --redteam-window.

    python3 scripts/redteam_window.py ../_shared_data/lanl/redteam.txt.gz
    python3 scripts/redteam_window.py <file> --margin-hours 6 --top-days 5

LANL numbers time in seconds from 1, so "day d" below means the half-open interval
[d*86400, (d+1)*86400) - the same arithmetic 01 and 02 use, and the same numbers you
pass to --from-time / --to-time.
"""
import argparse, gzip, sys
from collections import Counter
from pathlib import Path

SECONDS_PER_DAY = 86400


def read_redteam(path):
    """Every (time, user, src, dst) in the label file. Reads .gz as well as plain text.

    The .gz handling is not cosmetic: read_text() on a gzip file returns binary noise
    that matches nothing, which is how 749 labels silently became 0.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"no red-team label file at {p}")
    opener = gzip.open if str(p).endswith(".gz") else open
    rows, malformed = [], 0
    with opener(p, "rt", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                if line.strip():
                    malformed += 1
                continue
            try:
                rows.append((int(parts[0]), parts[1], parts[2], parts[3]))
            except ValueError:
                malformed += 1
    return rows, malformed


# Measured on this corpus, for sizing a window before paying for it:
#   1,087,893 logons were kept from the 12,973-second window t=147285..160258,
#   i.e. 83.9 logons a second of log time, 7.25 million a day.
#   02_run_experiment.py's column store measured 142 bytes of peak RSS per logon on
#   that slice (122 bytes marginal), so a window of D days of log time costs roughly
#   7.25e6 * D logons and 0.95 * D GB of RAM.
LOGONS_PER_SECOND = 83.9
BYTES_PER_LOGON = 142


def _cost(seconds):
    logons = LOGONS_PER_SECOND * max(0, seconds)
    return logons, logons * BYTES_PER_LOGON / 1e9


def describe(rows, margin_hours=6.0, top_days=8, lead_hours=None, trail_hours=None):
    """The numbers a slice is chosen from."""
    if not rows:
        return None
    ts = sorted(r[0] for r in rows)
    per_day = Counter(t // SECONDS_PER_DAY for t in ts)
    margin = int(round(margin_hours * 3600))
    # ASYMMETRIC BY DEFAULT, and this is the whole reason the helper exists rather
    # than a one-line grep. The anomaly score is "has this user used this edge, and
    # reached this destination, on an EARLIER day?" - so a window that starts a few
    # hours before the attack gives the scorer no earlier day to compare against.
    # Measured on t=147285..160258: every one of 1,087,893 logons scored either 3.5 or
    # 4.5, because in a cold window every edge is new; the ten red-team logons ranked
    # 361,474th to 828,184th of the day and a budget of nine caught none of them. The
    # slice was not too small, it was too LATE. Lead with at least one whole day of
    # benign history, so "an earlier day" means something.
    lead = int(round((lead_hours if lead_hours is not None else 24.0) * 3600))
    trail = int(round((trail_hours if trail_hours is not None else margin_hours) * 3600))
    # The suggested window is around the FIRST burst, not around all 58 days: the
    # whole span is 29 days of input and defeats the purpose. The first burst is the
    # cheapest slice that contains real labels.
    first_day = min(per_day)
    busiest_day, busiest_n = max(per_day.items(), key=lambda kv: (kv[1], -kv[0]))
    first_burst = [t for t in ts if t // SECONDS_PER_DAY == first_day]
    return {
        "label_rows": len(rows),
        "first_time": ts[0],
        "last_time": ts[-1],
        "first_day": ts[0] // SECONDS_PER_DAY,
        "last_day": ts[-1] // SECONDS_PER_DAY,
        "days_with_events": len(per_day),
        "per_day": dict(sorted(per_day.items())),
        "top_days": sorted(per_day.items(), key=lambda kv: (-kv[1], kv[0]))[:top_days],
        "busiest_day": busiest_day,
        "busiest_day_events": busiest_n,
        "margin_seconds": margin,
        "lead_seconds": lead,
        "trail_seconds": trail,
        "suggested": {
            "name": "first burst + history",
            "from_time": max(0, first_burst[0] - lead),
            "to_time": first_burst[-1] + trail,
            "events_inside": sum(1 for t in ts
                                 if max(0, first_burst[0] - lead) <= t
                                 <= first_burst[-1] + trail),
        },
        "suggested_cold": {
            "name": "first burst, NO history",
            "from_time": max(0, first_burst[0] - margin),
            "to_time": first_burst[-1] + margin,
            "events_inside": sum(1 for t in ts
                                 if max(0, first_burst[0] - margin) <= t
                                 <= first_burst[-1] + margin),
        },
        "suggested_busiest": {
            "name": f"busiest day ({busiest_day}) + history",
            "from_time": max(0, min(t for t in ts if t // SECONDS_PER_DAY == busiest_day)
                             - lead),
            "to_time": max(t for t in ts if t // SECONDS_PER_DAY == busiest_day) + margin,
            "events_inside": busiest_n,
        },
        "suggested_all": {
            "name": "every labelled event",
            "from_time": max(0, ts[0] - lead),
            "to_time": ts[-1] + margin,
            "events_inside": len(ts),
        },
    }


def _window_line(w, d):
    span = w["to_time"] - w["from_time"]
    logons, gb = _cost(span)
    return (f"  {w['name']:28s} --from-time {w['from_time']:<9d} "
            f"--to-time {w['to_time']:<9d}\n"
            f"      {span / SECONDS_PER_DAY:6.2f} day(s) of log, "
            f"{w['events_inside']:4d} of {d['label_rows']} label(s) inside, "
            f"~{logons / 1e6:.1f}M logons, ~{gb:.2f} GB of RAM in 02")


def report(path, margin_hours=6.0, top_days=8, out=print, lead_hours=None,
           trail_hours=None):
    rows, malformed = read_redteam(path)
    out(f"red-team label file: {path}")
    out(f"  {len(rows)} label row(s) read"
        + (f", {malformed} malformed line(s)" if malformed else ""))
    if not rows:
        raise SystemExit(
            "the label file parsed to ZERO rows. If it is a .gz, check it is not "
            "truncated; if it is plain text, check the columns are "
            "time,user@domain,src,dst.")
    d = describe(rows, margin_hours=margin_hours, top_days=top_days,
                 lead_hours=lead_hours, trail_hours=trail_hours)
    out("")
    out(f"  first red-team event  t={d['first_time']:<9d} (day {d['first_day']})")
    out(f"  last  red-team event  t={d['last_time']:<9d} (day {d['last_day']})")
    out(f"  labelled events span  {d['days_with_events']} day(s) of the 58 in the corpus")
    out("")
    out(f"  events per day (day = t // 86400, the same arithmetic 01 and 02 use):")
    line = "    "
    for day, n in d["per_day"].items():
        bit = f"day {day:2d}: {n:4d}   "
        if len(line) + len(bit) > 96:
            out(line.rstrip())
            line = "    "
        line += bit
    if line.strip():
        out(line.rstrip())
    out("")
    out(f"  busiest single day: day {d['busiest_day']} with {d['busiest_day_events']} "
        f"event(s)")
    out("")
    out(f"SUGGESTED WINDOWS ({d['lead_seconds'] / 3600:.1f}h of benign history BEFORE "
        f"the first event, {d['trail_seconds'] / 3600:.1f}h after the last):")
    for key in ("suggested", "suggested_busiest", "suggested_all", "suggested_cold"):
        out(_window_line(d[key], d))
    out("")
    out("WHY THE LEAD MATTERS, measured rather than argued. The score asks whether this")
    out("user has used this edge, and reached this destination, on an EARLIER day. A")
    out("window that opens a few hours before the attack has no earlier day in it, so")
    out("every logon is novel and the ranking carries no information. On the cold window")
    out("t=147285..160258 all 1,087,893 logons scored 3.5 or 4.5, the ten red-team logons")
    out("ranked 361,474th to 828,184th of the day, and a budget of nine caught none of")
    out("them - a recall of 0/10 events and 0/1 paths that says nothing about the")
    out("detector. Lead with at least a day; --lead-hours sets it.")
    out("")
    s = d["suggested"]
    out("Note that")
    out("--from-time does NOT make the gzip seekable: 01 still has to decompress from the")
    out("start of auth.txt.gz to reach it, it just does not WRITE or hold the rows before")
    out("the window - so auth.jsonl, and 02's memory, are the size of the window alone.")
    out("01 STOPS as soon as it sees a timestamp past --to-time, so the rest of the 7.6 GB")
    out("is never read at all.")
    out("")
    out("  python3 scripts/01_make_auth_data.py \\")
    out(f"      --lanl ../_shared_data/lanl/auth.txt.gz \\")
    out(f"      --lanl-redteam {path} \\")
    out(f"      --from-time {s['from_time']} --to-time {s['to_time']} \\")
    out(f"      --max-events 1000000000 --restart")
    out("  python3 scripts/02_run_experiment.py --tag window_d"
        f"{d['first_day']}")
    out("")
    out("Whatever window you pick, every number the run reports is a number FOR THAT")
    out("WINDOW. 01 records it in data/auth_meta.json and 02 copies it into")
    out("results/experiment_<tag>.json and results/PROGRESS_<tag>.{json,md}, so the")
    out("window travels with the figures instead of being remembered separately.")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("redteam", nargs="?",
                    default="../_shared_data/lanl/redteam.txt.gz",
                    help="path to redteam.txt or redteam.txt.gz")
    ap.add_argument("--margin-hours", type=float, default=6.0,
                    help="context either side of the labelled events in the COLD "
                         "window, the one kept for comparison. See --lead-hours.")
    ap.add_argument("--lead-hours", type=float, default=24.0,
                    help="benign history BEFORE the first labelled event in the suggested "
                         "window. Stated rather than hidden, and a day by default: the "
                         "scorer compares against earlier days, so a window with no "
                         "earlier day in it ranks every logon as novel and catches "
                         "nothing. Measured: see the note the report prints.")
    ap.add_argument("--trail-hours", type=float, default=6.0,
                    help="context after the last labelled event in the suggested window")
    ap.add_argument("--top-days", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="machine-readable output only")
    a = ap.parse_args(argv)
    if a.json:
        import json as _json
        rows, _ = read_redteam(a.redteam)
        print(_json.dumps(describe(rows, a.margin_hours, a.top_days,
                                   a.lead_hours, a.trail_hours), indent=2))
        return 0
    report(a.redteam, margin_hours=a.margin_hours, top_days=a.top_days,
           lead_hours=a.lead_hours, trail_hours=a.trail_hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
