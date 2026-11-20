"""Progress, partial results and clean interruption for a run that takes days.

02_run_experiment.py is not a streaming detector: it loads the whole log and then
sweeps it once per setting (baseline, each wrong-origin draw, each point on the cost
curve, each point on the confidence sweep). So "how far through" has two parts, and
this module keeps them separate rather than inventing one number that means neither:

  * how much of data/auth.jsonl has been read into memory, in bytes; and
  * which pass, of how many, is running, and how far into it.

The partial numbers are produced by DayBudgetTracker, which shadows the budgeted
path exactly: for every day that is COMPLETE in the rows seen so far it keeps the
top --budget scores in a bounded heap, with the same tie-break as the final sort, so
the partial alert and recall figures are not estimates. They are the final figures
restricted to the days that have finished. At 100% they equal the final numbers, and
the test suite checks that.
"""
import heapq, json, os, signal, time
from pathlib import Path

_STOP = {"signum": None}


def _on_signal(signum, frame):
    _STOP["signum"] = signum


def install_signal_handlers():
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _on_signal)
        except (ValueError, OSError):
            pass


def stop_requested():
    return _STOP["signum"]


def stop_name():
    s = _STOP["signum"]
    return signal.Signals(s).name if s else None


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


def atomic_write(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


class DayBudgetTracker:
    """The budgeted path's alerts, exact, for every day that has finished.

    A day's alerts are the top `budget` rows of that day by score, among rows that
    went to the budget rather than the rules, keeping only those with score > 0. A
    bounded min-heap of size `budget` keyed on (score, -index) reproduces the final
    code's stable descending sort exactly, including ties, at constant memory.

    A day is finished when a row from a later day arrives. Rows arrive in time order,
    so at most one day is open at a time; a day that is still open contributes
    nothing, which is why the partial figures are a lower bound that only ever grows.
    """

    def __init__(self, budget_per_day):
        self.budget = max(1, int(budget_per_day))
        self._heaps = {}
        self._max_day = None
        self.days_complete = 0
        self.alerts = 0
        self.rows_seen = 0
        self.red_seen = []            # every red-team row seen (LANL has 749 labels)
        self.red_caught_keys = set()
        self.last_time = None

    def add(self, day, score, idx, row):
        """Row-dict form, kept so any caller holding dicts still works."""
        return self.add_row(day, score, idx, row["time"], row["user"], row["src"],
                            row["dst"], row["label"] == "redteam", extra=row)

    def add_row(self, day, score, idx, t, user, src, dst, is_red, extra=None):
        """Column form: the four identifying fields passed directly.

        02_run_experiment.py holds the log in typed parallel arrays rather than one
        dict per logon, so there is no row dict to hand over; building one per row
        purely to call add() would put back a slice of the per-logon cost the column
        store exists to remove. Only a RED-team row is materialised as a dict, and
        there are 749 of those in the whole LANL corpus.
        """
        self.rows_seen += 1
        self.last_time = t
        key = (t, user, src, dst)
        if is_red:
            row = {"time": t, "user": user, "src": src, "dst": dst,
                   "label": "redteam", "day": day}
            if extra:
                row.update(extra)
                row["day"] = day
            self.red_seen.append(row)
        if self._max_day is None or day > self._max_day:
            for d in [d for d in self._heaps if d < day]:
                self._close(d)
            self._max_day = day
        h = self._heaps.setdefault(day, [])
        item = (score, -idx, is_red, key)
        if len(h) < self.budget:
            heapq.heappush(h, item)
        elif item > h[0]:
            heapq.heapreplace(h, item)

    def _close(self, day):
        for score, _negidx, is_red, key in self._heaps.pop(day):
            if score > 0:
                self.alerts += 1
                if is_red:
                    self.red_caught_keys.add(key)
        self.days_complete += 1

    def snapshot(self, path_fn=None, path_gap=86400):
        """Partial figures. `path_fn` groups red-team rows into attack paths, so the
        per-path recall uses the kit's own definition rather than a second one."""
        red_keys = {(r["time"], r["user"], r["src"], r["dst"]) for r in self.red_seen}
        caught = self.red_caught_keys & red_keys
        out = {
            "days_complete": self.days_complete,
            "day_in_progress": self._max_day,
            "rows_seen": self.rows_seen,
            "last_time": self.last_time,
            "alerts_over_complete_days": self.alerts,
            "alerts_per_day_so_far": (round(self.alerts / self.days_complete, 2)
                                      if self.days_complete else None),
            "redteam_events_seen": len(red_keys),
            "redteam_events_caught": len(caught),
            "redteam_recall_over_events_so_far": (round(len(caught) / len(red_keys), 4)
                                                  if red_keys else None),
        }
        if path_fn is not None and self.red_seen:
            paths = path_fn(self.red_seen, path_gap)
            out["redteam_paths_seen"] = len(paths)
            out["redteam_paths_caught"] = sum(1 for p in paths if p["keys"] & caught)
            out["redteam_recall_over_paths_so_far"] = (
                round(out["redteam_paths_caught"] / len(paths), 4) if paths else None)
        else:
            out["redteam_paths_seen"] = 0
            out["redteam_paths_caught"] = 0
            out["redteam_recall_over_paths_so_far"] = None
        out["note"] = ("alerts and recall cover the %d day(s) that have FINISHED in the "
                       "rows read so far; day %s is still open and contributes nothing "
                       "yet. These are not estimates - they are the final figures "
                       "restricted to finished days." %
                       (self.days_complete, out["day_in_progress"]))
        return out


class ProgressReport:
    """Writes results/PROGRESS_<tag>.json and .md every `every_seconds`.

    Small, self-contained and readable on its own, because the point of it is to be
    sent to someone else mid-run for a sanity check.
    """

    def __init__(self, tag, results_dir, every_seconds=180, enabled=True):
        self.tag = tag
        self.json_path = Path(results_dir) / f"PROGRESS_{tag}.json"
        self.md_path = Path(results_dir) / f"PROGRESS_{tag}.md"
        self.every = max(5, int(every_seconds))
        self.enabled = enabled
        self.started = time.time()
        self._last = 0.0
        self.stage = "starting"
        self.config = {}
        self.log_file = {}
        self.load = {}
        self.phases_total = None
        self.phase_done = 0
        self.phase_name = None
        self.phase_rows = 0
        self.phase_rows_total = 0
        self.assumption = None
        self.settings = {}
        self.partial = None
        # The slice of the corpus the numbers below belong to. Carried here so that a
        # PROGRESS file sent to someone mid-run cannot be read as a figure for the whole
        # 58-day LANL corpus when it is a figure for a few hours of it.
        self.window = None
        self.status = "running"
        self.note = ""

    # ---- what the run tells us as it goes
    def set_stage(self, stage):
        self.stage = stage
        self.tick(force=True)

    def start_phase(self, name, rows_total):
        self.phase_name = name
        self.phase_rows = 0
        self.phase_rows_total = int(rows_total or 0)
        self.tick(force=True)

    def finish_phase(self, name, summary=None):
        self.phase_done += 1
        if summary is not None:
            self.settings[name] = summary
        self.phase_name = None
        self.tick(force=True)

    # ---- derived numbers
    def log_consumed_pct(self):
        """How much of the log this report's numbers are based on."""
        if self.stage == "loading":
            return self.load.get("pct")
        if self.phase_rows_total:
            return round(100.0 * self.phase_rows / self.phase_rows_total, 2)
        return 100.0 if self.load.get("rows") else None

    def work_pct(self):
        if not self.phases_total:
            return None
        frac = 0.0
        if self.phase_rows_total:
            frac = min(1.0, self.phase_rows / self.phase_rows_total)
        return round(100.0 * (self.phase_done + frac) / self.phases_total, 2)

    def eta_seconds(self):
        w = self.work_pct()
        el = time.time() - self.started
        if not w or w <= 0.5:
            return None
        return el * (100.0 - w) / w

    def as_dict(self):
        pct = self.log_consumed_pct()
        return {
            "format": 1,
            "kit": "T1021_anchor",
            "stage_script": "02_run_experiment.py",
            "tag": self.tag,
            "status": self.status,
            "note": self.note,
            "partial": self.status != "complete",
            "banner": self.banner(),
            "pid": os.getpid(),
            "started_at": round(self.started, 1),
            "updated_at": round(time.time(), 1),
            "elapsed_seconds": round(time.time() - self.started, 1),
            "elapsed_human": fmt_dur(time.time() - self.started),
            "est_remaining_human": fmt_dur(self.eta_seconds()),
            "stage": self.stage,
            "log_file": self.log_file,
            "log_load": self.load,
            "log_consumed_pct": pct,
            "pass_of": {"done": self.phase_done, "total": self.phases_total,
                        "current": self.phase_name,
                        "rows": self.phase_rows, "rows_total": self.phase_rows_total},
            "work_done_pct": self.work_pct(),
            "config": self.config,
            "window": self.window,
            "assumption_check_so_far": self.assumption,
            "baseline_partial": self.partial,
            "settings_complete": self.settings,
        }

    def banner(self):
        """Both numbers, because either on its own misleads.

        "42% of the log consumed" is true of the pass that is running and says nothing
        about the run as a whole; "55% of the work done" says nothing about where the
        numbers below came from. So the banner carries both, and names the pass."""
        pct = self.log_consumed_pct()
        p = "unknown" if pct is None else f"{pct:.1f}%"
        if self.status == "complete":
            return (f"COMPLETE - the whole log was consumed, all "
                    f"{self.phases_total} pass(es) finished ({self.tag})")
        lead = "PARTIAL, INTERRUPTED" if self.status == "interrupted" else "PARTIAL"
        if self.stage == "loading":
            return f"{lead} - still loading the log, {p} of it read"
        if self.phases_total:
            return (f"{lead} - pass {self.phase_done + 1} of {self.phases_total}"
                    + (f" ({self.phase_name})" if self.phase_name else "")
                    + f", {p} of the log consumed in this pass; "
                      f"{self.work_pct()}% of the work done")
        return f"{lead} - {p} of the log consumed ({self.stage})"

    # ---- the file a human reads
    def as_markdown(self):
        d = self.as_dict()
        L = []
        L.append(f"# {d['banner']}")
        L.append("")
        L.append(f"T1021 anchor experiment, tag `{self.tag}` - **this file is a "
                 f"mid-run snapshot, not a result.**")
        L.append("")
        L.append(f"- stage: **{d['stage']}**"
                 + (f", pass {d['pass_of']['done']+1} of {d['pass_of']['total']}"
                    f" (`{d['pass_of']['current']}`)" if d['pass_of']['current'] else ""))
        L.append(f"- elapsed {d['elapsed_human']}, estimated remaining "
                 f"{d['est_remaining_human']}")
        if self.load:
            L.append(f"- log: {self.load.get('rows', 0):,} rows loaded from "
                     f"`{self.log_file.get('name')}` "
                     f"({self.load.get('bytes', 0):,} of "
                     f"{self.log_file.get('bytes', 0):,} bytes, "
                     f"{self.load.get('pct')}%)")
        if d["work_done_pct"] is not None:
            L.append(f"- passes finished: {d['pass_of']['done']}/"
                     f"{d['pass_of']['total']} ({d['work_done_pct']}% of the work)")
        L.append("")
        if self.window:
            w = self.window
            L.append("## the window these numbers are for")
            L.append("")
            if w.get("applied"):
                L.append(f"- **t = {w.get('from_time')} .. {w.get('to_time')}** "
                         f"({w.get('days')} day(s) of the 58-day corpus), "
                         f"source: {w.get('source')}")
            else:
                L.append(f"- no window: the whole of `{w.get('log_file') or 'auth.jsonl'}` "
                         f"as 01 built it")
            L.append(f"- rows kept: **{w.get('rows_kept'):,}**"
                     if w.get("rows_kept") is not None else "- rows kept: unknown")
            if w.get("first_time") is not None:
                L.append(f"- actual span of the rows loaded: t = {w.get('first_time')} .. "
                         f"{w.get('last_time')} ({w.get('days_covered')} day(s))")
            if w.get("redteam_rows") is not None:
                L.append(f"- red-team logons inside it: **{w.get('redteam_rows')}**")
            L.append("")
            L.append("> Every figure in this file is a figure for that window.")
            L.append("")
        L.append("## config")
        L.append("")
        for k in sorted(self.config):
            L.append(f"- `{k}` = `{self.config[k]}`")
        L.append("")
        if self.assumption:
            a = self.assumption
            L.append("## the assumption: how often a server-like host starts a logon")
            L.append("")
            L.append(f"No attack label is used for this. It is computed over the whole "
                     f"loaded log ({a.get('total_logons', 0):,} logons, "
                     f"{a.get('days')} day(s)).")
            L.append("")
            L.append(f"- server-like hosts: **{a.get('server_like_hosts')}** of "
                     f"{a.get('hosts_total')} hosts seen")
            L.append(f"- logons they initiate: **{a.get('logons_initiated_by_them'):,}** "
                     f"of {a.get('total_logons'):,} "
                     f"(**{a.get('share_of_all_logons')}**, "
                     f"**{a.get('per_day')}/day**)")
            for cls, c in (a.get("by_class") or {}).items():
                L.append(f"  - `{cls}`: {c['hosts']} hosts, "
                         f"{c['in_server_like_class']} server-like, initiate "
                         f"{c['logons_initiated']:,} ({c['per_day']}/day), receive "
                         f"{c['logons_received']:,}")
            L.append("")
        if self.partial:
            p = self.partial
            L.append("## baseline (budget only), so far")
            L.append("")
            L.append(f"- days finished: **{p['days_complete']}** "
                     f"(day {p['day_in_progress']} still open, contributes nothing)")
            L.append(f"- alerts over finished days: **{p['alerts_over_complete_days']:,}**"
                     f" = **{p['alerts_per_day_so_far']}/day** at the budget")
            L.append(f"- red-team events seen: **{p['redteam_events_seen']}**, caught "
                     f"**{p['redteam_events_caught']}** "
                     f"(recall {p['redteam_recall_over_events_so_far']})")
            L.append(f"- red-team paths seen: **{p['redteam_paths_seen']}**, caught "
                     f"**{p['redteam_paths_caught']}** "
                     f"(recall {p['redteam_recall_over_paths_so_far']})")
            L.append("")
            L.append(f"> {p['note']}")
            L.append("")
        main_settings = {k: v for k, v in self.settings.items()
                         if not k.startswith("sweep_")}
        n_sweep = len(self.settings) - len(main_settings)
        if main_settings:
            L.append("## settings that have finished")
            L.append("")
            L.append("| setting | alerts/day | event recall | path recall | anchored logons |")
            L.append("|---|---|---|---|---|")
            for name, s in main_settings.items():
                L.append(f"| `{name}` | {s.get('alerts_per_day')} | "
                         f"{s.get('redteam_recall_over_events')} | "
                         f"{s.get('redteam_recall_over_paths')} | "
                         f"{s.get('anchored_paths')} |")
            L.append("")
            if n_sweep:
                L.append(f"{n_sweep} confidence-sweep pass(es) have also finished; they "
                         f"are in the JSON beside this file, not here, to keep this "
                         f"readable.")
                L.append("")
        L.append("---")
        L.append(f"written {time.strftime('%Y-%m-%d %H:%M:%S')} by "
                 f"02_run_experiment.py (pid {os.getpid()}); "
                 f"`scripts/status.py` prints the same thing live.")
        L.append("")
        return "\n".join(L)

    def beat(self, every=30):
        """True on a stdout cadence, which is shorter than the file cadence: a human
        watching `tail -f` needs a line every half minute, a file that is sent to
        someone does not need rewriting that often."""
        now = time.time()
        if now - getattr(self, "_lastbeat", 0.0) >= max(1, every):
            self._lastbeat = now
            return True
        return False

    def tick(self, force=False):
        if not self.enabled:
            return False
        now = time.time()
        if not force and now - self._last < self.every:
            return False
        self._last = now
        atomic_write(self.json_path, json.dumps(self.as_dict(), indent=2, default=str))
        atomic_write(self.md_path, self.as_markdown())
        return True

    def finish(self, status="complete", note=""):
        self.status = status
        self.note = note
        self.tick(force=True)
