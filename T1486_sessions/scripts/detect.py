"""The detector, in the design's three stages, and the baseline it is compared with.

Data processing  operations counted in one-second windows, as in Aranyi et al.,
                 but keyed by CLIENT SESSION rather than by server. The baseline
                 keys by server, which is what happens when every client's activity
                 is mixed into one stream.

Threat detection A window is flagged when it shows the repeated read / write /
                 rename pattern ERW-Radar keys on. A flag is confirmed only if a
                 sample of what was written classifies as encrypted rather than
                 compressed, as in Casino et al. Anchor A3 removes the need for
                 that confirmation, because stage 3 has already raised suspicion.
"""
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The shared anchor module sits beside the kits; appended, not inserted, so this kit's
# own scripts/common.py keeps shadowing the Implementation/common package.
sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from common import chi_square_uniform, shannon_entropy, log
from anchors import relaxation

# The entropy an encrypted window must still clear however strong the anchor. A
# proportional relaxation has to relax TOWARDS something, and 6.0 bits is roughly where
# compressed data already sits: below it the content test would be agreeing with
# anything, so the anchor would be buying a bypass by the back door.
ENT_FLOOR = 6.0

SAMPLE_BYTES = 4096

# The server-keyed warning is worth saying once a process, not once a window.
_WARNED_SERVER_ANCHORS = False


def windows(ops, key, period=1.0):
    """Bucket operations into `period`-second windows under the chosen key."""
    out = defaultdict(list)
    for o in ops:
        out[(key(o), int(o.t // period))].append(o)
    return out


def behaviour_flag(ws, min_writes=3, min_renames=1, min_reads=2):
    """ERW-Radar's signal: the read / encrypted-write / rename loop, repeated."""
    counts = defaultdict(int)
    for o in ws:
        counts[o.op] += 1
    return (counts["write"] >= min_writes and counts["rename"] >= min_renames
            and counts["read"] >= min_reads)


def content_says_encrypted(ws, chi_max=350.0, ent_min=7.5, agg="mean"):
    """Casino et al.'s test: high entropy alone is not enough, because compressed
    data is high entropy too. Encrypted bytes sit much closer to uniform, so the
    chi-square statistic against uniform is small."""
    blobs = [o.content[:SAMPLE_BYTES] for o in ws if o.op == "write" and o.content]
    if not blobs:
        return False, None, None
    ents, chis = [], []
    for b in blobs:
        if len(b) < 1024:
            continue
        ents.append(shannon_entropy(b))
        chis.append(chi_square_uniform(b))
    if not ents:
        return False, None, None
    if agg == "max":
        # The most encrypted-looking sample in the window decides. This is the fair
        # version of the server-keyed baseline: mixing in benign writes can no longer
        # drag the average down and hide a single encrypted burst.
        best = min(range(len(ents)), key=lambda i: chis[i])
        ent, chi = ents[best], chis[best]
    else:
        ent = sum(ents) / len(ents)
        chi = sum(chis) / len(chis)
    return (ent >= ent_min and chi <= chi_max), ent, chi


def run(ops, keyed_by="session", anchor_store=None, anchor_min_confidence=0.0,
        period=1.0, chi_max=350.0, ent_min=7.5, content_agg="mean", link="A3"):
    """Return per-key detections, the per-window decisions behind them, and the alerts.

    keyed_by "session" is the design; "server" is the baseline in which every
    client's activity is mixed into one stream.

    `anchor_store` is a common/anchors.py AnchorStore, or None. An anchor is looked
    up per window, on the subject kind "session" and at the window's own time, so it
    can no longer apply to the whole log including traffic that predates it.
    `anchor_min_confidence` is the floor a record must clear to be matched at all, and
    is the parameter the sweep varies.

    What a matched anchor then BUYS is graded, not all-or-nothing. The relaxation from
    common/anchors.py - the A3 policy's max_drop times the anchor's confidence decayed
    to this window's own time - scales the content test proportionally:

        chi_max -> chi_max * (1 + r)        the uniformity test is loosened
        ent_min -> ent_min - r * (ent_min - ENT_FLOOR)

    so a strong, fresh anchor lets a window through on weaker content evidence while a
    weak or aged one barely moves the bar. Only r >= 0.99 is a full bypass, i.e. the
    behavioural flag alone. The earlier code treated any anchor at or above the
    threshold as a complete bypass of the content confirmation, which made the cost of
    a wrong anchor a step rather than a curve and threw away the confidence it was
    carrying.
    """
    global _WARNED_SERVER_ANCHORS
    anchors_applicable = True
    if keyed_by == "session":
        key = lambda o: o.session
    else:
        key = lambda o: "SERVER"
        if anchor_store is not None and len(anchor_store):
            # The baseline's key is the constant "SERVER", not a session id, so a
            # session anchor can never match it. Said out loud rather than left to
            # look like a result: an anchored baseline run IS the unanchored one.
            if not _WARNED_SERVER_ANCHORS:
                log("warning: anchors were supplied for a server-keyed run. The key is "
                    "the constant 'SERVER', so no session anchor can apply and these "
                    "runs are identical to the unanchored baseline.")
                _WARNED_SERVER_ANCHORS = True
            anchors_applicable = False
            anchor_store = None
    ws = windows(ops, key, period)

    rows = []
    alerts, per_key = [], defaultdict(lambda: {"windows": 0, "flagged": 0, "confirmed": 0,
                                               "alert": False, "first_alert_t": None})
    for (k, second), group in sorted(ws.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        per_key[k]["windows"] += 1
        counts = defaultdict(int)
        for o in group:
            counts[o.op] += 1
        flagged = behaviour_flag(group)
        # One row per window, flagged or not: the thresholds, the ROC and the
        # alerts-per-hour figure are all recoverable from these afterwards, and
        # throwing them away is what made every earlier rate a per-run boolean.
        row = {"keyed_by": keyed_by, "key": k, "window_start": second * period,
               "n_ops": len(group), "creates": counts["create"], "reads": counts["read"],
               "writes": counts["write"], "renames": counts["rename"],
               "deletes": counts["delete"], "behaviour_flag": bool(flagged),
               "entropy": None, "chi_square": None, "confirmed": False,
               "anchored": False, "anchor_id": None, "alert": False,
               # the graded relaxation applied in this window, and the thresholds it
               # produced. 0.0 on every window where no anchor was consulted, which
               # includes every window the behavioural flag did not fire in.
               "relaxation": 0.0, "anchor_full_bypass": False,
               "eff_chi_max": chi_max, "eff_ent_min": ent_min}
        rows.append(row)
        if not flagged:
            continue
        per_key[k]["flagged"] += 1
        anchor = None
        if anchor_store is not None:
            # `link=` is not optional. Without it a stray A4 record in the file would
            # match on its subject and then be relaxed under A4's policy - a 0.50
            # max_drop applied by a stage that A4 is not addressed to.
            anchor = anchor_store.match("session", k, at=second * period,
                                        min_confidence=anchor_min_confidence,
                                        link=link)
        # How much this anchor is worth AT THIS WINDOW'S TIME: the link's licensed
        # max_drop times its confidence, decayed. Zero when there is no anchor, which
        # is the unanchored fallback every design in the paper promises.
        r = 0.0 if anchor is None else relaxation(anchor, at=second * period)
        anchored = anchor is not None and r > 0.0
        full_bypass = r >= 0.99
        eff_chi_max = chi_max * (1.0 + r)
        eff_ent_min = ent_min - r * (ent_min - ENT_FLOOR)
        confirmed, ent, chi = content_says_encrypted(group, eff_chi_max, eff_ent_min,
                                                     content_agg)
        if confirmed:
            per_key[k]["confirmed"] += 1
        row["entropy"] = None if ent is None else round(ent, 3)
        row["chi_square"] = None if chi is None else round(chi, 1)
        row["confirmed"] = bool(confirmed)
        row["anchored"] = bool(anchored)
        row["anchor_id"] = None if anchor is None else anchor.anchor_id
        row["relaxation"] = round(r, 6)
        row["anchor_full_bypass"] = bool(full_bypass)
        row["eff_chi_max"] = round(eff_chi_max, 3)
        row["eff_ent_min"] = round(eff_ent_min, 4)
        # Anchor A3 lowers the bar in proportion to what it is worth. Only a relaxation
        # of 0.99 or more - a fresh anchor at the confidence cap - drops the content
        # confirmation entirely; anything less has to be met part of the way by the
        # content itself, through the loosened thresholds above.
        if full_bypass or confirmed:
            row["alert"] = True
            if not per_key[k]["alert"]:
                per_key[k]["alert"] = True
                per_key[k]["first_alert_t"] = second * period
            alerts.append({"key": k, "t": second * period, "anchored": anchored,
                           "relaxation": round(r, 6),
                           "anchor_full_bypass": bool(full_bypass),
                           "confirmed_by_content": confirmed,
                           "entropy": None if ent is None else round(ent, 3),
                           "chi_square": None if chi is None else round(chi, 1),
                           "anchor_id": None if anchor is None else anchor.anchor_id,
                           "anchor_link": None if anchor is None else anchor.link,
                           "anchor_technique": None if anchor is None else anchor.technique,
                           "anchor_confidence": None if anchor is None else anchor.confidence})
    return {"per_key": {k: dict(v) for k, v in per_key.items()}, "alerts": alerts,
            "windows": rows, "anchors_applicable": anchors_applicable}
