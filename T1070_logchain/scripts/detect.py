"""The three detectors compared.

  clearing_rules   the maintained rules of the second track: an event saying the log
                   was cleared (Sigma d99b79d2 / 100ef69e, event 1102 / 104)
  gap_check        are the local sequence numbers contiguous?
  content_compare  the design: every record the collector holds must still be present
                   locally, matched BY CONTENT rather than by sequence number
"""
import math
import statistics
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import content_hash

CLEAR_EVENT_IDS = {1102, 104}

# How many times the median second's rate a second must carry before the collector
# counts as busy. Named here rather than buried in the function, because it is the
# number that decides whether "dropped under load" means anything at all.
LOAD_MULTIPLIER = 3.0

# How many surplus records the collector must hold before the detector CALLS it removal.
# One, which is the behaviour this kit has always had: a single record the collector
# holds and the local log does not is a removal. It is named here, and made a parameter,
# because it is the threshold link A4's policy is about - "lower the threshold for
# treating a missing record as removal" - and a threshold hard-coded at its minimum
# cannot be lowered, so the policy was unimplementable while it was a literal 1. A
# deployment that tolerates collector-side noise would run it higher.
REMOVAL_MIN_SURPLUS = 1


def _record_stub(r, h=None):
    return {"content_hash": h if h is not None else content_hash(r),
            "seq": r.get("seq"), "event_id": r.get("event_id")}


def clearing_rules(local, collector, load, **kw):
    hits = [r for r in local if r.get("event_id") in CLEAR_EVENT_IDS]
    return {
        "alert": bool(hits),
        "flagged": len(hits),
        "detail": f"{len(hits)} log-cleared event(s)" if hits else "no log-cleared event",
        "missing_content": [],
        # The records it flags are the clearing events themselves, not the records that
        # went missing: it names the symptom, never the loss.
        "flagged_records": [_record_stub(r) for r in hits],
    }


def gap_check(local, collector, load, **kw):
    pairs = [(i, r.get("seq")) for i, r in enumerate(local) if isinstance(r.get("seq"), int)]
    gaps = []
    for (ia, a), (ib, b) in zip(pairs, pairs[1:]):
        if b != a + 1:
            gaps.append(local[ib])
    return {
        "alert": bool(gaps),
        "flagged": len(gaps),
        "detail": f"{len(gaps)} break(s) in the sequence numbers",
        "missing_content": [],
        "flagged_records": [_record_stub(r) for r in gaps],
    }


def content_compare(local, collector, load, load_threshold=None,
                    load_multiplier=LOAD_MULTIPLIER, anchor_relaxation=0.0,
                    removal_min_surplus=REMOVAL_MIN_SURPLUS):
    """Compare by content in both directions, as MULTISETS.

    in collector, not local  -> the record was removed after it was protected: an attack
    in local, not collector  -> the collector never got it: a drop, reported with the
                                load at that moment so an analyst can tell the two apart

    Counting multiplicities, not set membership, is the whole correctness of this
    detector. Two records that differ only in sequence number have the same content
    hash by design - the sequence number is excluded precisely because the tool
    Fox-IT analysed renumbers records. Under set membership, removing one of a
    colliding pair leaves the hash still present locally and the removal is
    invisible; under multiset comparison the collector's two copies against the
    local one are a surplus of exactly one, which is the loss.
    """
    local_counts = Counter()
    for r in local:
        local_counts[content_hash(r)] += 1

    coll_pairs = [(content_hash(r), r) for r in collector]
    coll_counts = Counter(h for h, _ in coll_pairs)

    # Surplus copies held by the collector, in the order the collector holds them.
    seen = Counter()
    removed = []
    for h, r in coll_pairs:
        seen[h] += 1
        if seen[h] > local_counts[h]:
            removed.append((h, r))

    seen = Counter()
    not_forwarded = []
    for r in local:
        h = content_hash(r)
        seen[h] += 1
        if seen[h] > coll_counts[h]:
            not_forwarded.append(r)

    # ---- what an A4 anchor is allowed to change here.
    # A4's policy licenses one thing: "lower the threshold for treating a missing record
    # as removal". content_compare has no score to threshold - a record the collector
    # holds in greater multiplicity than the local log either is missing or is not - so
    # the threshold it DOES have is the load threshold, and that is the one the policy
    # is about. A missing record is excused as collector load when the second it was
    # written in was busy; "busy" is a multiple of the median second's rate. Lowering
    # that multiple means fewer seconds count as busy, so fewer losses are explained
    # away and more of them are left as the unexplained residue an analyst must treat as
    # removal. That is exactly the sentence in the policy, and it is the only threshold
    # in this detector that the sentence can refer to.
    #
    # The DIRECTION matters and is easy to get backwards. A missing record is excused as
    # load when the second it was written in was busy, so excusing fewer of them means
    # calling fewer seconds busy, which means demanding a HIGHER rate before a second
    # counts as busy. The multiplier therefore goes UP with the relaxation, not down:
    # lowering it would excuse more losses, which is the opposite of the policy.
    #
    # relaxation is A4's max_drop (0.50) times the anchor's decayed confidence. Zero when
    # no anchor matched, which is the unanchored fallback: both expressions below are
    # then exact no-ops (x * 1.0 and max(1, ceil(m * 1.0))) and every number this
    # detector reports is bit-identical to the pre-anchor version.
    effective_multiplier = load_multiplier * (1.0 + anchor_relaxation)
    # And the threshold the policy names in so many words. At the default of 1 this is
    # already at its floor and an anchor cannot move it, which is worth saying plainly:
    # the relaxation is only observable here in a deployment that runs the threshold
    # above 1.
    effective_min_surplus = max(1, math.ceil(removal_min_surplus * (1.0 - anchor_relaxation)))

    # ---- how busy is "busy"?
    # The old threshold was the 90th percentile over SECONDS, unweighted by volume.
    # Most seconds in this log are quiet, so the 90th percentile of the per-second
    # counts sat barely above the quiet rate and nearly every second counted as busy,
    # which made not_forwarded_under_load equal not_forwarded by construction. The
    # threshold is now a multiple of the median second's rate, so "under load" means
    # a rate several times the run's normal one, and the median, the multiplier and
    # the distribution are all reported so the number can be read.
    dist = {}
    median_rate = None
    if load:
        vals = sorted(load.values())
        median_rate = statistics.median(vals)
        thr = (load_threshold if load_threshold is not None
               else max(1.0, effective_multiplier * median_rate))

        def pct(q):
            return vals[min(len(vals) - 1, int(q * (len(vals) - 1)))]
        dist = {"seconds": len(vals), "min": vals[0], "p50": pct(0.5), "p75": pct(0.75),
                "p90": pct(0.9), "p99": pct(0.99), "max": vals[-1],
                "mean": round(statistics.mean(vals), 3)}
    else:
        thr = 0.0
    busy = {s for s, n in load.items() if n >= thr}
    dist["busy_seconds"] = len(busy)
    dist["busy_second_share"] = (round(len(busy) / len(load), 4) if load else None)

    under_load = sum(1 for r in not_forwarded if r.get("_load_second") in busy)
    load_unknown = sum(1 for r in not_forwarded if r.get("_load_second") is None)
    # The count that actually carries information: a record the collector never
    # received while it was NOT busy cannot be explained by load, so it is the
    # residue an analyst has to account for some other way. Reported alongside the
    # busy count rather than left to be subtracted, because the whole point of the
    # threshold is the difference between the two.
    not_under_load = sum(1 for r in not_forwarded
                         if r.get("_load_second") is not None
                         and r.get("_load_second") not in busy)

    return {
        # The records NAMED are unchanged - the surplus-count test is exact and an anchor
        # cannot add or remove one. What the threshold decides is whether that many
        # surplus records is called removal.
        "alert": len(removed) >= effective_min_surplus,
        "flagged": len(removed),
        "detail": (f"{len(removed)} record(s) held by the collector are missing locally; "
                   f"{len(not_forwarded)} local record(s) never reached the collector, "
                   f"{under_load} of them while the collector was busy and "
                   f"{not_under_load} while it was not "
                   f"(busy is >= {round(thr, 2)} records/s, "
                   f"{round(effective_multiplier, 3)}x the median of {median_rate}"
                   + (f"; A4 anchor RAISED the multiplier from "
                      f"{round(load_multiplier, 2)} at relaxation "
                      f"{round(anchor_relaxation, 4)}, so fewer losses are excused as "
                      f"load; removal is called at {effective_min_surplus} surplus "
                      f"record(s) rather than {removal_min_surplus}"
                      if anchor_relaxation > 0 else "")
                   + ")"),
        "missing_content": [h for h, _ in removed],
        "flagged_records": [_record_stub(r, h) for h, r in removed],
        "not_forwarded": len(not_forwarded),
        "not_forwarded_under_load": under_load,
        "not_forwarded_not_under_load": not_under_load,
        "not_forwarded_load_unknown": load_unknown,
        # The hashes themselves, not just how many: a drop cannot be investigated
        # from a count, and these are the records the design says it cannot protect.
        "not_forwarded_content": [content_hash(r) for r in not_forwarded],
        "load_threshold_records_per_second": round(thr, 3),
        "load_median_records_per_second": median_rate,
        "load_threshold_multiplier": load_multiplier,
        "load_threshold_multiplier_effective": round(effective_multiplier, 6),
        "anchor_relaxation": round(anchor_relaxation, 6),
        "removal_min_surplus": removal_min_surplus,
        "removal_min_surplus_effective": effective_min_surplus,
        "load_distribution": dist,
        # How many content hashes the local log holds more than once. Reported so the
        # condition the multiset comparison exists for is visible in the output,
        # rather than assumed: with no colliding groups the comparison is untested.
        "n_colliding_content_groups": sum(1 for c in local_counts.values() if c > 1),
        "content_collision_surplus": sum(c - 1 for c in local_counts.values() if c > 1),
    }


# Only one of the three can say WHICH records went missing; the other two can only
# raise a flag. Recall is therefore reported for that one alone.
RECOVERS_RECORDS = {"content_compare"}

DETECTORS = {
    "clearing_rules": clearing_rules,
    "gap_check": gap_check,
    "content_compare": content_compare,
}


# ---------------------------------------------------------------------------
#  The comparison this kit does NOT use, kept so the difference can be measured
# ---------------------------------------------------------------------------
def content_compare_setwise(local, collector, load, **kw):
    """`content_compare` with SET membership instead of multiset counting.

    This is not a detector in the experiment. It is the earlier, wrong version of the
    same idea, kept so the cost of the mistake can be reported as a number rather than
    asserted in a comment. A record removed from a colliding pair leaves its content
    hash still present locally - the surviving copy carries it - so set subtraction
    sees nothing missing. The multiset version sees the collector holding two copies
    against the local log's one and reports a surplus of exactly one.

    On real Windows Security records this is not a corner case: roughly a quarter of
    the corpus's records are identical in content to another record written in the same
    millisecond, so any removal landing on one of them is invisible to this comparison.
    """
    local_set = {content_hash(r) for r in local}
    removed = [(h, r) for h, r in ((content_hash(r), r) for r in collector)
               if h not in local_set]
    # de-duplicate: a set comparison cannot name the same missing content twice
    seen, uniq = set(), []
    for h, r in removed:
        if h not in seen:
            seen.add(h)
            uniq.append((h, r))
    return {
        "alert": bool(uniq),
        "flagged": len(uniq),
        "detail": (f"{len(uniq)} distinct content hash(es) held by the collector are "
                   f"absent from the local log, by set membership"),
        "missing_content": [h for h, _ in uniq],
        "flagged_records": [_record_stub(r, h) for h, r in uniq],
    }
