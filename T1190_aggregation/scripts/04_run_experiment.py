"""The experiment: does mean-or-max aggregation beat max-only at the same false-positive rate?

One chunk scorer is trained per seed and held fixed; only the file-level
aggregation rule changes. That isolates the variable Kwon 2026 leaves open
("file-level detection performance depends on how chunk scores are aggregated,
and the maximum-score aggregation strategy used in this study may not generalize").

Rules compared, all on the same chunk probabilities p_j:
  mean         alert iff mean_j p_j >= t_mean          (Wang et al. 2025, average pooling)
  max          alert iff max_j  p_j >= t_max           (Kwon 2026, file score = max chunk)
  mean_or_max  alert iff mean >= t_mean OR max >= t_max, with t_max >= t_mean   (this design)

Thresholds are chosen on the calibration split, whose sources are disjoint from
both training and test, under a false-positive budget. Nothing is tuned on test.
"""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The shared anchor module sits beside the kits. APPENDED, not inserted, so that this
# kit's own scripts/common.py keeps shadowing the repository's common/ package.
sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from common import (DATA, RESULTS, append_jsonl, log, read_jsonl, read_text, save_json,
                    truncate_jsonl)
import anchors
import bridge

# scikit-learn is what the design specifies. It is not installable on every machine -
# this one's egress proxy refuses PyPI - so its absence is detected here rather than
# crashing on import, and --scorer numpy falls back to a numpy-only stand-in whose
# numbers are labelled as not a measurement everywhere they are written.
try:
    import sklearn
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    SKLEARN_IMPORT_ERROR = None
except ImportError as _e:          # pragma: no cover - depends on the machine
    sklearn = None
    HashingVectorizer = LogisticRegression = None
    SKLEARN_IMPORT_ERROR = str(_e)
import numpy_scorer

EPS = 1e-12

# Diluted files are reported by the dilution actually achieved, not the ratio asked
# for: 03_make_dilution_set.py enforces a 60-byte floor on the injected snippet, so a
# file requested at 0.001 of a small host can really carry several times that.
ACHIEVED_RATIO_EDGES = [0.0, 0.002, 0.01, 0.05, 1.0]
ACHIEVED_RATIO_LABELS = [
    f"[{ACHIEVED_RATIO_EDGES[i]:g}, {ACHIEVED_RATIO_EDGES[i + 1]:g}"
    + ("]" if i == len(ACHIEVED_RATIO_EDGES) - 2 else ")")
    for i in range(len(ACHIEVED_RATIO_EDGES) - 1)
]


def achieved_ratio_of(row):
    """Injected bytes as a fraction of the host file; NaN when not a diluted file.

    03_make_dilution_set.py records this, because the 60-byte floor it puts on the
    snippet makes the requested ratio wrong for small hosts. The fallbacks are only
    for dilution sets written before that field existed; dividing by the finished
    file's size instead of the host's differs by the injected bytes themselves.
    """
    if row.get("achieved_ratio") is not None:
        return float(row["achieved_ratio"])
    inj = row.get("injected_bytes")
    size = row.get("host_size") or row.get("size")
    if inj is None or not size:
        return float("nan")
    return float(inj) / float(size)


def ratio_bins(achieved):
    """Bin index per file for ACHIEVED_RATIO_EDGES; -1 where the ratio is unknown."""
    a = np.asarray(achieved, dtype=float)
    out = np.full(a.shape, -1, dtype=int)
    ok = np.isfinite(a)
    last = len(ACHIEVED_RATIO_EDGES) - 2
    for i in range(last + 1):
        lo, hi = ACHIEVED_RATIO_EDGES[i], ACHIEVED_RATIO_EDGES[i + 1]
        m = ok & (a >= lo) & ((a < hi) if i < last else (a <= hi))
        out[m] = i
    return out


# ----------------------------------------------------------------- ranking metrics
def roc_auc(y_true, score):
    """ROC AUC in its Mann-Whitney form, ties credited one half. numpy only."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    plain = np.arange(1, len(s) + 1, dtype=float)      # 1-based ranks
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = plain[i:j + 1].mean()  # average rank within a tie group
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(y_true, score):
    """Average precision (PR AUC), sum of (R_n - R_n-1) * P_n over thresholds. numpy only."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return None
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    last = np.append(np.diff(ss) != 0, True)           # one point per distinct score
    tp, fp = tp[last], fp[last]
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / npos
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


AUC_SUBSETS = ("all_positives", "plain_positives", "diluted_positives",
               "diluted_positives_with_hosts")


def auc_block(mean_s, max_s, is_pos, is_dil, is_dil_host=None):
    """ROC AUC and average precision for both file scores, threshold-free.

    Reported over every positive, over the plain shells only, and over the diluted
    positives only, because a rule can rank plain shells perfectly and still rank the
    diluted ones below benign code.

    THE DILUTED SUBSET EXCLUDES THE HOST FILES THE DILUTED POSITIVES WERE BUILT FROM.
    Each diluted positive IS one of the benign hosts with a shell injected into it; at
    the achieved ratios of 0.1-2% the two files are 98-99.9% byte-identical and sit on
    opposite sides of the label. Keeping both bounds the diluted-only AUC near chance
    BY CONSTRUCTION, whatever the detector does - so the old "mean 0.5912" was partly
    measuring the impossibility of separating a file from itself.

    `diluted_positives_with_hosts` keeps the old definition so the change is visible
    and the previously published figure stays traceable. The two subsets therefore do
    NOT share a negative set; `n_negatives` and `n_hosts_excluded` are reported per
    subset so no one compares them as if they did.
    """
    if is_dil_host is None:
        is_dil_host = np.zeros(len(is_pos), dtype=bool)
    dil_sub = (~is_pos) | (is_pos & is_dil)
    subsets = {
        "all_positives": np.ones(len(is_pos), dtype=bool),
        "plain_positives": (~is_pos) | (is_pos & ~is_dil),
        "diluted_positives": dil_sub & ~is_dil_host,
        "diluted_positives_with_hosts": dil_sub,
    }
    out = {}
    for name, sub_m in subsets.items():
        y = is_pos[sub_m]
        out[name] = dict(
            n_positives=int(y.sum()), n_negatives=int((~y).sum()),
            n_hosts_excluded=(int((is_dil_host & ((~is_pos) | (is_pos & is_dil))).sum())
                              if name == "diluted_positives" else 0),
            mean_score=dict(roc_auc=roc_auc(y, mean_s[sub_m]),
                            average_precision=average_precision(y, mean_s[sub_m])),
            max_score=dict(roc_auc=roc_auc(y, max_s[sub_m]),
                           average_precision=average_precision(y, max_s[sub_m])),
        )
    return out


# ----------------------------------------------------------------- chunking
MISSING_FILES = []          # a quarantined or deleted file must not kill a long run


def file_chunks(path, size, stride, max_chunks):
    try:
        b = read_text(path).encode("utf-8", "replace")
    except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
        # Antivirus quarantine between indexing and scoring is a real scenario, and so
        # is an index built under a different mount point. Record it and carry on: a
        # three-hour run must not die on one file, but the count MUST reach the results
        # so nobody reports a corpus size the run did not actually use.
        MISSING_FILES.append({"path": str(path), "error": type(e).__name__})
        return [], False
    if len(b) <= size:
        out = [b]
    else:
        out, i = [], 0
        while i < len(b):
            out.append(b[i:i + size])
            if i + size >= len(b):
                break
            i += stride
    truncated = len(out) > max_chunks
    return [c.decode("latin-1") for c in out[:max_chunks]], truncated


def build_chunks(rows, size, stride, max_chunks):
    texts, owner, trunc = [], [], 0
    for idx, r in enumerate(rows):
        cs, t = file_chunks(r["path"], size, stride, max_chunks)
        trunc += int(t)
        texts.extend(cs)
        owner.extend([idx] * len(cs))
    return texts, np.asarray(owner, dtype=np.int64), trunc


# ----------------------------------------------------------------- scoring
def per_file_scores(model, vec, rows, size, stride, max_chunks, batch=2000):
    """Return (mean_score, max_score, n_chunks) per file, streamed to bound memory."""
    means = np.zeros(len(rows)); maxes = np.zeros(len(rows)); counts = np.zeros(len(rows), dtype=int)
    buf_texts, buf_owner = [], []

    def flush():
        if not buf_texts:
            return
        X = vec.transform(buf_texts)
        p = model.predict_proba(X)[:, 1]
        for o, pv in zip(buf_owner, p):
            means[o] += pv
            counts[o] += 1
            if pv > maxes[o]:
                maxes[o] = pv
        buf_texts.clear(); buf_owner.clear()

    for idx, r in enumerate(rows):
        cs, _ = file_chunks(r["path"], size, stride, max_chunks)
        buf_texts.extend(cs); buf_owner.extend([idx] * len(cs))
        if len(buf_texts) >= batch:
            flush()
    flush()
    counts = np.maximum(counts, 1)
    return means / counts, maxes, counts


def score_rows(rows, mean_s, max_s, counts, seed, split):
    """One row per file per seed: the file scores plus everything needed to re-slice them.

    This is the only record of the scores themselves. With it, ROC and PR curves, any
    other false-positive budget and any per-language or per-source breakdown can be
    computed afterwards without retraining, which is the expensive part of the run.
    """
    out = []
    for r, m, x, n in zip(rows, mean_s, max_s, counts):
        ach = achieved_ratio_of(r)
        out.append(dict(
            seed=int(seed), split=split, path=r["path"], source_id=r.get("source_id"),
            group=r.get("group"), label=int(r["label"]), kind=r.get("kind", "plain"),
            inject_ratio=r.get("inject_ratio"),
            achieved_ratio=None if not np.isfinite(ach) else float(ach),
            injected_bytes=r.get("injected_bytes"), lang=r.get("lang"), size=r.get("size"),
            mean_score=float(m), max_score=float(x), n_chunks=int(n),
        ))
    return out


# ----------------------------------------------------------------- thresholds
def fpr_recall_1d(bad_scores, benign_scores, t):
    return (benign_scores >= t).mean(), (bad_scores >= t).mean()


def pick_1d(benign, budget, grid=400):
    """Smallest threshold whose calibration FPR is within budget (so recall is highest)."""
    cands = np.unique(np.concatenate([
        np.quantile(benign, np.linspace(0, 1, grid)),
        [benign.max() + 1e-9],
    ]))
    bs = np.sort(benign)
    n = len(bs)
    best = None
    for t in cands:
        fp = n - np.searchsorted(bs, t, side="left")
        if fp / n <= budget + EPS:
            best = float(t)
            break
    if best is None:
        best = float(benign.max() + 1e-9)
    return best


def pick_2d(benign_mean, benign_max, pos_mean, pos_max, pos_is_diluted, budget,
            grid=400, extra_max_candidates=()):
    """Grid search over (t_mean, t_max) with t_max >= t_mean, FPR within budget.

    Returns (t_mean, t_max, degenerate). Both candidate lists end in a sentinel just
    above the largest benign score - the threshold that alerts on no benign file at all
    - because without it on the mean side every candidate pair alerted on at least one
    benign file, so at a tight budget no pair qualified and the rule was reported at
    zero recall.

    THE GRID WAS 80 AND pick_1d's IS 400, WHICH MADE THE COMPARISON UNFAIR. With 80
    quantiles the spacing on the benign-max axis is 1/79 = 1.27%, so between the
    98.73rd percentile and the maximum there was no candidate at all: at budgets 0.01
    and 0.005 the only admissible t_max was at or above the benign maximum, mean_or_max
    spent none of its budget (calibration FPR exactly 0.0, test FPR 0.0005 against
    max-only's 0.0039), and the two rules were compared ~8x apart in false-positive
    spend. The resulting "mean_or_max is 0.153 worse than max" was substantially an
    artefact of the candidate set, not a property of the rule.

    Two changes remove it. The grid now matches pick_1d's resolution, and the caller
    passes the max-only threshold for this budget in `extra_max_candidates`, so the
    reduction of mean_or_max to max-only (t_mean above every benign mean, t_max at
    max-only's own threshold) is always exactly representable. On a common candidate
    set mean_or_max can no longer score BELOW max-only at the same budget, which is the
    arithmetic the earlier result violated.

    The inner loop is vectorised over the t_max candidates, so 400x400 costs about what
    80x80 did unvectorised.
    """
    tm_c = np.unique(np.concatenate([np.quantile(benign_mean, np.linspace(0, 1, grid)),
                                     [benign_mean.max() + 1e-9]]))
    tx_c = np.unique(np.concatenate([
        np.quantile(benign_max, np.linspace(0, 1, grid)),
        np.asarray(list(extra_max_candidates), dtype=float).ravel(),
        [benign_max.max() + 1e-9]]))
    nb, npos = len(benign_mean), len(pos_mean)
    n_dil = int(pos_is_diluted.sum())
    best = None
    for tm in tm_c:
        mb = benign_mean >= tm
        nb_caught = int(mb.sum())
        rest_b = np.sort(benign_max[~mb])
        mp = pos_mean >= tm
        np_caught = int(mp.sum())
        rest_p = np.sort(pos_max[~mp])
        dil_by_mean = int((mp & pos_is_diluted).sum())
        rest_dil = np.sort(pos_max[(~mp) & pos_is_diluted])

        tx = tx_c[tx_c >= tm]
        if not len(tx):
            continue
        fp = nb_caught + (len(rest_b) - np.searchsorted(rest_b, tx, side="left"))
        adm = (fp / nb) <= budget + EPS
        if not adm.any():
            continue
        tx, fp = tx[adm], fp[adm]
        tp = np_caught + (len(rest_p) - np.searchsorted(rest_p, tx, side="left"))
        dil = dil_by_mean + (len(rest_dil) - np.searchsorted(rest_dil, tx, side="left"))
        rec = tp / npos
        dil_rec = (dil / n_dil) if n_dil else np.zeros(len(tx), dtype=float)
        # lexicographic, best first: recall, then diluted recall, then the lower FPR.
        # np.lexsort takes the PRIMARY key last.
        j = int(np.lexsort((fp / nb, -dil_rec, -rec))[0])
        key = (float(rec[j]), float(dil_rec[j]), -float(fp[j]) / nb)
        if best is None or key > best[0]:
            best = (key, float(tm), float(tx[j]))
    if best is None:
        return float(benign_mean.max() + 1e-9), float(benign_max.max() + 1e-9), True
    return best[1], best[2], False


# ----------------------------------------------------------------- evaluation
def evaluate(rule, thresholds, mean_s, max_s, is_pos, is_dil, requested, achieved):
    tm, tx = thresholds
    if rule == "mean":
        alert = mean_s >= tm
    elif rule == "max":
        alert = max_s >= tx
    else:
        alert = (mean_s >= tm) | (max_s >= tx)
    benign = ~is_pos
    normal_pos = is_pos & ~is_dil
    out = dict(
        t_mean=tm, t_max=tx,
        fpr=float(alert[benign].mean()) if benign.any() else None,
        false_positives=int(alert[benign].sum()), n_benign=int(benign.sum()),
        recall_all=float(alert[is_pos].mean()) if is_pos.any() else None,
        recall_normal_shell=float(alert[normal_pos].mean()) if normal_pos.any() else None,
        recall_diluted=float(alert[is_dil].mean()) if is_dil.any() else None,
        n_normal_shell=int(normal_pos.sum()), n_diluted=int(is_dil.sum()),
    )
    # Headline breakdown: the dilution actually achieved, in fixed bins.
    bins = ratio_bins(achieved)
    per_bin = {}
    for i, lab in enumerate(ACHIEVED_RATIO_LABELS):
        m = is_dil & (bins == i)
        if not m.any():
            continue
        per_bin[lab] = dict(n=int(m.sum()), recall=float(alert[m].mean()))
    out["recall_diluted_by_achieved_ratio"] = per_bin
    out["n_diluted_without_achieved_ratio"] = int((is_dil & (bins < 0)).sum())
    # Kept for traceability only: the ratio that was requested of the generator.
    per_requested = {}
    for r in sorted(set(requested[is_dil].tolist())):
        if not np.isfinite(r):
            continue
        m = is_dil & (requested == r)
        per_requested[f"{r:g}"] = dict(n=int(m.sum()), recall=float(alert[m].mean()))
    out["recall_diluted_by_requested_ratio"] = per_requested
    tp = int(alert[is_pos].sum()); fp = out["false_positives"]
    fn = int(is_pos.sum()) - tp
    out["precision"] = tp / (tp + fp) if (tp + fp) else None
    out["f1"] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    return out


# ============================================================== link A1, consumed
# A1 is emitted by T1593 when its grader records a decoy USE. Its policy licenses ONE
# thing: raising priority. max_drop is 0.25 and the `why` says it reprioritises only,
# because a decoy use proves harvesting, not exploitation - nobody exploited anything
# by reading a planted path. So the alert DECISION here is left exactly as it was, and
# what the anchor changes is the order of the queue an analyst works down.
#
# The counterfactual - what would have happened had A1 been allowed to move the
# threshold too - is measured alongside it and reported as a counterfactual, because
# "whether it changed any decision" is the question asked of this link and the answer
# "none, by construction" is only informative next to the number it is being compared
# with.
LINK_IN = "A1"
LINK_OUT = "A2"


def a1_lookup(rows, store, br):
    """relaxation per test file, from the A1 anchor bound to its provenance group.

    A corpus file has no domain. The binding in common/bridge.json names, for each
    decoy channel, the provenance group that channel is assumed to stand for, so the
    lookup is: this file's group -> the decoy domain bound to it -> the strongest A1
    anchor on that domain.

    The anchor is evaluated at its OWN emission time. The corpus has no clock, so there
    is no consumer-side moment to ask about; the binding records that, and the validity
    window is therefore not a test in this link. Said here, in the code that relies on
    it, and again in the results.
    """
    out = {}
    if store is None or not br.enabled:
        return out
    for i, r in enumerate(rows):
        grp = r.get("group")
        if not grp:
            continue
        bound = br.reverse("T1190", "path", grp, from_kit="T1593")
        for b in bound:
            anc = store.match("domain", b.from_subject, link=LINK_IN)
            if anc is None:
                continue
            # Hard requirement 2: the policy names the identifier this consumer is
            # ALLOWED to match on. Matching an A1 anchor on anything but a domain
            # would be the silent kind-confusion common/anchors.py exists to remove.
            requires = anchors.policy_for(anc)["requires"]
            if requires != "domain":
                raise anchors.AnchorError(
                    f"link {anc.link} licenses relaxation only against subject_kind "
                    f"{requires!r}, but this kit matches on 'domain'")
            r_i = anchors.relaxation(anc, at=anc.emitted_at)
            if i not in out or r_i > out[i][0]:
                out[i] = (r_i, anc, b)
    return out


def a1_block(rows, max_s, is_pos, t_max, relax_of, br, anchor_file):
    """What the A1 anchor did: the reordering it licenses, and the decisions it did not take."""
    n = len(max_s)
    order_before = np.argsort(-max_s, kind="mergesort")
    rank_before = np.empty(n, dtype=int)
    rank_before[order_before] = np.arange(1, n + 1)

    pri = max_s.copy()
    for i, (r_i, _a, _b) in relax_of.items():
        # Priority only: the score is moved towards 1 by the licensed fraction, which
        # can only change the ORDER. The threshold below is the unchanged one.
        pri[i] = max_s[i] + r_i * (1.0 - max_s[i])
    order_after = np.argsort(-pri, kind="mergesort")
    rank_after = np.empty(n, dtype=int)
    rank_after[order_after] = np.arange(1, n + 1)

    alert = max_s >= t_max                        # the decision, unchanged by A1
    k = int(alert.sum())
    idx = sorted(relax_of)
    moved = [i for i in idx if rank_after[i] < rank_before[i]]
    into_topk = [i for i in moved if rank_after[i] <= k < rank_before[i]]

    # the counterfactual A1 does NOT license
    cf = alert.copy()
    for i, (r_i, _a, _b) in relax_of.items():
        cf[i] = max_s[i] >= t_max * (1.0 - r_i)
    changed = np.where(cf != alert)[0]

    subjects = sorted({a.subject for _r, a, _b in relax_of.values()})
    groups = sorted({rows[i].get("group") for i in idx})
    return {
        "link": LINK_IN,
        "anchor_file": anchor_file,
        "policy": anchors.CONSUMER_POLICY[LINK_IN],
        "matched_on_subject_kind": "domain",
        "anchor_subjects_matched": subjects,
        "bound_groups": groups,
        "n_test_files": n,
        "n_files_with_an_anchor": len(idx),
        "n_files_reprioritised": len(moved),
        "mean_rank_gain": (round(float(np.mean([rank_before[i] - rank_after[i]
                                                for i in moved])), 2) if moved else None),
        "max_rank_gain": (int(max(rank_before[i] - rank_after[i] for i in moved))
                          if moved else None),
        "n_alerts_unchanged": k,
        "n_moved_into_the_alerting_prefix": len(into_topk),
        "n_positives_moved_into_the_alerting_prefix": int(sum(1 for i in into_topk
                                                              if is_pos[i])),
        "decisions_changed": 0,
        "decisions_changed_note": (
            "zero BY CONSTRUCTION. A1's policy licenses reprioritisation only, so this "
            "kit applies the relaxation to the score's priority and never to the "
            "threshold. The alert set is bit-identical to the unanchored one."),
        "counterfactual_if_a1_could_move_the_threshold": {
            "decisions_changed": int(len(changed)),
            "newly_alerting_positives": int(sum(1 for i in changed
                                                if is_pos[i] and cf[i])),
            "newly_alerting_benign": int(sum(1 for i in changed
                                             if not is_pos[i] and cf[i])),
            "note": ("NOT licensed by A1 and NOT applied. Measured so the cost of the "
                     "policy's restriction is a number rather than an assertion."),
        },
        "window_note": ("the anchor was evaluated at its own emission time: the corpus "
                        "has no clock, so the validity window is not a test in this link"),
        **br.result_block(LINK_IN, anchor_file=anchor_file, n_matched=len(relax_of)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="independent runs (Kwon uses 5)")
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--budgets", nargs="*", type=float, default=[0.02, 0.01, 0.005],
                    help="false-positive budgets on the calibration split")
    ap.add_argument("--chunk-bytes", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-chunks", type=int, default=2048)
    ap.add_argument("--features", type=int, default=18, help="hashing features = 2**n")
    ap.add_argument("--ngram", nargs=2, type=int, default=[3, 5])
    ap.add_argument("--train-fraction", type=float, default=0.85,
                    help="files sampled per seed, so the runs differ as Kwon's do")
    ap.add_argument("--max-train-chunks", type=int, default=120_000)
    ap.add_argument("--tag", default="linear")
    ap.add_argument("--prior", type=float, default=0.01,
                    help="base rate of compromise among hosts of this kind. Reporting "
                         "only here: it turns this kit's operating point into the "
                         "confidence the A2 anchor it emits would carry, so the author "
                         "can see what this detector's word is worth to stage 3.")
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE RUN: cap each split at this many files, drawn "
                         "label-stratified and deterministically. The real run scores "
                         "every file and takes tens of minutes, which is too slow to "
                         "demonstrate a LINK. Numbers from a --limit run are labelled a "
                         "smoke run everywhere they are written and are NOT a result.")
    ap.add_argument("--fast", action="store_true",
                    help="shorthand for --limit 600 --seeds 1: the smallest run that "
                         "still trains, calibrates and emits an anchor.")
    ap.add_argument("--anchor-file",
                    help="link A1: a T1593 anchors JSONL. Given this, files whose "
                         "provenance group is bound to a decoy channel in "
                         "common/bridge.json have their PRIORITY raised (A1's max_drop "
                         "is 0.25 and it licenses reprioritisation only, so the alert "
                         "decision is left alone). Without it the kit behaves exactly as "
                         "it did before.")
    ap.add_argument("--anchor-min-confidence", type=float, default=0.0,
                    help="floor an A1 anchor must clear to be matched at all")
    ap.add_argument("--anchors-out", default=None,
                    help="where the A2 anchors this kit EMITS go "
                         "(default results/anchors_<tag>.jsonl)")
    ap.add_argument("--no-anchors", action="store_true",
                    help="emit no A2 anchor file")
    ap.add_argument("--max-anchors-per-subject", type=int, default=50,
                    help="cap on A2 anchors emitted per bound host. Every flagged file "
                         "gets an anchor, but one group can flag thousands of files and "
                         "they all name the same host, so the highest-scoring ones are "
                         "kept and the rest are counted and reported.")
    ap.add_argument("--flag-seed-majority", type=float, default=0.5,
                    help="a test file is treated as FLAGGED, and so anchored, when at "
                         "least this fraction of seeds flagged it at the chosen budget")
    ap.add_argument("--scorer", choices=("sklearn", "numpy"), default="sklearn",
                    help="the chunk scorer. 'sklearn' is the design: HashingVectorizer "
                         "over character n-grams plus liblinear logistic regression. "
                         "'numpy' is a STAND-IN for machines where scikit-learn cannot "
                         "be installed - a real classifier on the real corpus, so the "
                         "flags it produces are real detector output, but not this "
                         "detector, so its rates are NOT a result about aggregation and "
                         "are labelled as such wherever they are written.")
    bridge.add_arguments(ap, link="A2")
    a = ap.parse_args()
    if a.scorer == "sklearn" and sklearn is None:
        raise SystemExit(
            "scikit-learn is not importable here (%s), and it is what this experiment's "
            "chunk scorer is. Either install it - `pip install scikit-learn` - or rerun "
            "with --scorer numpy, which uses the numpy stand-in in "
            "scripts/numpy_scorer.py and labels every number it produces as not a "
            "measurement." % SKLEARN_IMPORT_ERROR)
    stand_in = a.scorer == "numpy"
    if stand_in:
        log("*** --scorer numpy: " + numpy_scorer.DESCRIPTION)
    if a.fast:
        a.limit = a.limit or 600
        a.seeds = 1
    smoke = bool(a.limit)

    br = bridge.load(a.bridge, no_bridge=a.no_bridge)
    in_store = None
    if a.anchor_file:
        in_store = anchors.AnchorStore.load(a.anchor_file)
        log(f"link A1: loaded {len(in_store)} anchor(s) from {a.anchor_file} "
            f"({in_store.malformed_lines} malformed line(s))")
        log(f"  bridge: {br.status}")
        if not br.enabled:
            log("  --no-bridge: no provenance group is bound to a decoy channel, so no "
                "A1 anchor can match and this run is identical to an unanchored one")

    index = read_jsonl(DATA / "index.jsonl")
    dil_path = DATA / "dilution.jsonl"
    dil = read_jsonl(dil_path) if dil_path.exists() else []
    for d in dil:
        d.setdefault("kind", "diluted")
    for r in index:
        r.setdefault("kind", "plain")

    train_rows = [r for r in index if r["split"] == "train"]
    calib_rows = [r for r in index if r["split"] == "calib"] + [d for d in dil if d["split"] == "calib"]
    test_rows = [r for r in index if r["split"] == "test"] + [d for d in dil if d["split"] == "test"]

    if smoke:
        # Label-stratified, deterministic, and it keeps every diluted file it can,
        # because the diluted positives are the thing the design is about and a plain
        # uniform sample of 600 files out of 10 000 would drop nearly all of them.
        # This is a SMOKE RUN. It exists so the anchor LINK can be demonstrated
        # end to end; its tpr and fpr are not a result and are labelled as such in the
        # operating point, in the results file and in the console output.
        def cap(rows, n):
            rng = np.random.default_rng(a.seed_base)
            pos = [r for r in rows if r["label"] == 1]
            neg = [r for r in rows if r["label"] != 1]
            dilp = [r for r in pos if r.get("kind") == "diluted"]
            plain = [r for r in pos if r.get("kind") != "diluted"]
            want_pos = max(1, n // 3)
            keep_dil = dilp[:min(len(dilp), max(1, want_pos // 2))] if dilp else []
            keep_plain = [plain[i] for i in
                          rng.permutation(len(plain))[:max(1, want_pos - len(keep_dil))]] if plain else []
            keep_pos = keep_dil + keep_plain
            keep_neg = [neg[i] for i in rng.permutation(len(neg))[:max(1, n - len(keep_pos))]]
            return keep_pos + keep_neg

        before = (len(train_rows), len(calib_rows), len(test_rows))
        train_rows = cap(train_rows, a.limit)
        calib_rows = cap(calib_rows, a.limit)
        test_rows = cap(test_rows, a.limit)
        log(f"*** SMOKE RUN (--limit {a.limit}): splits cut from {before} to "
            f"{(len(train_rows), len(calib_rows), len(test_rows))} files. The numbers "
            f"below demonstrate the LINK, not the design.")
    for name, rows in (("train", train_rows), ("calib", calib_rows), ("test", test_rows)):
        pos = sum(r["label"] for r in rows)
        log(f"{name:6s} {len(rows):6d} files  ({pos} malicious, {len(rows)-pos} benign)")
        if not rows or pos == 0 or pos == len(rows):
            sys.exit(f"{name} split needs both classes")

    def arrays(rows):
        is_pos = np.array([r["label"] == 1 for r in rows])
        is_dil = np.array([r.get("kind") == "diluted" for r in rows])
        # The benign files that diluted positives in THIS split were built from. They are
        # near-identical to those positives, so they are held out of the diluted-only AUC
        # negatives (see auc_block). Matched on the stored host_path.
        hosts = {r["host_path"] for r in rows
                 if r.get("kind") == "diluted" and r.get("host_path")}
        is_host = np.array([(r["label"] != 1) and (r.get("path") in hosts) for r in rows])
        requested = np.array([r.get("inject_ratio", np.nan) for r in rows], dtype=float)
        achieved = np.array([achieved_ratio_of(r) for r in rows], dtype=float)
        n_dil = int(is_dil.sum())
        if n_dil and not is_host.any():
            log(f"  WARNING: {n_dil} diluted file(s) in this split but NOT ONE of their "
                f"host_path values matched a benign row's path, so no host could be held "
                f"out of the diluted-only AUC. The paths are stored in a different form "
                f"than index.jsonl uses - fix that before trusting diluted_positives.")
        return is_pos, is_dil, is_host, requested, achieved

    c_pos, c_dil, c_host, c_req, c_ach = arrays(calib_rows)
    t_pos, t_dil, t_host, t_req, t_ach = arrays(test_rows)

    if stand_in:
        vec = numpy_scorer.HashedNgrams(n_features=2 ** a.features,
                                        ngram_range=tuple(a.ngram))
    else:
        vec = HashingVectorizer(analyzer="char", ngram_range=tuple(a.ngram),
                                n_features=2 ** a.features, alternate_sign=False,
                                norm="l2")

    config = dict(vars(a))
    config.update(sklearn_version=(sklearn.__version__ if sklearn else None),
                  sklearn_import_error=SKLEARN_IMPORT_ERROR,
                  scorer_is_a_stand_in=stand_in,
                  scorer_description=(numpy_scorer.DESCRIPTION if stand_in
                                      else "scikit-learn HashingVectorizer + liblinear"),
                  numpy_version=np.__version__,
                  platform=platform.platform())
    splits_block = {k: dict(n=len(v), sources=sorted({r["source_id"] for r in v}))
                    for k, v in (("train", train_rows), ("calib", calib_rows), ("test", test_rows))}
    n_diluted_block = dict(calib=int(c_dil.sum()), test=int(t_dil.sum()))

    # Which test file carries which A1 anchor. It depends only on the bridge and the
    # anchor file, not on the model, so it is resolved once and reused per seed.
    a1_relax = a1_lookup(test_rows, in_store, br)
    if in_store is not None:
        log(f"link A1: {len(a1_relax)} of {len(test_rows)} test files are bound to a "
            f"decoy channel with a live anchor"
            + (f"; relaxation {min(r for r, _, _ in a1_relax.values()):.4f}.."
               f"{max(r for r, _, _ in a1_relax.values()):.4f}" if a1_relax else ""))

    RESULTS.mkdir(parents=True, exist_ok=True)
    scores_path = truncate_jsonl(RESULTS / f"scores_{a.tag}.jsonl")
    log(f"per-file scores -> {scores_path} (emptied; one row per file per seed)")
    runs = []

    def save_partial():
        """Persist the raw per-seed results as each seed finishes.

        Scoring every seed is the expensive part of the run, so it must not be lost
        to a failure in a later seed or in the aggregation below.
        """
        save_json(RESULTS / f"experiment_{a.tag}_runs.json",
                  dict(config=config, splits=splits_block, n_diluted=n_diluted_block,
                       partial=True, n_seeds_requested=a.seeds, runs=runs))

    for k in range(a.seeds):
        seed = a.seed_base + k
        t0 = time.time()
        rng = np.random.default_rng(seed)
        sel = rng.random(len(train_rows)) < a.train_fraction
        rows_k = [r for r, s in zip(train_rows, sel) if s] or train_rows
        texts, owner, trunc = build_chunks(rows_k, a.chunk_bytes, a.stride, a.max_chunks)
        y = np.array([rows_k[o]["label"] for o in owner])
        if len(texts) > a.max_train_chunks:
            keep = rng.choice(len(texts), a.max_train_chunks, replace=False)
            texts = [texts[i] for i in keep]; y = y[keep]
        log(f"seed {seed}: {len(rows_k)} train files -> {len(texts)} chunks "
            f"({int(y.sum())} malicious){'  [some files truncated]' if trunc else ''}")
        X = vec.transform(texts)
        if stand_in:
            model = numpy_scorer.LogisticSGD(random_state=seed)
        else:
            model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                       solver="liblinear", random_state=seed)
        model.fit(X, y)

        cm, cx, cn = per_file_scores(model, vec, calib_rows, a.chunk_bytes, a.stride, a.max_chunks)
        tm_, tx_, tn = per_file_scores(model, vec, test_rows, a.chunk_bytes, a.stride, a.max_chunks)

        # The per-file scores are appended before anything is thresholded, so AUC, ROC
        # and PR curves, any other false-positive budget and any error analysis can be
        # done later without retraining. One file for the run, one row per file per seed.
        append_jsonl(scores_path, score_rows(calib_rows, cm, cx, cn, seed, "calib")
                     + score_rows(test_rows, tm_, tx_, tn, seed, "test"))

        run = dict(seed=seed, n_train_files=len(rows_k), n_train_chunks=int(len(y)),
                   seconds=round(time.time() - t0, 1),
                   trunc=dict(train_files=int(trunc),
                              calib_files_at_chunk_cap=int((cn >= a.max_chunks).sum()),
                              test_files_at_chunk_cap=int((tn >= a.max_chunks).sum())),
                   auc=auc_block(tm_, tx_, t_pos, t_dil, t_host), budgets={})
        for B in a.budgets:
            t_mean = pick_1d(cm[~c_pos], B)
            t_max = pick_1d(cx[~c_pos], B)
            tm2, tx2, degenerate = pick_2d(cm[~c_pos], cx[~c_pos], cm[c_pos], cx[c_pos],
                                           c_dil[c_pos], B,
                                           extra_max_candidates=(t_max,))
            if degenerate:
                log(f"  WARNING seed {seed}, budget {B:g}: no (t_mean, t_max) pair met the "
                    f"false-positive budget on calibration; mean_or_max fell back to "
                    f"thresholds above every benign score, so its recall here is not a "
                    f"result but an artefact. Reported as degenerate.")
            res = {
                "mean": evaluate("mean", (t_mean, np.nan), tm_, tx_, t_pos, t_dil, t_req, t_ach),
                "max": evaluate("max", (np.nan, t_max), tm_, tx_, t_pos, t_dil, t_req, t_ach),
                "mean_or_max": evaluate("mean_or_max", (tm2, tx2), tm_, tx_, t_pos, t_dil,
                                        t_req, t_ach),
            }
            res["degenerate"] = bool(degenerate)
            res["calibration"] = dict(
                budget=B,
                mean=dict(zip(("fpr", "recall"), map(float, fpr_recall_1d(cm[c_pos], cm[~c_pos], t_mean)))),
                max=dict(zip(("fpr", "recall"), map(float, fpr_recall_1d(cx[c_pos], cx[~c_pos], t_max)))),
                mean_or_max=dict(
                    fpr=float((((cm[~c_pos] >= tm2) | (cx[~c_pos] >= tx2))).mean()),
                    recall=float((((cm[c_pos] >= tm2) | (cx[c_pos] >= tx2))).mean())),
            )
            if in_store is not None:
                res["anchor_a1"] = a1_block(test_rows, tx_, t_pos, t_max, a1_relax, br,
                                            a.anchor_file)
            # which test files the MAX rule actually flagged this seed, for link A2
            res["max_flagged_idx"] = [int(i) for i in np.where(tx_ >= t_max)[0]]
            run["budgets"][f"{B:g}"] = res
        runs.append(run)
        save_partial()
        log(f"  done in {run['seconds']}s")

    # ------------------------------------------------------------ aggregate
    if not runs:
        sys.exit("no seed completed, so there is nothing to aggregate "
                 f"(--seeds was {a.seeds}); the per-seed file, if any, is "
                 f"{RESULTS / f'experiment_{a.tag}_runs.json'}")

    # The results file is written now, holding every per-seed number, and rewritten
    # with the summary once that is built. Summarising and printing come after, so a
    # failure there can no longer throw away a run that has already been scored.
    # Every file the scorer could not read. An unreadable file currently scores 0.0 and
    # becomes a confident negative, so this count MUST travel with the corpus size or a
    # quarantined malicious file is silently reported as a correct answer. It is recorded
    # rather than fatal: a three-hour run must not be thrown away at the last step.
    _missing = sorted({m["path"]: m for m in MISSING_FILES}.values(),
                      key=lambda m: m["path"])
    missing_block = dict(
        n_unreadable_files=len(_missing),
        corpus_complete=not _missing,
        files=_missing[:200],
        truncated_to=200 if len(_missing) > 200 else None,
        note=("files the scorer could not read. Each one scored 0.0 and was counted in "
              "its class, so a non-zero count means the reported corpus size is larger "
              "than the corpus actually used and every rate here has the wrong "
              "denominator. Re-index before reporting."),
    )
    if _missing:
        log(f"\n  WARNING: {len(_missing)} file(s) could not be read and scored 0.0, so "
            f"they were counted as confident negatives. corpus_complete is false and the "
            f"denominators in this file are wrong. First few: "
            f"{[m['path'] for m in _missing[:3]]}")

    out = dict(config=config, splits=splits_block, n_diluted=n_diluted_block,
               unreadable=missing_block,
               runs=runs, summary=None, auc_summary=None, summarised=False)
    save_json(RESULTS / f"experiment_{a.tag}.json", out)

    def agg(budget, rule, field):
        vals = [r["budgets"][budget][rule][field] for r in runs
                if r["budgets"][budget][rule][field] is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    values=[float(v) for v in vals])

    summary = {}
    for B in a.budgets:
        key = f"{B:g}"
        s = {}
        for rule in ("mean", "max", "mean_or_max"):
            s[rule] = {f: agg(key, rule, f) for f in
                       ("fpr", "recall_all", "recall_normal_shell", "recall_diluted", "precision", "f1")}
        # recall_diluted is None whenever the test split holds no diluted file
        # (dilution.jsonl absent, or none of it landed in test), so the gain and the
        # paired test are simply not defined; that must not end the run.
        pairs = [(r["budgets"][key]["mean_or_max"]["recall_diluted"],
                  r["budgets"][key]["max"]["recall_diluted"]) for r in runs]
        usable = [(p, q) for p, q in pairs if p is not None and q is not None]
        d = [p - q for p, q in usable]
        if d:
            s["diluted_recall_gain_over_max"] = dict(
                mean=float(np.mean(d)), sd=float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                values=[float(x) for x in d])
        else:
            s["diluted_recall_gain_over_max"] = None
            s["diluted_recall_gain_over_max_reason"] = (
                "no diluted files in the test split, so diluted recall is undefined for "
                "both rules; check that data/dilution.jsonl exists and that "
                "03_make_dilution_set.py was run with test among --splits")
        try:
            from scipy.stats import ttest_rel
            mm = [p for p, _ in usable]
            mx = [q for _, q in usable]
            if len(mm) > 1 and np.std(np.array(mm) - np.array(mx)) > 0:
                t, p = ttest_rel(mm, mx)
                s["paired_t_test_vs_max"] = dict(t=float(t), p=float(p), n=len(mm))
            elif not mm:
                s["paired_t_test_vs_max"] = dict(note="no diluted recall to compare", n=0)
            else:
                s["paired_t_test_vs_max"] = dict(note="no variation across seeds", n=len(mm))
        except Exception as e:
            s["paired_t_test_vs_max"] = dict(error=str(e))
        s["degenerate_seeds"] = [r["seed"] for r in runs if r["budgets"][key].get("degenerate")]
        summary[key] = s

    def agg_vals(vals):
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    values=vals)

    # Threshold-free, so it sits beside the per-budget summary rather than inside it.
    auc_summary = {}
    for subset in AUC_SUBSETS:
        auc_summary[subset] = {
            score: {m: agg_vals([r["auc"][subset][score][m] for r in runs])
                    for m in ("roc_auc", "average_precision")}
            for score in ("mean_score", "max_score")}
        auc_summary[subset]["n_positives"] = runs[0]["auc"][subset]["n_positives"]
        auc_summary[subset]["n_negatives"] = runs[0]["auc"][subset]["n_negatives"]

    # ---- the operating point this kit hands to stage 3 (link A2, consumed by T1021).
    # The max rule is the one the anchor stands behind: it is Kwon's published rule and
    # the one the design is measured against, so a confidence derived from it is a
    # statement about a detector that exists elsewhere in the literature rather than
    # about this kit's own variant. The TIGHTEST budget with a defined recall is chosen,
    # because an anchor is only worth emitting at an alert rate an analyst would run at.
    op_path = RESULTS / f"operating_point_{a.tag}.json"
    op_block = None
    chosen_key, chosen_op, op_refused, implied = None, None, None, None
    n_pos_test, n_neg_test = int(t_pos.sum()), int((~t_pos).sum())
    for B in sorted(a.budgets):
        key = f"{B:g}"
        g = summary.get(key, {}).get("max") or {}
        if not g.get("recall_all") or not g.get("fpr"):
            continue
        t_max_vals = [r["budgets"][key]["max"]["t_max"] for r in runs]
        own_op = anchors.OperatingPoint(
            tpr=float(g["recall_all"]["mean"]), fpr=float(g["fpr"]["mean"]),
            n_pos=n_pos_test, n_neg=n_neg_test,
            # the chosen t_max, averaged over seeds: each seed picks its own on its own
            # calibration split, and the sd is in summary[...]["max"] already
            threshold=float(np.mean(t_max_vals)),
            detector=(f"T1190 max-rule @ budget {key}"
                      + (" [SMOKE RUN on a --limit subset: not a measurement]"
                         if smoke else "")
                      + (" [NUMPY STAND-IN SCORER, not scikit-learn: not a measurement]"
                         if stand_in else "")))
        own_op.save(op_path)
        op_block = own_op.as_dict()
        chosen_key, chosen_op = key, own_op
        # An operating point can be REFUSED as evidence - LR+ at or below 1, or a
        # measured FPR of zero with no negatives. common/anchors.py raises AnchorError
        # for both; letting it propagate here would throw away a run that has already
        # been scored, so it is caught, reported, and no anchor is emitted.
        try:
            implied = anchors.confidence_from_operating_point(own_op, a.prior)
        except anchors.AnchorError as e:
            op_refused = str(e)
            log(f"\nREFUSED: the max rule at budget {key} is not evidence, so link A2 "
                f"emits nothing. {op_refused}")
            break
        log(f"\n--- the operating point this kit hands to stage 3 (link A2) ---")
        log(f"  wrote {op_path.name}: max rule at the tightest budget with a defined "
            f"recall ({key}); tpr {own_op.tpr:.4f} over {own_op.n_pos} malicious test "
            f"file(s), fpr {own_op.fpr:.6g} over {own_op.n_neg} benign, effective fpr "
            f"{own_op.effective_fpr():.6g}, t_max {own_op.threshold:.6g}, LR+ "
            f"{own_op.lr_plus():.6g}")
        log(f"  at prior {a.prior} the A2 anchor T1021 builds from it carries confidence "
            f"{implied:.4f}  (run T1021 with --emitter-operating-point "
            f"{op_path} --prior {a.prior})")
        break
    else:
        log(f"\nNOT writing {op_path.name}: no budget produced a defined test recall for "
            f"the max rule, so there is no operating point to hand to stage 3.")

    # ================================================== link A2, emitted
    # One anchor per test file this kit's detector ACTUALLY FLAGGED at the chosen
    # operating point - not per malicious file, and not per labelled anything. A file
    # is flagged when at least --flag-seed-majority of the seeds put its max chunk
    # score at or above that seed's own t_max, which is the decision the detector took.
    # The subject is a HOST, because link A2's policy requires one and T1021 can only
    # match on one, and the host comes from the declared binding: the corpus has no
    # host, so each provenance group is treated as one web server.
    anchors_path = Path(a.anchors_out) if a.anchors_out else RESULTS / f"anchors_{a.tag}.jsonl"
    a2 = {"link": LINK_OUT, "emitted": 0, "flagged_files": 0, "unbound_files": 0,
          "capped": 0, "refused": op_refused, "anchors_file": None,
          "by_subject": {}, "unbound_groups": [], "note": "",
          "smoke_run": smoke}
    if a.no_anchors:
        a2["note"] = "--no-anchors: nothing emitted"
    elif chosen_op is None:
        a2["note"] = ("no budget produced a defined test recall for the max rule, so "
                      "there is no operating point and nothing to anchor from")
    elif op_refused:
        a2["note"] = "the operating point was refused as evidence: " + op_refused
    elif not br.enabled:
        a2["note"] = ("--no-bridge: no provenance group is bound to a host, so no file "
                      "can be given a host subject and nothing is emitted. This is the "
                      "ablation.")
        log("\nlink A2: --no-bridge, so no A2 anchor can be given a subject; nothing emitted")
    else:
        need = max(1, int(round(a.flag_seed_majority * len(runs))))
        hits = {}
        for r in runs:
            for i in r["budgets"][chosen_key]["max_flagged_idx"]:
                hits[i] = hits.get(i, 0) + 1
        flagged = sorted([i for i, c in hits.items() if c >= need])
        a2["flagged_files"] = len(flagged)
        a2["flag_rule"] = (f"max chunk score >= that seed's own t_max at budget "
                           f"{chosen_key}, in at least {need} of {len(runs)} seed(s)")
        # How many seeds flagged each file, so the per-host cap keeps the detections the
        # detector was most consistent about rather than whichever file came first.
        score_of = {i: hits[i] for i in flagged}
        per_subject = {}
        for i in flagged:
            grp = test_rows[i].get("group")
            b = br.resolve("T1190", "path", grp, "T1021") if grp else None
            if b is None:
                a2["unbound_files"] += 1
                if grp and grp not in a2["unbound_groups"]:
                    a2["unbound_groups"].append(grp)
                continue
            per_subject.setdefault(b.to_subject, []).append((i, b))
        anchors_path.parent.mkdir(parents=True, exist_ok=True)
        anchors_path.write_text("")       # one emission log per tagged run
        store = anchors.AnchorStore(anchors_path)
        for subj, items in sorted(per_subject.items()):
            items.sort(key=lambda t: (-score_of[t[0]], t[0]))
            keep, drop = items[:a.max_anchors_per_subject], items[a.max_anchors_per_subject:]
            a2["capped"] += len(drop)
            for i, b in keep:
                row = test_rows[i]
                # The window comes from the BINDING, not from the corpus: the corpus has
                # no clock at all, so a foothold moment has to be declared, and
                # bridge.json's to_time_basis says where the declaration came from.
                vf = b.to_valid_from if b.to_valid_from is not None else 0.0
                ttl = b.to_ttl_seconds if b.to_ttl_seconds is not None else 86400.0
                try:
                    store.emit(anchors.Anchor.create(
                        LINK_OUT, "host", subj, emitted_at=float(vf),
                        ttl_seconds=float(ttl),
                        operating_point=chosen_op, prior=a.prior,
                        evidence={
                            "emitting_stage": "stage 2 foothold (T1190)",
                            "detection": "max-rule flag on a corpus file",
                            "file_path": row["path"], "group": row.get("group"),
                            "source_id": row.get("source_id"),
                            "kind": row.get("kind"),
                            "label_for_bookkeeping_only": row["label"],
                            "seeds_flagging": hits[i], "seeds": len(runs),
                            "budget": chosen_key,
                            "smoke_run": smoke,
                            "bridge_binding": b.as_dict(),
                            "bridge_status": br.status,
                            "window_basis": b.to_time_basis,
                        },
                        note=("real detector output: this file was flagged by the max "
                              "rule at the chosen operating point"
                              + (" [SMOKE RUN]" if smoke else ""))))
                    a2["emitted"] += 1
                    a2["by_subject"][subj] = a2["by_subject"].get(subj, 0) + 1
                except anchors.AnchorError as e:
                    a2["refused"] = str(e)
                    log(f"  REFUSED one A2 anchor: {e}")
        a2["anchors_file"] = anchors_path.name
        a2["loaded_anchor_summary"] = anchors.AnchorStore.load(anchors_path).summary()
        log(f"\n--- link A2: anchors emitted for stage 3 (T1021) ---")
        log(f"  {a2['flagged_files']} test file(s) flagged by the max rule at budget "
            f"{chosen_key} ({a2['flag_rule']})")
        log(f"  wrote {anchors_path.name}: {a2['emitted']} anchor(s), subject_kind "
            f"'host', confidence {implied:.4f} at prior {a.prior}")
        for subj, n in sorted(a2["by_subject"].items()):
            log(f"    host {subj:10s} {n:4d} anchor(s)")
        if a2["capped"]:
            log(f"  {a2['capped']} further flagged file(s) named a host that already had "
                f"{a.max_anchors_per_subject} anchors and were not emitted "
                f"(--max-anchors-per-subject)")
        if a2["unbound_files"]:
            log(f"  {a2['unbound_files']} flagged file(s) are in groups with no host "
                f"binding, so they cannot reach stage 3: {a2['unbound_groups']}")
        log(f"  next: python3 ../T1021_anchor/scripts/02_run_experiment.py "
            f"--anchor-file {anchors_path} --tag {a.tag}")
    a2.update(br.result_block(LINK_OUT, anchor_file=a2["anchors_file"], n_matched=0))

    # The per-seed flag index lists were only needed to build the anchors; they are
    # thousands of integers per seed on a full run, so they do not go in the results.
    for r in runs:
        for blk in r["budgets"].values():
            blk.pop("max_flagged_idx", None)

    a1_runs = [r["budgets"][k].get("anchor_a1") for r in runs for k in r["budgets"]
               if r["budgets"][k].get("anchor_a1")]
    a1_summary = None
    if a1_runs:
        a1_summary = {
            "n_blocks": len(a1_runs),
            **br.result_block(LINK_IN, anchor_file=a.anchor_file,
                              n_matched=len(a1_relax)),
            "n_files_with_an_anchor": a1_runs[0]["n_files_with_an_anchor"],
            "n_files_reprioritised_mean": round(float(np.mean(
                [b["n_files_reprioritised"] for b in a1_runs])), 2),
            "n_moved_into_the_alerting_prefix_mean": round(float(np.mean(
                [b["n_moved_into_the_alerting_prefix"] for b in a1_runs])), 2),
            "n_positives_moved_into_the_alerting_prefix_mean": round(float(np.mean(
                [b["n_positives_moved_into_the_alerting_prefix"] for b in a1_runs])), 2),
            "decisions_changed": 0,
            "counterfactual_decisions_changed_mean": round(float(np.mean(
                [b["counterfactual_if_a1_could_move_the_threshold"]["decisions_changed"]
                 for b in a1_runs])), 2),
            "example": a1_runs[0],
        }
        log(f"\n--- link A1: what the decoy anchor did to this kit ---")
        log(f"  {a1_summary['n_files_with_an_anchor']} of {len(test_rows)} test files "
            f"carry an A1 anchor through the declared binding")
        log(f"  reprioritised (rank improved): "
            f"{a1_summary['n_files_reprioritised_mean']} file(s) per block; "
            f"{a1_summary['n_moved_into_the_alerting_prefix_mean']} moved into the "
            f"alerting prefix, of which "
            f"{a1_summary['n_positives_moved_into_the_alerting_prefix_mean']} were "
            f"malicious")
        log(f"  decisions changed: 0, BY CONSTRUCTION - A1 licenses reprioritisation "
            f"only (max_drop 0.25, 'it reprioritises only')")
        log(f"  counterfactual, had A1 been allowed to move the threshold too: "
            f"{a1_summary['counterfactual_decisions_changed_mean']} decision(s) would "
            f"have changed. NOT applied.")

    out.update(summary=summary, auc_summary=auc_summary, summarised=True,
               scorer=a.scorer, scorer_is_a_stand_in=stand_in,
               scorer_note=(numpy_scorer.DESCRIPTION if stand_in else None),
               smoke_run=smoke,
               smoke_run_note=(("--limit %d was used: each split was cut to a stratified "
                                "subset so the anchor chain could be run end to end. "
                                "Every number in this file is a SMOKE RUN and is not a "
                                "result about the design." % a.limit) if smoke else None),
               anchor_a1_consumed=a1_summary,
               anchor_a2_emitted=a2,
               operating_point_emitted=op_block,
               operating_point_refused=op_refused,
               operating_point_file=(op_path.name if op_block else None),
               operating_point_prior_for_report=a.prior,
               operating_point_note=("The max rule (Kwon 2026) at the tightest "
                                     "false-positive budget whose test recall is defined. "
                                     "tpr and fpr are the measured test figures, averaged "
                                     "over seeds; n_pos and n_neg are the test counts. "
                                     "T1021 reads this file and derives its A2 anchor "
                                     "confidence from it, so stage 3 never asserts one."))
    save_json(RESULTS / f"experiment_{a.tag}.json", out)

    lines = ["budget,rule,fpr_mean,fpr_sd,recall_diluted_mean,recall_diluted_sd,"
             "recall_normal_mean,recall_normal_sd,f1_mean"]
    for B in a.budgets:
        key = f"{B:g}"
        for rule in ("mean", "max", "mean_or_max"):
            g = summary[key][rule]
            def v(f, k="mean"):
                return "" if g[f] is None else f"{g[f][k]:.4f}"
            lines.append(",".join([key, rule, v("fpr"), v("fpr", "sd"), v("recall_diluted"),
                                   v("recall_diluted", "sd"), v("recall_normal_shell"),
                                   v("recall_normal_shell", "sd"), v("f1")]))
    # Second table: ranking quality, which has no threshold and so no budget or rule.
    lines.append("")
    lines.append("positives,score,metric,mean,sd,n_seeds")
    for subset in AUC_SUBSETS:
        for score in ("mean_score", "max_score"):
            for m in ("roc_auc", "average_precision"):
                g = auc_summary[subset][score][m]
                lines.append(",".join([
                    subset, score, m,
                    "" if g is None else f"{g['mean']:.4f}",
                    "" if g is None else f"{g['sd']:.4f}",
                    "0" if g is None else str(len(g["values"]))]))
    (RESULTS / f"summary_{a.tag}.csv").write_text("\n".join(lines) + "\n")
    log(f"wrote {RESULTS / f'summary_{a.tag}.csv'}")

    log("\n--- test split, mean over seeds ---")
    for B in a.budgets:
        key = f"{B:g}"
        log(f"calibration FPR budget {key}")
        for rule in ("mean", "max", "mean_or_max"):
            g = summary[key][rule]

            def f4(field, g=g):
                # recall_diluted is None when the test split holds no diluted file.
                return "n/a" if g.get(field) is None else f"{g[field]['mean']:.4f}"

            log(f"  {rule:12s} fpr={f4('fpr')}  diluted recall={f4('recall_diluted')}"
                f"  plain-shell recall={f4('recall_normal_shell')}")
        gn = summary[key]["diluted_recall_gain_over_max"]
        if gn is None:
            log(f"  mean_or_max - max on diluted recall: not available "
                f"({summary[key]['diluted_recall_gain_over_max_reason']})")
        else:
            log(f"  mean_or_max - max on diluted recall: {gn['mean']:+.4f} (sd {gn['sd']:.4f})")
        if summary[key]["degenerate_seeds"]:
            log(f"  WARNING degenerate mean_or_max thresholds for seed(s) "
                f"{summary[key]['degenerate_seeds']} at this budget")

    log("\n--- ranking quality on the test split, mean over seeds (threshold-free) ---")
    for subset in AUC_SUBSETS:
        for score in ("mean_score", "max_score"):
            b = auc_summary[subset][score]

            def fa(metric, b=b):
                return "n/a" if b[metric] is None else f"{b[metric]['mean']:.4f}"

            log(f"  {subset:18s} {score:10s} roc_auc={fa('roc_auc')}  "
                f"average_precision={fa('average_precision')}")
    log("\nnext: python3 scripts/05_bundle_results.py")


if __name__ == "__main__":
    main()
