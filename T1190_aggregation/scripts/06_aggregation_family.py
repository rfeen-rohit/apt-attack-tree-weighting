#!/usr/bin/env python3
"""Is there ANY aggregation rule, or ANY chunk size, that recovers diluted web shells?

WHY THIS EXPERIMENT EXISTS
--------------------------
04_run_experiment.py answered the question the paper's design asks and the answer was
negative: the proposed `mean_or_max` rule is WORSE than the maximum alone on diluted
recall at every false-positive budget (-0.018 at budget 0.02, -0.153 at 0.01, -0.117
at 0.005), and threshold-free on diluted positives the max score ranks far better than
the mean (ROC AUC 0.8376 vs 0.5912). Max-only recovered 16-23% of diluted shells;
mean-only 3-4%.

`max` is **Kwon 2026's published rule**, not this project's. So "max wins" is not a
finding this project may claim; it is the incumbent. The open question, and the one
this script answers, is:

    Is there an aggregation rule, or a chunk size, that beats Kwon's max on diluted
    payloads - and if not, what is the SHAPE of the space?

Three things are swept, all on the same chunk scores so that only the named variable
moves:

  Part 1  THE AGGREGATION FAMILY. `mean` and `max` are the two ends of one family.
          Between them: the quantiles of a file's chunk-score distribution, the mean of
          the k highest chunk scores, and a count of how many chunks clear a threshold.
          The top-k mean is the principled middle - an injected payload spans a few
          chunks, so a handful of high scores is the signal, while the mean dilutes it
          over hundreds and the max throws away the corroboration. The count is the
          other shape of the same intuition: a dilution signature is "a few very high
          chunks", which neither a mean nor a max expresses.

  Part 2  THE CHUNK-SIZE SWEEP. Nobody has swept this, and it is probably the more
          important half. A smaller chunk means an injected payload occupies a LARGER
          FRACTION of the chunk that contains it, so its chunk score should rise.
          Sizes {128, 256, 512, 1024, 2048} with stride = size/2.

  Part 3  A LEARNED COMBINER. Logistic regression over the per-file aggregate features,
          trained on the CALIBRATION split only and evaluated on test. The interesting
          outcome is whether it drives the mean's weight to zero, which would say
          outright that the mean carries no information under dilution.

FAIRNESS, WHICH IS NOT OPTIONAL
-------------------------------
This kit has a past defect where a proposed rule was tuned on a signal the baselines
were denied. The guards here:

  1. EVERY rule gets the SAME selection objective, computed by one function
     (`pick_threshold`): on the CALIBRATION split, the threshold that maximises
     calibration recall subject to calibration FPR <= budget, ties broken by lower FPR
     and then by the larger threshold. The candidate set is every distinct calibration
     score, so no rule gets a finer grid than another. Nothing is tuned on test.
  2. Rules with MORE THAN ONE free parameter are labelled with their parameter count
     and reported separately, never mixed into the single-threshold table as though
     they cost the same.
  3. The learned combiner has more free parameters than any threshold rule, so it ships
     with a PERMUTATION CONTROL: the same fit on shuffled calibration labels, which is
     the test performance that chance buys at that parameter count.
  4. Chunk coverage is held equal across chunk sizes. `--max-chunks` is scaled as
     512/size so every size sees the same number of BYTES of each file; otherwise a
     small chunk size would be penalised for being truncated earlier, and an injected
     payload past the cap would be invisible at 128 bytes and visible at 2048.
  5. Diluted results are binned by the dilution ACHIEVED (`achieved_ratio`), never by
     the ratio requested of the generator.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not retrain the chunk scorer per aggregation rule: within one chunk size, one
scorer per seed is trained and held fixed and only the aggregation changes, exactly as
in 04. ACROSS chunk sizes the scorer necessarily retrains, because a scorer fitted on
512-byte chunks would be mismatched to 128-byte ones; that is a property of the
question, and it is stated in the results file.

The chunking and the achieved-ratio handling are IMPORTED from 04_run_experiment.py
rather than reimplemented, so this script cannot drift from the run it is compared with
and 04 itself is not touched.
"""
from __future__ import annotations

import argparse, importlib.util, json, math, platform, sys, time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.append(str(SCRIPTS.parents[1] / "common"))

from common import (DATA, RESULTS, append_jsonl, log, read_jsonl, save_json,
                    truncate_jsonl, write_jsonl)
import numpy_scorer


def _load_04():
    """Import 04_run_experiment.py as a module. Its name starts with a digit, so the
    import system cannot be asked for it by name; nothing in it runs at import time
    except definitions, because its body is under `if __name__ == '__main__'`."""
    path = SCRIPTS / "04_run_experiment.py"
    spec = importlib.util.spec_from_file_location("exp04", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M4 = _load_04()
file_chunks = M4.file_chunks                      # the chunker, unchanged
roc_auc = M4.roc_auc                              # Mann-Whitney AUC, ties half-credited
average_precision = M4.average_precision
achieved_ratio_of = M4.achieved_ratio_of
ratio_bins = M4.ratio_bins
RATIO_LABELS = M4.ACHIEVED_RATIO_LABELS
pick_2d = M4.pick_2d                              # the existing mean_or_max picker

EPS = 1e-12

QUANTS = (0.50, 0.75, 0.90, 0.95, 0.99, 1.00)     # 1.00 is exactly max; 0.50 the median
TOPKS = (1, 2, 3, 5, 10)                          # 1 is exactly max
COUNT_GRID = np.round(np.linspace(0.0, 1.0, 1001), 4)   # chunk-score thresholds for counts
HIST_BINS = 10000                                 # resolution of the benign-chunk histogram
POPULATIONS = ("all_positives", "plain_positives", "diluted_positives",
               "diluted_positives_with_hosts")


# ----------------------------------------------------------------- per-file features
def file_features(rows, model, vec, size, stride, max_chunks, hist_mask=None, batch=4000):
    """Score every chunk of every file, then reduce each file to its aggregate features.

    Returns (features dict of arrays, benign-chunk histogram, truncation count).
    The chunk scores themselves are reduced and dropped: the point of the aggregate
    features is that everything in the rule family is a function of them, so they are
    what gets written out.
    """
    n = len(rows)
    pieces = [[] for _ in range(n)]
    spans, buf = [], []
    truncated = 0

    def flush():
        if not buf:
            return
        X = vec.transform(buf)
        p = model.predict_proba(X)[:, 1]
        cur = 0
        for idx, cnt in spans:
            pieces[idx].append(p[cur:cur + cnt].astype(np.float64))
            cur += cnt
        buf.clear(); spans.clear()

    for i, r in enumerate(rows):
        cs, t = file_chunks(r["path"], size, stride, max_chunks)
        truncated += int(t)
        if cs:
            buf.extend(cs); spans.append((i, len(cs)))
        if len(buf) >= batch:
            flush()
    flush()

    F = {"n_chunks": np.zeros(n, dtype=np.int64),
         "mean": np.zeros(n), "max": np.zeros(n), "std": np.zeros(n)}
    for q in QUANTS:
        F[f"q{int(round(q * 100))}"] = np.zeros(n)
    for k in TOPKS:
        F[f"top{k}"] = np.zeros(n)
    counts = np.zeros((n, len(COUNT_GRID)), dtype=np.int32)
    hist = np.zeros(HIST_BINS + 1, dtype=np.int64)

    for i in range(n):
        p = np.concatenate(pieces[i]) if pieces[i] else np.zeros(1)
        pieces[i] = None                                   # free as we go
        asc = np.sort(p)
        F["n_chunks"][i] = len(p)
        F["mean"][i] = p.mean()
        F["max"][i] = asc[-1]
        F["std"][i] = p.std()
        for q in QUANTS:
            F[f"q{int(round(q * 100))}"][i] = np.quantile(asc, q)
        for k in TOPKS:
            F[f"top{k}"][i] = asc[-k:].mean() if len(asc) >= k else asc.mean()
        # count of chunks with score >= t, exact for every t on COUNT_GRID
        counts[i] = len(asc) - np.searchsorted(asc, COUNT_GRID, side="left")
        if hist_mask is not None and hist_mask[i]:
            hist += np.bincount(np.clip((p * HIST_BINS).astype(np.int64), 0, HIST_BINS),
                                minlength=HIST_BINS + 1)
    return F, counts, hist, truncated


def hist_quantile(hist, q):
    """The q-quantile of the chunk scores summarised in `hist`, to 1/HIST_BINS."""
    tot = hist.sum()
    if tot == 0:
        return float("nan")
    cdf = np.cumsum(hist) / tot
    j = int(np.searchsorted(cdf, q, side="left"))
    return float(min(j, HIST_BINS) / HIST_BINS)


def grid_index(t):
    """Index of the smallest COUNT_GRID value >= t, so a count is never overstated."""
    j = int(np.searchsorted(COUNT_GRID, t - 1e-9, side="left"))
    return min(j, len(COUNT_GRID) - 1)


# ----------------------------------------------------------------- the rule family
def build_rule_scores(F, counts, t_bq99, t_bq999):
    """One per-file score per rule, all derived from the SAME chunk scores.

    Returns an ordered dict name -> (score array, free parameter count, note).
    """
    S = {}
    S["mean"] = (F["mean"], 1, "Wang et al. 2025 average pooling: the reference low end")
    S["max"] = (F["max"], 1, "Kwon 2026, PUBLISHED: file score = highest chunk score. "
                             "The incumbent this experiment is measured against.")
    for q in QUANTS:
        key = f"q{int(round(q * 100))}"
        note = "exactly max" if q == 1.0 else ("the median" if q == 0.5 else "")
        S[f"quantile_{q:.2f}"] = (F[key], 1, note)
    for k in TOPKS:
        note = "exactly max" if k == 1 else "mean of the k highest chunk scores"
        S[f"top{k}_mean"] = (F[f"top{k}"], 1, note)
    i99, i999 = grid_index(t_bq99), grid_index(t_bq999)
    c99 = counts[:, i99].astype(np.float64)
    c999 = counts[:, i999].astype(np.float64)
    S["count_above_bq99"] = (c99, 1,
        f"chunks scoring >= {COUNT_GRID[i99]:.4f}, the 0.99 quantile of BENIGN "
        f"calibration chunk scores. t is fixed by that rule, not tuned for recall, so "
        f"this rule costs one threshold like the others.")
    S["count_above_bq999"] = (c999, 1,
        f"chunks scoring >= {COUNT_GRID[i999]:.4f}, the 0.999 quantile of benign "
        f"calibration chunk scores")
    S["frac_above_bq99"] = (c99 / np.maximum(F["n_chunks"], 1), 1,
        "the same count divided by the file's chunk count: a length-normalised count")
    S["max_minus_mean"] = (F["max"] - F["mean"], 1,
        "how far the top chunk stands above the file's own average - a shape feature, "
        "included because it is what a dilution signature looks like")
    return S


SINGLE_PARAM_RULES = None    # filled at run time from build_rule_scores' key order


# ----------------------------------------------------------------- selection objective
def pick_threshold(pos, neg, budget):
    """The ONE selection objective every rule gets.

    Maximise calibration recall subject to calibration FPR <= budget; break ties by
    lower FPR, then by the larger threshold. Candidates are every distinct calibration
    score plus a sentinel above all of them (the threshold that alerts on nothing), so
    no rule is given a finer grid than another.
    """
    pos = np.asarray(pos, dtype=float); neg = np.asarray(neg, dtype=float)
    hi = max(float(neg.max()) if len(neg) else 0.0, float(pos.max()) if len(pos) else 0.0)
    cands = np.unique(np.concatenate([neg, pos, [hi + 1.0]]))
    ns, ps = np.sort(neg), np.sort(pos)
    fp = len(ns) - np.searchsorted(ns, cands, side="left")
    tp = len(ps) - np.searchsorted(ps, cands, side="left")
    fpr = fp / max(len(ns), 1)
    rec = tp / max(len(ps), 1)
    ok = fpr <= budget + EPS
    if not ok.any():
        return float(cands[-1]), 0.0, 0.0, True
    best_rec = rec[ok].max()
    m = ok & (rec >= best_rec - EPS)
    best_fpr = fpr[m].min()
    m2 = m & (fpr <= best_fpr + EPS)
    j = int(np.where(m2)[0][-1])
    return float(cands[j]), float(fpr[j]), float(rec[j]), False


def pick_count_2d(counts_c, is_pos_c, budget):
    """The GENEROUS count rule: both the chunk threshold t and the count cutoff tuned
    on calibration. Two free parameters, and reported as such."""
    best = None
    for j in range(0, len(COUNT_GRID), 5):            # every 0.005 of chunk score
        c = counts_c[:, j].astype(np.float64)
        t, fpr, rec, degen = pick_threshold(c[is_pos_c], c[~is_pos_c], budget)
        if degen:
            continue
        key = (rec, -fpr)
        if best is None or key > best[0]:
            best = (key, j, t, fpr, rec)
    if best is None:
        return 0, float("inf"), 0.0, 0.0, True
    _k, j, t, fpr, rec = best
    return j, t, fpr, rec, False


# ----------------------------------------------------------------- evaluation
def evaluate(alert, is_pos, is_dil, achieved):
    benign = ~is_pos
    plain = is_pos & ~is_dil
    out = dict(
        fpr=float(alert[benign].mean()) if benign.any() else None,
        false_positives=int(alert[benign].sum()), n_benign=int(benign.sum()),
        recall_all=float(alert[is_pos].mean()) if is_pos.any() else None,
        recall_plain=float(alert[plain].mean()) if plain.any() else None,
        recall_diluted=float(alert[is_dil].mean()) if is_dil.any() else None,
        n_plain=int(plain.sum()), n_diluted=int(is_dil.sum()),
    )
    bins = ratio_bins(achieved)
    per = {}
    for i, lab in enumerate(RATIO_LABELS):
        m = is_dil & (bins == i)
        if m.any():
            per[lab] = dict(n=int(m.sum()), recall=float(alert[m].mean()))
    out["recall_diluted_by_achieved_ratio"] = per
    return out


def auc_populations(score, is_pos, is_dil, is_dil_host=None):
    """ROC AUC and average precision on four populations.

    `diluted_positives` EXCLUDES the benign host files the diluted positives were built
    from. A diluted positive IS one of those hosts with a shell injected; at the achieved
    ratios of 0.1-2% the two files are 98-99.9% byte-identical and sit on opposite sides
    of the label, so keeping both bounds this AUC near chance BY CONSTRUCTION whatever
    the rule does. Ported from 04_run_experiment.py on 13 Oct 2026; before that this
    script used the old definition for every chunk size, which understated the whole
    sweep.

    `diluted_positives_with_hosts` keeps the old definition so every previously reported
    figure stays traceable. The two do NOT share a negative set, so `n_negatives` and
    `n_hosts_excluded` are reported per population and must not be compared across them.
    """
    if is_dil_host is None:
        is_dil_host = np.zeros(len(is_pos), dtype=bool)
    dil_sub = (~is_pos) | (is_pos & is_dil)
    subs = {
        "all_positives": np.ones(len(is_pos), dtype=bool),
        "plain_positives": (~is_pos) | (is_pos & ~is_dil),
        "diluted_positives": dil_sub & ~is_dil_host,
        "diluted_positives_with_hosts": dil_sub,
    }
    out = {}
    for name, m in subs.items():
        y = is_pos[m]
        out[name] = dict(roc_auc=roc_auc(y, score[m]),
                         average_precision=average_precision(y, score[m]),
                         n_positives=int(y.sum()), n_negatives=int((~y).sum()),
                         n_hosts_excluded=(int((is_dil_host & dil_sub).sum())
                                           if name == "diluted_positives" else 0))
    return out


# ----------------------------------------------------------------- learned combiner
COMBINER_FEATURES = ("mean", "max", "top3_mean", "q90", "count_above_t",
                     "n_chunks", "max_minus_mean", "log_size")
# n_chunks and log(size) are both file LENGTH. The dilution generator only injects into
# hosts of at least --min-host-bytes, so in this corpus a diluted positive is
# systematically longer than an average benign file: a combiner given length can score
# well by learning the generator rather than the payload. The variant below drops both
# length features, and the pair is reported together so the size shortcut is visible
# rather than assumed away.
COMBINER_VARIANTS = {
    "all_features": COMBINER_FEATURES,
    "no_size_features": ("mean", "max", "top3_mean", "q90", "count_above_t",
                         "max_minus_mean"),
}


def combiner_matrix(F, counts, t_bq99, sizes):
    j = grid_index(t_bq99)
    return np.column_stack([
        F["mean"], F["max"], F["top3"], F["q90"],
        counts[:, j].astype(np.float64), F["n_chunks"].astype(np.float64),
        F["max"] - F["mean"], np.log(np.maximum(sizes, 1.0)),
    ])


def fit_logreg_dense(X, y, l2=1.0, balanced=True, iters=100, tol=1e-9):
    """L2-penalised logistic regression by Newton-Raphson on the exact objective.

    Eight features and a bias, so the Newton step is a 9x9 solve and converges to the
    optimum of the same convex objective scikit-learn's LogisticRegression minimises.
    The bias is not penalised. Written out rather than imported so that this part of
    the experiment gives the same answer on a machine where scikit-learn cannot be
    installed; `--combiner sklearn` uses scikit-learn instead when it is available, and
    the results file records which was used.
    """
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    A = np.column_stack([X, np.ones(n)])
    if balanced:
        npos, nneg = float((y == 1).sum()), float((y != 1).sum())
        wpos = (npos + nneg) / (2 * npos) if npos else 1.0
        wneg = (npos + nneg) / (2 * nneg) if nneg else 1.0
        sw = np.where(y == 1, wpos, wneg)
    else:
        sw = np.ones(n)
    reg = np.full(d + 1, l2); reg[-1] = 0.0
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = A @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = A.T @ (sw * (p - y)) + reg * w
        s = sw * p * (1.0 - p) + 1e-10
        H = A.T @ (A * s[:, None]) + np.diag(reg) + 1e-9 * np.eye(d + 1)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w_new = w - step
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w[:-1], float(w[-1]), sw


def combiner_scores(X, w, b):
    return X @ w + b


# ----------------------------------------------------------------- small helpers
def agg(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None
    return dict(mean=float(np.mean(v)),
                sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, values=v)


def paired_t(d):
    """Paired t statistic and two-sided p, computed without scipy (it is not
    installable on every machine this kit runs on). p comes from the Student-t
    survival function via its incomplete-beta form in math.lgamma."""
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], dtype=float)
    n = len(d)
    if n < 2:
        return dict(n=n, note="fewer than two paired seeds")
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        return dict(n=n, t=None, p=None, note="no variation across seeds",
                    mean_difference=float(d.mean()))
    t = float(d.mean() / (sd / math.sqrt(n)))
    df = n - 1
    x = df / (df + t * t)

    def betainc(a, b, x, terms=2000):            # regularised incomplete beta, series
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
        s, term = 1.0, 1.0
        for k in range(terms):
            term *= (a + b + k) * x / (a + 1 + k)
            s += term
            if abs(term) < 1e-14 * abs(s):
                break
        return min(1.0, max(0.0, front * s))

    p = betainc(df / 2.0, 0.5, x)
    return dict(n=n, t=t, p=float(p), mean_difference=float(d.mean()), sd=sd,
                test="paired t over seeds, two-sided, no-scipy implementation")


def fmt(x, nd=4):
    return "   n/a" if x is None else f"{x:.{nd}f}"


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=3,
                    help="independent runs. Default 3: enough for a variance estimate, "
                         "which is the floor 04's README sets.")
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--budgets", nargs="*", type=float, default=[0.02, 0.01, 0.005],
                    help="the same calibration false-positive budgets as 04")
    ap.add_argument("--chunk-sizes", nargs="*", type=int,
                    default=[128, 256, 512, 1024, 2048],
                    help="chunk sizes to sweep; stride is always size/2")
    ap.add_argument("--max-chunks-at-512", type=int, default=2048,
                    help="chunk cap AT 512 bytes. Every other size gets "
                         "round(this * 512 / size), so all sizes see the same number "
                         "of BYTES of each file. Without this the small sizes would be "
                         "truncated first and the sweep would measure truncation.")
    ap.add_argument("--features", type=int, default=18, help="hashing features = 2**n")
    ap.add_argument("--ngram", nargs=2, type=int, default=[3, 5])
    ap.add_argument("--train-fraction", type=float, default=0.85)
    ap.add_argument("--max-train-chunks", type=int, default=120_000,
                    help="cap on training chunks, so training cost does not itself "
                         "change with chunk size")
    ap.add_argument("--combiner-l2", type=float, default=1.0)
    ap.add_argument("--permutations", type=int, default=20,
                    help="label-shuffled refits of the learned combiner: the chance "
                         "control for its extra free parameters")
    ap.add_argument("--scorer", choices=("sklearn", "numpy"), default="sklearn",
                    help="the chunk scorer, same switch as 04. 'sklearn' is the "
                         "design (HashingVectorizer + liblinear). 'numpy' is 04's "
                         "declared STAND-IN for machines where scikit-learn cannot be "
                         "installed; its rates are labelled not-a-measurement "
                         "everywhere they are written.")
    ap.add_argument("--combiner", choices=("auto", "sklearn", "newton"), default="auto",
                    help="solver for Part 3. 'newton' optimises the same L2 logistic "
                         "objective directly and needs no scikit-learn.")
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE RUN: cap each split at this many files. Labelled "
                         "not-a-result everywhere.")
    ap.add_argument("--tag", default="family")
    ap.add_argument("--no-jsonl", action="store_true",
                    help="skip the per-file feature dump (it is the big output)")
    a = ap.parse_args()

    if a.scorer == "sklearn" and M4.sklearn is None:
        raise SystemExit(
            "scikit-learn is not importable here (%s) and it is what this kit's chunk "
            "scorer is. Install it, or rerun with --scorer numpy, which uses the "
            "stand-in in scripts/numpy_scorer.py and labels every number it produces "
            "as not a measurement." % M4.SKLEARN_IMPORT_ERROR)
    stand_in = a.scorer == "numpy"
    if stand_in:
        log("*** --scorer numpy: " + numpy_scorer.DESCRIPTION)
    smoke = bool(a.limit)

    # ------------------------------------------------------------ data
    index = read_jsonl(DATA / "index.jsonl")
    dil_path = DATA / "dilution.jsonl"
    dil = read_jsonl(dil_path) if dil_path.exists() else []
    for d in dil:
        d.setdefault("kind", "diluted")
    for r in index:
        r.setdefault("kind", "plain")
    train_rows = [r for r in index if r["split"] == "train"]
    calib_rows = ([r for r in index if r["split"] == "calib"]
                  + [d for d in dil if d["split"] == "calib"])
    test_rows = ([r for r in index if r["split"] == "test"]
                 + [d for d in dil if d["split"] == "test"])

    if smoke:
        def cap(rows, n):
            rng = np.random.default_rng(a.seed_base)
            pos = [r for r in rows if r["label"] == 1]
            neg = [r for r in rows if r["label"] != 1]
            dilp = [r for r in pos if r.get("kind") == "diluted"]
            plain = [r for r in pos if r.get("kind") != "diluted"]
            want = max(2, n // 3)
            keep_d = dilp[:min(len(dilp), max(1, want // 2))]
            keep_p = ([plain[i] for i in rng.permutation(len(plain))[:max(1, want - len(keep_d))]]
                      if plain else [])
            kp = keep_d + keep_p
            kn = [neg[i] for i in rng.permutation(len(neg))[:max(1, n - len(kp))]]
            return kp + kn
        before = (len(train_rows), len(calib_rows), len(test_rows))
        train_rows = cap(train_rows, a.limit)
        calib_rows = cap(calib_rows, a.limit)
        test_rows = cap(test_rows, a.limit)
        log(f"*** SMOKE RUN (--limit {a.limit}): splits cut from {before} to "
            f"{(len(train_rows), len(calib_rows), len(test_rows))}. Not a result.")

    def arrays(rows):
        # The benign files that diluted positives in THIS split were built from, matched
        # on the stored host_path, so they can be held out of the diluted-only AUC.
        hosts = {r["host_path"] for r in rows
                 if r.get("kind") == "diluted" and r.get("host_path")}
        is_pos = np.array([r["label"] == 1 for r in rows])
        is_dil = np.array([r.get("kind") == "diluted" for r in rows])
        is_host = np.array([(r["label"] != 1) and (r.get("path") in hosts)
                            for r in rows])
        n_dil = int(is_dil.sum())
        if n_dil and not is_host.any():
            log(f"  WARNING: {n_dil} diluted file(s) in this split but NOT ONE of their "
                f"host_path values matched a benign row's path, so no host could be held "
                f"out of the diluted-only AUC. Fix that before trusting "
                f"diluted_positives.")
        return (is_pos, is_dil, is_host,
                np.array([achieved_ratio_of(r) for r in rows], dtype=float),
                np.array([float(r.get("size") or 0) for r in rows]))

    c_pos, c_dil, c_host, c_ach, c_size = arrays(calib_rows)
    t_pos, t_dil, t_host, t_ach, t_size = arrays(test_rows)
    for name, rows, pos in (("train", train_rows, None), ("calib", calib_rows, c_pos),
                            ("test", test_rows, t_pos)):
        npos = sum(r["label"] for r in rows)
        log(f"{name:6s} {len(rows):6d} files  ({npos} malicious, {len(rows) - npos} benign"
            + (f", {int(pos[np.array([r.get('kind') == 'diluted' for r in rows])].sum()) if len(rows) else 0} diluted)"
               if pos is not None else ")"))
        if not rows or npos in (0, len(rows)):
            sys.exit(f"{name} split needs both classes")

    config = dict(vars(a))
    config.update(scorer_is_a_stand_in=stand_in,
                  scorer_description=(numpy_scorer.DESCRIPTION if stand_in
                                      else "scikit-learn HashingVectorizer + liblinear"),
                  sklearn_version=(M4.sklearn.__version__ if M4.sklearn else None),
                  sklearn_import_error=M4.SKLEARN_IMPORT_ERROR,
                  numpy_version=np.__version__, platform=platform.platform(),
                  smoke_run=smoke,
                  imported_from="04_run_experiment.py (file_chunks, roc_auc, "
                                "average_precision, achieved_ratio_of, ratio_bins, "
                                "pick_2d) - 04 itself is unmodified",
                  selection_objective="on CALIBRATION: maximise recall subject to "
                                      "FPR <= budget; ties by lower FPR, then larger "
                                      "threshold. Identical for every rule.",
                  framing="max is Kwon 2026's PUBLISHED rule and is the incumbent "
                          "here, not a contribution of this project.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    # One checkpoint per (chunk size, seed). The sweep takes long enough that a run can
    # be interrupted - and on some hosts a long job is killed outright - so each
    # (size, seed) is written as soon as it is finished and reused on the next
    # invocation. Re-running the same --tag resumes instead of rescoring. Delete
    # results/.aggfam_<tag>/ to force a clean run.
    CK = RESULTS / f".aggfam_{a.tag}"
    CK.mkdir(parents=True, exist_ok=True)
    jsonl_path = None if a.no_jsonl else RESULTS / f"aggregation_scores_{a.tag}.jsonl"
    log(f"checkpoints -> {CK}  (resume is automatic; delete the folder to force a "
        f"clean run)")

    per_size = {}
    t_start = time.time()

    for size in a.chunk_sizes:
        stride = max(1, size // 2)
        max_chunks = max(1, int(round(a.max_chunks_at_512 * 512 / size)))
        log(f"\n================ chunk size {size} bytes, stride {stride}, "
            f"max_chunks {max_chunks} ({max_chunks * stride} bytes of coverage) "
            f"================")
        seeds_out = []
        for k in range(a.seeds):
            seed = a.seed_base + k
            ck_j = CK / f"s{size}_seed{seed}.json"
            ck_l = CK / f"s{size}_seed{seed}.jsonl"
            if ck_j.exists() and (a.no_jsonl or ck_l.exists()):
                try:
                    seeds_out.append(json.loads(ck_j.read_text()))
                    log(f"seed {seed}: reusing checkpoint {ck_j.name}")
                    continue
                except (json.JSONDecodeError, OSError):
                    log(f"seed {seed}: checkpoint {ck_j.name} unreadable, recomputing")
            t0 = time.time()
            rng = np.random.default_rng(seed)
            sel = rng.random(len(train_rows)) < a.train_fraction
            rows_k = [r for r, s in zip(train_rows, sel) if s] or train_rows
            texts, owner, trunc = M4.build_chunks(rows_k, size, stride, max_chunks)
            y = np.array([rows_k[o]["label"] for o in owner])
            if len(texts) > a.max_train_chunks:
                keep = rng.choice(len(texts), a.max_train_chunks, replace=False)
                texts = [texts[i] for i in keep]; y = y[keep]
            if stand_in:
                vec = numpy_scorer.HashedNgrams(n_features=2 ** a.features,
                                                ngram_range=tuple(a.ngram))
                model = numpy_scorer.LogisticSGD(random_state=seed)
            else:
                vec = M4.HashingVectorizer(analyzer="char", ngram_range=tuple(a.ngram),
                                           n_features=2 ** a.features,
                                           alternate_sign=False, norm="l2")
                model = M4.LogisticRegression(max_iter=2000, C=1.0,
                                              class_weight="balanced",
                                              solver="liblinear", random_state=seed)
            log(f"seed {seed}: {len(rows_k)} train files -> {len(y)} chunks "
                f"({int(y.sum())} malicious)")
            model.fit(vec.transform(texts), y)
            del texts

            Fc, Cc, hist_c, trunc_c = file_features(calib_rows, model, vec, size, stride,
                                                    max_chunks, hist_mask=~c_pos)
            Ft, Ct, _h, trunc_t = file_features(test_rows, model, vec, size, stride,
                                                max_chunks)
            t_bq99 = hist_quantile(hist_c, 0.99)
            t_bq999 = hist_quantile(hist_c, 0.999)

            Sc = build_rule_scores(Fc, Cc, t_bq99, t_bq999)
            St = build_rule_scores(Ft, Ct, t_bq99, t_bq999)
            global SINGLE_PARAM_RULES
            SINGLE_PARAM_RULES = list(Sc.keys())

            rec = dict(seed=seed, n_train_files=len(rows_k), n_train_chunks=int(len(y)),
                       benign_calib_chunk_quantiles=dict(
                           q0_99=t_bq99, q0_999=t_bq999,
                           n_benign_chunks=int(hist_c.sum()),
                           note="t for the count rules, from BENIGN calibration chunk "
                                "scores only - no labels of positives are used to set it"),
                       truncated_files=dict(train=int(trunc), calib=int(trunc_c),
                                            test=int(trunc_t)),
                       mean_chunks_per_file=dict(
                           calib=float(Fc["n_chunks"].mean()),
                           test=float(Ft["n_chunks"].mean())),
                       auc={}, budgets={}, identities={})

            # sanity: the family must contain max exactly at both ends
            for alias in ("quantile_1.00", "top1_mean"):
                same = bool(np.array_equal(St[alias][0], St["max"][0]))
                rec["identities"][alias] = dict(identical_to_max=same)
                if not same:
                    log(f"  WARNING {alias} should be bit-identical to max and is not")

            for name, (s, _np_, _n) in St.items():
                rec["auc"][name] = auc_populations(s, t_pos, t_dil, t_host)

            for B in a.budgets:
                key = f"{B:g}"
                res = {}
                for name in SINGLE_PARAM_RULES:
                    sc, _p, _n = Sc[name]
                    st, _p2, _n2 = St[name]
                    thr, cfpr, crec, degen = pick_threshold(sc[c_pos], sc[~c_pos], B)
                    e = evaluate(st >= thr, t_pos, t_dil, t_ach)
                    e.update(threshold=thr, calibration=dict(fpr=cfpr, recall=crec),
                             free_parameters=1, degenerate=degen)
                    res[name] = e
                # --- diagnostic: recall at MAX's OWN test FPR.
                # The threshold above meets the budget on CALIBRATION; what each rule
                # then costs on TEST is its own business, and the table shows the two
                # drifting apart. So the same comparison is also made at a matched
                # operating point, by reading each rule's test ROC at max's measured
                # test FPR. This is an ORACLE number - it uses the test set to place
                # the threshold - so it is NOT a selectable operating point and never
                # feeds the verdict. It is here to separate "ranks diluted files
                # better" from "runs at a looser threshold".
                f0 = res["max"]["fpr"]
                if f0 is not None:
                    for name in SINGLE_PARAM_RULES:
                        st, _p, _n = St[name]
                        _t, _f, rd, _d = pick_threshold(st[t_dil], st[~t_pos], f0)
                        res[name]["recall_diluted_at_max_test_fpr"] = float(rd)
                    res["matched_test_fpr"] = dict(
                        target_fpr=f0,
                        note="ORACLE diagnostic: thresholds placed on the TEST split to "
                             "match max's test FPR. Not selectable, not in the verdict.")
                # --- reference rules with MORE than one free parameter
                tm, tx, degen2 = pick_2d(Fc["mean"][~c_pos], Fc["max"][~c_pos],
                                         Fc["mean"][c_pos], Fc["max"][c_pos],
                                         c_dil[c_pos], B)
                e = evaluate((Ft["mean"] >= tm) | (Ft["max"] >= tx), t_pos, t_dil, t_ach)
                e.update(threshold=dict(t_mean=tm, t_max=tx), free_parameters=2,
                         degenerate=bool(degen2),
                         note="the paper's proposed rule, picked by 04's own pick_2d. "
                              "TWO thresholds, so it is not comparable to the "
                              "single-parameter rules on equal terms.")
                res["mean_or_max"] = e
                j, ct, cfpr, crec, degenc = pick_count_2d(Cc, c_pos, B)
                e = evaluate(Ct[:, j].astype(np.float64) >= ct, t_pos, t_dil, t_ach)
                e.update(threshold=dict(chunk_t=float(COUNT_GRID[j]), count_cutoff=ct),
                         calibration=dict(fpr=cfpr, recall=crec), free_parameters=2,
                         degenerate=bool(degenc),
                         note="the GENEROUS count rule: both the chunk threshold and "
                              "the count cutoff tuned on calibration. TWO free "
                              "parameters.")
                res["count_above_t_tuned"] = e
                rec["budgets"][key] = res
            seeds_out.append(rec)
            log(f"  seed {seed} done in {time.time() - t0:.1f}s "
                f"(mean chunks/file: calib {Fc['n_chunks'].mean():.0f}, "
                f"test {Ft['n_chunks'].mean():.0f})")

            if not a.no_jsonl:
                rows_out = []
                for tagsplit, rows, F, C in (("calib", calib_rows, Fc, Cc),
                                             ("test", test_rows, Ft, Ct)):
                    i99, i999 = grid_index(t_bq99), grid_index(t_bq999)
                    for i, r in enumerate(rows):
                        ach = achieved_ratio_of(r)
                        row = dict(seed=int(seed), chunk_bytes=int(size),
                                   stride=int(stride), split=tagsplit, path=r["path"],
                                   source_id=r.get("source_id"), group=r.get("group"),
                                   label=int(r["label"]), kind=r.get("kind", "plain"),
                                   lang=r.get("lang"), size=r.get("size"),
                                   inject_ratio=r.get("inject_ratio"),
                                   achieved_ratio=(None if not np.isfinite(ach)
                                                   else float(ach)),
                                   injected_bytes=r.get("injected_bytes"),
                                   n_chunks=int(F["n_chunks"][i]),
                                   mean=float(F["mean"][i]), max=float(F["max"][i]),
                                   std=float(F["std"][i]),
                                   count_above_bq99=int(C[i, i99]),
                                   count_above_bq999=int(C[i, i999]),
                                   t_bq99=t_bq99, t_bq999=t_bq999)
                        for q in QUANTS:
                            kq = f"q{int(round(q * 100))}"
                            row[kq] = float(F[kq][i])
                        for kk in TOPKS:
                            row[f"top{kk}_mean"] = float(F[f"top{kk}"][i])
                        rows_out.append(row)
                write_jsonl(ck_l, rows_out)

            # ------------------------------------------------ Part 3, learned combiner
            Xc_all = combiner_matrix(Fc, Cc, t_bq99, c_size)
            Xt_all = combiner_matrix(Ft, Ct, t_bq99, t_size)
            for vname, feats in COMBINER_VARIANTS.items():
                cols = [COMBINER_FEATURES.index(f) for f in feats]
                Xc, Xt = Xc_all[:, cols], Xt_all[:, cols]
                mu, sd = Xc.mean(0), Xc.std(0)
                sd[sd == 0] = 1.0
                Zc, Zt = (Xc - mu) / sd, (Xt - mu) / sd
                solver = a.combiner
                if solver == "auto":
                    solver = "sklearn" if M4.sklearn is not None else "newton"

                def _fit(Z, yv):
                    if solver == "sklearn":
                        lm = M4.LogisticRegression(
                            max_iter=5000, C=1.0 / max(a.combiner_l2, EPS),
                            class_weight="balanced", solver="lbfgs")
                        lm.fit(Z, yv)
                        return lm.coef_[0].copy(), float(lm.intercept_[0])
                    w_, b_, _s = fit_logreg_dense(Z, yv, l2=a.combiner_l2)
                    return w_, b_

                w, b = _fit(Zc, c_pos.astype(int))
                dc, dt = combiner_scores(Zc, w, b), combiner_scores(Zt, w, b)
                comb = dict(variant=vname, solver=solver, features=list(feats),
                            l2=a.combiner_l2, class_weight="balanced",
                            standardised_weights={f: float(x) for f, x in zip(feats, w)},
                            raw_weights={f: float(x / sv) for f, x, sv
                                         in zip(feats, w, sd)},
                            intercept=b, free_parameters=len(feats) + 1,
                            fairness_note=(
                                f"{len(feats) + 1} free parameters against ONE for "
                                f"every threshold rule, and it is FIT on calibration "
                                f"and THRESHOLDED on calibration, so it is the most "
                                f"favoured competitor in this file. The permutation "
                                f"control is what that parameter count buys by chance."),
                            auc=auc_populations(dt, t_pos, t_dil, t_host), budgets={})
                for B in a.budgets:
                    thr, cfpr, crec, degen = pick_threshold(dc[c_pos], dc[~c_pos], B)
                    e = evaluate(dt >= thr, t_pos, t_dil, t_ach)
                    e.update(threshold=thr, calibration=dict(fpr=cfpr, recall=crec),
                             degenerate=degen)
                    comb["budgets"][f"{B:g}"] = e
                perms = {f"{B:g}": [] for B in a.budgets}
                perm_auc = []
                prng = np.random.default_rng(seed * 1000 + 7 + len(feats))
                for _ in range(a.permutations):
                    yp = prng.permutation(c_pos.astype(int))
                    wp, bp = _fit(Zc, yp)
                    dcp, dtp = combiner_scores(Zc, wp, bp), combiner_scores(Zt, wp, bp)
                    perm_auc.append(auc_populations(dtp, t_pos, t_dil, t_host)
                                    ["diluted_positives"]["roc_auc"])
                    for B in a.budgets:
                        thr, _f, _r, _d = pick_threshold(dcp[c_pos], dcp[~c_pos], B)
                        perms[f"{B:g}"].append(
                            evaluate(dtp >= thr, t_pos, t_dil, t_ach)["recall_diluted"])
                comb["permutation_control"] = dict(
                    n_permutations=a.permutations,
                    what_was_permuted="the calibration LABELS, before refitting the "
                                      "combiner. The threshold is then chosen on "
                                      "calibration with the TRUE labels by the same "
                                      "objective every rule gets, so only the weights "
                                      "are chance.",
                    diluted_roc_auc=agg(perm_auc),
                    diluted_recall={k: agg(v) for k, v in perms.items()},
                    diluted_recall_p95={
                        k: (float(np.percentile([x for x in v if x is not None], 95))
                            if any(x is not None for x in v) else None)
                        for k, v in perms.items()})
                key = ("learned_combiner" if vname == "all_features"
                       else f"learned_combiner_{vname}")
                rec[key] = comb
                log(f"  combiner [{vname}] ({solver}): "
                    + ", ".join(f"{f}={x:+.3f}" for f, x in zip(feats, w)))
            save_json(ck_j, rec)

        per_size[str(size)] = dict(stride=stride, max_chunks=max_chunks,
                                   bytes_of_coverage=max_chunks * stride,
                                   seeds=seeds_out)

    # ------------------------------------------------------------ summarise
    sizes = [str(s) for s in a.chunk_sizes]
    budget_keys = [f"{B:g}" for B in a.budgets]
    # the budget block also carries the matched-FPR diagnostic, which is not a rule
    _b0 = per_size[sizes[0]]["seeds"][0]["budgets"][budget_keys[0]]
    rule_order = [r for r, v in _b0.items()
                  if isinstance(v, dict) and "free_parameters" in v]
    one_param = [r for r in rule_order if _b0[r]["free_parameters"] == 1]
    multi_param = [r for r in rule_order if r not in one_param]
    del _b0

    # AUC is only defined for rules that ARE one score. mean_or_max is a two-threshold
    # decision, and count_above_t_tuned's score changes with the budget, so neither has
    # a single ranking to measure threshold-free - said here rather than silently
    # dropping them from the table.
    auc_rules = list(per_size[sizes[0]]["seeds"][0]["auc"].keys())
    summary = {}
    for sz in sizes:
        seeds_out = per_size[sz]["seeds"]
        blk = {"budgets": {}, "auc": {}}
        for name in auc_rules:
            blk["auc"][name] = {
                pop: {m: agg([s["auc"][name][pop][m] for s in seeds_out])
                      for m in ("roc_auc", "average_precision")}
                for pop in POPULATIONS}
        for bk in budget_keys:
            rows = {}
            max_vals = [s["budgets"][bk]["max"]["recall_diluted"] for s in seeds_out]
            for name in rule_order:
                vals = {f: agg([s["budgets"][bk][name].get(f) for s in seeds_out])
                        for f in ("fpr", "recall_all", "recall_plain", "recall_diluted",
                                  "recall_diluted_at_max_test_fpr")}
                d = [(s["budgets"][bk][name]["recall_diluted"] - mv)
                     for s, mv in zip(seeds_out, max_vals)
                     if s["budgets"][bk][name]["recall_diluted"] is not None
                     and mv is not None]
                vals["free_parameters"] = \
                    seeds_out[0]["budgets"][bk][name]["free_parameters"]
                vals["diluted_recall_gain_over_max"] = agg(d)
                vals["paired_t_vs_max"] = paired_t(d)
                vals["degenerate_seeds"] = [s["seed"] for s in seeds_out
                                            if s["budgets"][bk][name].get("degenerate")]
                by_bin = {}
                for lab in RATIO_LABELS:
                    v = [s["budgets"][bk][name]["recall_diluted_by_achieved_ratio"]
                         .get(lab, {}).get("recall") for s in seeds_out]
                    g = agg(v)
                    if g is not None:
                        g["n"] = (seeds_out[0]["budgets"][bk][name]
                                  ["recall_diluted_by_achieved_ratio"][lab]["n"])
                        by_bin[lab] = g
                vals["recall_diluted_by_achieved_ratio"] = by_bin
                rows[name] = vals
            blk["budgets"][bk] = rows
        for ckey in ("learned_combiner", "learned_combiner_no_size_features"):
            if ckey not in seeds_out[0]:
                continue
            feats = seeds_out[0][ckey]["features"]
            blk[ckey] = {
                "variant": seeds_out[0][ckey].get("variant"),
                "features": feats,
                "standardised_weights": {
                    f: agg([sd_[ckey]["standardised_weights"][f] for sd_ in seeds_out])
                    for f in feats},
                "intercept": agg([sd_[ckey]["intercept"] for sd_ in seeds_out]),
                "solver": seeds_out[0][ckey]["solver"],
                "free_parameters": seeds_out[0][ckey]["free_parameters"],
                "fairness_note": seeds_out[0][ckey]["fairness_note"],
                "auc": {pop: {m: agg([sd_[ckey]["auc"][pop][m] for sd_ in seeds_out])
                              for m in ("roc_auc", "average_precision")}
                        for pop in POPULATIONS},
                "budgets": {bk: {f: agg([sd_[ckey]["budgets"][bk][f]
                                         for sd_ in seeds_out])
                                 for f in ("fpr", "recall_all", "recall_plain",
                                           "recall_diluted")}
                            for bk in budget_keys},
                "diluted_recall_gain_over_max": {
                    bk: agg([sd_[ckey]["budgets"][bk]["recall_diluted"]
                             - sd_["budgets"][bk]["max"]["recall_diluted"]
                             for sd_ in seeds_out
                             if sd_[ckey]["budgets"][bk]["recall_diluted"] is not None
                             and sd_["budgets"][bk]["max"]["recall_diluted"] is not None])
                    for bk in budget_keys},
                "permutation_control": {
                    "n_permutations": a.permutations,
                    "what_was_permuted": seeds_out[0][ckey]["permutation_control"]
                                         ["what_was_permuted"],
                    "diluted_roc_auc": agg([v for sd_ in seeds_out for v in
                                            (sd_[ckey]["permutation_control"]
                                             ["diluted_roc_auc"] or {}).get("values", [])]),
                    "diluted_recall": {
                        bk: agg([v for sd_ in seeds_out for v in
                                 (sd_[ckey]["permutation_control"]["diluted_recall"][bk]
                                  or {}).get("values", [])])
                        for bk in budget_keys},
                },
            }
        summary[sz] = blk

    # the sweep view: diluted recall and diluted AUC against chunk size, per rule
    sweep = {}
    for name in rule_order:
        sweep[name] = {}
        for sz in sizes:
            A = summary[sz]["auc"].get(name)
            sweep[name][sz] = dict(
                diluted_roc_auc=(A["diluted_positives"]["roc_auc"] if A else None),
                diluted_average_precision=(A["diluted_positives"]["average_precision"]
                                           if A else None),
                has_single_score=bool(A),
                recall_diluted={bk: summary[sz]["budgets"][bk][name]["recall_diluted"]
                                for bk in budget_keys},
                fpr={bk: summary[sz]["budgets"][bk][name]["fpr"]
                     for bk in budget_keys})

    # ------------------------------------------------------------ the verdict
    def m_of(g):
        return None if g is None else g["mean"]

    winners_same_size, winners_any = [], []
    for sz in sizes:
        for bk in budget_keys:
            base = m_of(summary[sz]["budgets"][bk]["max"]["recall_diluted"])
            if base is None:
                continue
            for name in rule_order:
                if name == "max":
                    continue
                v = m_of(summary[sz]["budgets"][bk][name]["recall_diluted"])
                if v is None or v <= base + 1e-12:
                    continue
                g = summary[sz]["budgets"][bk][name]["diluted_recall_gain_over_max"]
                item = dict(chunk_bytes=int(sz), budget=bk, rule=name,
                            recall_diluted=v, max_recall_diluted=base, gain=v - base,
                            free_parameters=summary[sz]["budgets"][bk][name]
                            ["free_parameters"],
                            gain_sd=(None if g is None else g["sd"]),
                            paired_t=summary[sz]["budgets"][bk][name]["paired_t_vs_max"],
                            fpr=m_of(summary[sz]["budgets"][bk][name]["fpr"]),
                            max_fpr=m_of(summary[sz]["budgets"][bk]["max"]["fpr"]))
                # Thresholds are picked to meet the budget on CALIBRATION, so two
                # rules can land at different FPRs on TEST. A rule that "wins" on
                # diluted recall while spending more test false positives than max has
                # not beaten max; it has moved the operating point. Recorded per
                # winner rather than asserted.
                item["test_fpr_ratio_vs_max"] = (
                    None if not item["max_fpr"] else item["fpr"] / item["max_fpr"])
                item["test_fpr_over_budget"] = bool(
                    item["fpr"] is not None and item["fpr"] > float(bk) + 1e-12)
                winners_any.append(item)
                if item["free_parameters"] == 1:
                    winners_same_size.append(item)
    # A winner is only interesting if the gain survives the seed-to-seed spread AND it
    # did not simply buy the recall with extra false positives.
    robust = [w for w in winners_same_size
              if w["gain_sd"] is not None and w["gain"] > w["gain_sd"]]
    robust_same_cost = [w for w in robust
                        if w["test_fpr_ratio_vs_max"] is not None
                        and w["test_fpr_ratio_vs_max"] <= 1.10]

    ref = "512" if "512" in sizes else sizes[0]
    small_vs_ref = {}
    for sz in sizes:
        small_vs_ref[sz] = dict(
            max_diluted_roc_auc=m_of(summary[sz]["auc"].get("max", {})
                                     .get("diluted_positives", {}).get("roc_auc")),
            max_recall_diluted={bk: m_of(summary[sz]["budgets"][bk]["max"]
                                         ["recall_diluted"]) for bk in budget_keys},
            best_single_param_rule={
                bk: max(((n, m_of(summary[sz]["budgets"][bk][n]["recall_diluted"]))
                         for n in one_param
                         if m_of(summary[sz]["budgets"][bk][n]["recall_diluted"])
                         is not None), key=lambda p: p[1], default=(None, None))
                for bk in budget_keys})
    smaller_helped = {}
    for bk in budget_keys:
        series = [(int(sz), small_vs_ref[sz]["max_recall_diluted"][bk]) for sz in sizes
                  if small_vs_ref[sz]["max_recall_diluted"][bk] is not None]
        series.sort()
        smaller_helped[bk] = dict(
            series=series,
            best_size=(min(series, key=lambda p: -p[1])[0] if series else None),
            monotone_improving_as_size_shrinks=bool(
                len(series) > 1 and all(series[i][1] >= series[i + 1][1] - 1e-12
                                        for i in range(len(series) - 1))))
    auc_series = sorted([(int(sz), small_vs_ref[sz]["max_diluted_roc_auc"])
                         for sz in sizes
                         if small_vs_ref[sz]["max_diluted_roc_auc"] is not None])

    verdict = dict(
        question="is there an aggregation rule, or a chunk size, that beats Kwon "
                 "2026's max on diluted payloads?",
        anything_beat_max_at_equal_parameter_cost=bool(winners_same_size),
        robust_winners_gain_exceeds_seed_sd=robust,
        robust_winners_that_also_spend_no_more_test_fpr=robust_same_cost,
        fpr_note="a threshold meets the budget on CALIBRATION; the test FPR each rule "
                 "lands at is reported per rule, and a 'winner' whose test FPR is "
                 "higher than max's has moved the operating point rather than beaten "
                 "the rule.",
        all_single_parameter_winners=winners_same_size,
        winners_including_multi_parameter_rules=winners_any,
        smaller_chunks_helped=smaller_helped,
        max_diluted_roc_auc_by_chunk_size=auc_series,
        per_chunk_size=small_vs_ref,
        reference_chunk_size=ref,
        caveat=("max is Kwon 2026's published rule. A negative result here is a "
                "characterisation of the aggregation space, not a defeat of this "
                "project's design; what it rules out is the paper claiming any "
                "file-level aggregation of its own as the fix for dilution."))

    out = dict(config=config,
               splits={k: dict(n=len(v), sources=sorted({r["source_id"] for r in v}))
                       for k, v in (("train", train_rows), ("calib", calib_rows),
                                    ("test", test_rows))},
               n_diluted=dict(calib=int(c_dil.sum()), test=int(t_dil.sum())),
               rules=dict(single_parameter=one_param, multi_parameter=multi_param,
                          notes={n: per_size[sizes[0]]["seeds"][0]["budgets"]
                                 [budget_keys[0]][n].get("note", "") for n in rule_order}),
               per_chunk_size=per_size, summary=summary, chunk_size_sweep=sweep,
               verdict=verdict, total_seconds=round(time.time() - t_start, 1),
               smoke_run=smoke,
               smoke_run_note=(f"--limit {a.limit} was used: not a result." if smoke
                               else None),
               scorer_is_a_stand_in=stand_in,
               scorer_note=(numpy_scorer.DESCRIPTION if stand_in else None))
    save_json(RESULTS / f"aggregation_family_{a.tag}.json", out)
    if jsonl_path is not None:
        n = 0
        with open(jsonl_path, "w") as fh:
            for size in a.chunk_sizes:
                for k in range(a.seeds):
                    shard = CK / f"s{size}_seed{a.seed_base + k}.jsonl"
                    if shard.exists():
                        for line in shard.open():
                            if line.strip():
                                fh.write(line); n += 1
        log(f"wrote {jsonl_path}  ({n} rows, stitched from the per-seed checkpoints)")

    # ------------------------------------------------------------ console
    log("\n" + "=" * 100)
    log("PART 1 - THE AGGREGATION FAMILY, test split, mean over %d seed(s)" % a.seeds)
    log("=" * 100)
    if stand_in:
        log("*** NUMPY STAND-IN SCORER: these rates are NOT a measurement of the "
            "design and must not be compared with Kwon 2026 or with "
            "experiment_linear.json. The COMPARISON BETWEEN RULES is still internally "
            "valid - every rule here reads the same chunk scores.")
    for sz in sizes:
        log(f"\n---- chunk size {sz} bytes (stride {per_size[sz]['stride']}, "
            f"{per_size[sz]['seeds'][0]['mean_chunks_per_file']['test']:.0f} "
            f"chunks/test file) ----")
        for bk in budget_keys:
            base = m_of(summary[sz]["budgets"][bk]["max"]["recall_diluted"])
            log(f"  calibration FPR budget {bk}   (dil@maxFPR is the ORACLE "
                f"diagnostic: diluted recall at max's own test FPR)")
            log(f"    {'rule':22s} {'par':>3s} {'fpr':>8s} {'dil.rec':>8s} "
                f"{'sd':>7s} {'vs max':>8s} {'plain':>7s} {'all':>7s} "
                f"{'dil@maxFPR':>10s}")
            ordered = sorted(one_param,
                             key=lambda n: -(m_of(summary[sz]["budgets"][bk][n]
                                                  ["recall_diluted"]) or -1)) + multi_param
            best = max((m_of(summary[sz]["budgets"][bk][n]["recall_diluted"]) or -1)
                       for n in one_param)
            for n in ordered:
                r = summary[sz]["budgets"][bk][n]
                dv = m_of(r["recall_diluted"])
                gap = None if (dv is None or base is None) else dv - base
                mark = ""
                if n in one_param and dv is not None and abs(dv - best) < 1e-12:
                    mark = "  <= best single-parameter rule"
                if n == "max":
                    mark += "  [Kwon 2026, the incumbent]"
                log(f"    {n:22s} {r['free_parameters']:>3d} "
                    f"{fmt(m_of(r['fpr']), 5):>8s} {fmt(dv):>8s} "
                    f"{fmt(None if r['recall_diluted'] is None else r['recall_diluted']['sd']):>7s} "
                    f"{('    n/a' if gap is None else f'{gap:+.4f}'):>8s} "
                    f"{fmt(m_of(r['recall_plain'])):>7s} "
                    f"{fmt(m_of(r['recall_all'])):>7s} "
                    f"{fmt(m_of(r.get('recall_diluted_at_max_test_fpr'))):>10s}{mark}")
        log(f"  threshold-free on the test split (every benign file is a negative in "
            f"all three columns):")
        log(f"    {'rule':22s} {'AUC all':>9s} {'AUC plain':>9s} {'AUC dil':>9s} "
            f"{'AP dil':>9s}")
        for n in sorted(auc_rules,
                        key=lambda n: -(m_of(summary[sz]["auc"][n]["diluted_positives"]
                                             ["roc_auc"]) or -1)):
            A = summary[sz]["auc"][n]
            log(f"    {n:22s} "
                f"{fmt(m_of(A['all_positives']['roc_auc'])):>9s} "
                f"{fmt(m_of(A['plain_positives']['roc_auc'])):>9s} "
                f"{fmt(m_of(A['diluted_positives']['roc_auc'])):>9s} "
                f"{fmt(m_of(A['diluted_positives']['average_precision'])):>9s}")

    log("\n" + "=" * 100)
    log("PART 2 - THE CHUNK-SIZE SWEEP: does a smaller chunk recover a diluted payload?")
    log("=" * 100)
    log("  a smaller chunk means the injected payload is a LARGER FRACTION of the chunk "
        "holding it")
    for bk in budget_keys:
        log(f"\n  diluted recall at budget {bk}, by chunk size")
        header = f"    {'rule':22s}" + "".join(f"{s + 'B':>10s}" for s in sizes)
        log(header)
        for n in ["max", "top2_mean", "top3_mean", "top5_mean", "top10_mean",
                  "quantile_0.99", "count_above_bq99", "max_minus_mean", "mean",
                  "mean_or_max", "count_above_t_tuned"]:
            if n not in rule_order:
                continue
            log(f"    {n:22s}" + "".join(
                f"{fmt(m_of(sweep[n][s]['recall_diluted'][bk])):>10s}" for s in sizes))
    log(f"\n  threshold-free ROC AUC on DILUTED positives, by chunk size")
    log(f"    {'rule':22s}" + "".join(f"{s + 'B':>10s}" for s in sizes))
    for n in ["max", "top2_mean", "top3_mean", "top5_mean", "top10_mean",
              "quantile_0.99", "count_above_bq99", "max_minus_mean", "mean"]:
        if n not in rule_order:
            continue
        log(f"    {n:22s}" + "".join(
            f"{fmt(m_of(sweep[n][s]['diluted_roc_auc'])):>10s}" for s in sizes))
    log(f"\n  cost: chunks per test file, and seconds per seed")
    for sz in sizes:
        s0 = per_size[sz]["seeds"]
        log(f"    {sz + 'B':>7s}  {s0[0]['mean_chunks_per_file']['test']:9.1f} "
            f"chunks/file   truncated test files: {s0[0]['truncated_files']['test']}")

    log("\n" + "=" * 100)
    log("PART 3 - THE LEARNED COMBINER (calibration-fitted, test-evaluated)")
    log("=" * 100)
    log("  it has more free parameters than any threshold rule, so the PERMUTED-LABEL "
        "control beside each number is what that parameter count buys by chance")
    for sz in sizes:
        for ckey in ("learned_combiner", "learned_combiner_no_size_features"):
            C = summary[sz].get(ckey)
            if C is None:
                continue
            log(f"\n  chunk size {sz}B  [{C['variant']}]  solver {C['solver']}, "
                f"{C['free_parameters']} free parameters vs ONE for a threshold rule")
            log("    standardised weight per feature (mean over seeds):")
            for f in C["features"]:
                g = C["standardised_weights"][f]
                flag = ("   <- driven to ~zero" if g is not None
                        and abs(g["mean"]) < 0.05 else "")
                log(f"      {f:16s} {fmt(m_of(g)):>9s}  "
                    f"(sd {fmt(None if g is None else g['sd'])}){flag}")
            for bk in budget_keys:
                b = C["budgets"][bk]
                gain = C["diluted_recall_gain_over_max"][bk]
                pc = C["permutation_control"]["diluted_recall"][bk]
                gain_txt = "n/a" if gain is None else "%+.4f" % gain["mean"]
                log(f"    budget {bk}: fpr {fmt(m_of(b['fpr']), 5)}  diluted recall "
                    f"{fmt(m_of(b['recall_diluted']))}  vs max {gain_txt}"
                    f"   | PERMUTED-LABEL control: {fmt(m_of(pc))}")
            log(f"    diluted ROC AUC "
                f"{fmt(m_of(C['auc']['diluted_positives']['roc_auc']))}"
                f"   | PERMUTED-LABEL control "
                f"{fmt(m_of(C['permutation_control']['diluted_roc_auc']))}")

    log("\n" + "=" * 100)
    log("THE ANSWER")
    log("=" * 100)
    if winners_same_size:
        log(f"  SOMETHING BEAT MAX on diluted recall at equal parameter cost: "
            f"{len(winners_same_size)} (rule, chunk size, budget) combination(s).")
        for w in sorted(winners_same_size, key=lambda w: -w["gain"])[:12]:
            sd_txt = "n/a" if w["gain_sd"] is None else "%.4f" % w["gain_sd"]
            log(f"    {w['rule']:22s} {w['chunk_bytes']:>5d}B budget {w['budget']:>6s}  "
                f"{w['recall_diluted']:.4f} vs max {w['max_recall_diluted']:.4f}  "
                f"gain {w['gain']:+.4f} (seed sd {sd_txt})")
        log(f"  of those, {len(robust)} have a gain larger than its own seed-to-seed "
            f"sd, and {len(robust_same_cost)} of those also spend no more test FPR "
            f"than max (within 10%) - i.e. they are not just a looser operating point.")
        for w in sorted(robust_same_cost, key=lambda w: -w["gain"])[:12]:
            log(f"    SURVIVES BOTH: {w['rule']:20s} {w['chunk_bytes']:>5d}B budget "
                f"{w['budget']:>6s}  gain {w['gain']:+.4f}  test fpr {w['fpr']:.5f} "
                f"vs max {w['max_fpr']:.5f}")
    else:
        log("  NOTHING BEAT MAX on diluted recall at equal parameter cost, at any "
            "chunk size, at any budget. Max-only remains the best single-threshold "
            "aggregation of a file's chunk scores in this corpus.")
    if winners_any and not winners_same_size:
        log(f"  ({len(winners_any)} multi-parameter rule(s) beat it, which is not an "
            f"equal-cost comparison; see verdict.winners_including_multi_parameter_rules)")
    log("  smaller chunks:")
    for bk in budget_keys:
        s = smaller_helped[bk]
        log(f"    budget {bk}: diluted recall by size "
            + "  ".join(f"{sz}B={v:.4f}" for sz, v in s["series"])
            + f"   best size {s['best_size']}B"
            + ("   MONOTONE: smaller is better" if
               s["monotone_improving_as_size_shrinks"] else ""))
    log("    max's diluted ROC AUC by size: "
        + "  ".join(f"{sz}B={v:.4f}" for sz, v in auc_series))
    log(f"\n  wrote results/aggregation_family_{a.tag}.json"
        + (f" and results/aggregation_scores_{a.tag}.jsonl" if jsonl_path else ""))
    log(f"  total {out['total_seconds']}s")


if __name__ == "__main__":
    main()
