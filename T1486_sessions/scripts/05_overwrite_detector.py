#!/usr/bin/env python3
"""T1486 stage-4 detector on REAL RanSMAP traces: high-entropy in-place overwrite.

This is SSD-Insider's signal (Baek et al., ICDCS 2018: "an overwrite is a write to
an LBA that was read within the time window"; FlashGuard, Huang et al., CCS 2017,
relies on the same read-then-overwrite behaviour), restricted here to writes whose
RanSMAP entropy is >= 0.9375 (7.5 bits). It is NOT a new detector; what this kit
adds is the measurement on RanSMAP against encryption, compression and wiping
negatives, and the hand-off of a measured operating point to link A4.

Subcommands (run from T1486_sessions/):

  extract  --split original|mix   per-run features + per-second series, resumable
  evaluate                        LOAO / LOCO / STRICT on original, both rules
  freeze                          fit both thresholds on ALL of original and write
                                  results/overwrite_frozen.json BEFORE mix is seen
  mix                             apply the frozen thresholds to dataset/mix
  emit                            write the measured operating point and A4 anchors

Rules (fixed 28 Oct, before any windowed or mix result was seen)

  R_share   run level. share = high-entropy writes landing on an LBA read EARLIER in
            the run / all high-entropy writes. Flag if share > t (midpoint of class
            medians on the training fold). Needs the whole run; reported for
            continuity with RESULTS_ORIGINAL_1440.md.
  R_win     online. Every second s, OW10(s) = number of high-entropy writes in
            (s-10, s] whose LBA was read at most READ_WINDOW_S = 10 s before the
            write. Alarm at the first s with OW10(s) >= t. t is the GEOMETRIC
            midpoint of the class medians of each run's max OW10 on the training
            fold (counts span orders of magnitude, so the arithmetic midpoint would
            be set by the largest runs).
            RESULT 28 Oct: FAILED on original (LOAO LR+ 1.7). AESCrypt, Zip and
            SDelete write so much that they produce thousands of incidental
            overwrites per window; a COUNT does not normalise for volume.

  R_winshare  POST HOC, added after R_win failed and before any mix data existed.
            Every second s, over (s-10, s]: high = high-entropy writes, ow = those
            whose LBA was read at most 10 s earlier. Alarm at the first s with
            high >= MIN_HIGH_PER_WINDOW (256 blocks = 1 MiB, the constant
            extract_features.py already used for `sustain`) and ow/high >= t. t is the
            midpoint of class medians of each run's maximum windowed share on the
            training fold. Because it was designed after seeing original, its only
            untainted test is `mix`, with t frozen first. It is the online form of
            R_share and, if it holds on mix, the one whose operating point feeds A4.

Labels: ransomware = Conti, Darkside, LockBit, REvil, Ryuk, WannaCry; every other
application is negative, AESCrypt, Zip and SDelete included. Unit of
generalisation: the application. In `mix` every run contains ransomware, so mix
measures recall under concurrent benign load only; FPR comes from `original`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics as st
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

# numpy/pandas are needed only by `extract`; evaluate/freeze/mix/emit (and the chain
# wrapper) are pure standard library, so they run on a Python without them.
np = pd = None


def _need_numpy():
    global np, pd
    if pd is None:
        import numpy as _np
        import pandas as _pd
        np, pd = _np, _pd

KIT = Path(__file__).resolve().parent.parent
IMPL = KIT.parent
sys.path.insert(0, str(IMPL))
from common import anchors  # noqa: E402

RESULTS = KIT / "results"
DATA = IMPL / "_shared_data" / "RanSMAP" / "dataset"
RANSOM = ("Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry")
H_ENTROPY = 0.9375
READ_WINDOW_S = 10.0
ALARM_WINDOW_S = 10
MIN_HIGH_PER_WINDOW = 256
COLS = ["sec", "ns", "addr", "size", "ent", "flag"]
CONST = {"H_ENTROPY": H_ENTROPY, "READ_WINDOW_S": READ_WINDOW_S,
         "ALARM_WINDOW_S": ALARM_WINDOW_S, "MIN_HIGH_PER_WINDOW": MIN_HIGH_PER_WINDOW,
         "fixed": "2026-10-28"}


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ extraction
def _load(run_dir: Path, sig: str, usecols):
    _need_numpy()
    kw = dict(header=None, names=COLS, usecols=usecols, skipinitialspace=True)
    p = run_dir / f"{sig}.csv"
    if p.exists():
        return pd.read_csv(p, **kw)
    z = run_dir / f"{sig}.zip"
    if z.exists():
        return pd.read_csv(z, compression="zip", **kw)
    raise FileNotFoundError(f"{sig}.csv/.zip missing")


def run_features(run_dir: Path) -> dict:
    w = _load(run_dir, "ata_write", ["sec", "ns", "addr", "ent"])
    r = _load(run_dir, "ata_read", ["sec", "ns", "addr"])
    wt = w["sec"].to_numpy(np.float64) + w["ns"].to_numpy(np.float64) * 1e-9
    rt = r["sec"].to_numpy(np.float64) + r["ns"].to_numpy(np.float64) * 1e-9
    t0 = float(min(wt.min(), rt.min() if rt.size else wt.min()))
    H = w["ent"].to_numpy(np.float64) >= H_ENTROPY
    hl, ht = w["addr"].to_numpy(np.int64)[H], wt[H]

    # most recent read of the same LBA strictly before each high-entropy write
    ev = pd.DataFrame({
        "lba": np.concatenate([r["addr"].to_numpy(np.int64), hl]),
        "t": np.concatenate([rt, ht]),
        "is_w": np.concatenate([np.zeros(len(rt), np.int8), np.ones(len(ht), np.int8)]),
    }).sort_values(["lba", "t", "is_w"], kind="mergesort")
    ev["last_read"] = ev["t"].where(ev["is_w"] == 0)
    ev["last_read"] = ev.groupby("lba")["last_read"].ffill()
    wv = ev[ev["is_w"] == 1]
    gap = (wv["t"] - wv["last_read"]).to_numpy()
    ever = ~np.isnan(gap)
    recent = ever & (gap <= READ_WINDOW_S)

    n_h = int(H.sum())
    f = {"n_writes": int(len(w)), "n_reads": int(len(r)), "n_high": n_h,
         "share": float(ever.mean()) if n_h else 0.0,
         "share_recent": float(recent.mean()) if n_h else 0.0,
         "duration_s": float(max(wt.max(), rt.max() if rt.size else 0) - t0)}
    # per-second series, sparse: seconds since t0
    sec_h = np.floor(ht - t0).astype(np.int64)
    sec_ow = np.floor(wv["t"].to_numpy()[recent] - t0).astype(np.int64)
    n_sec = int(math.floor(f["duration_s"])) + 1
    ow = np.bincount(sec_ow, minlength=n_sec) if sec_ow.size else np.zeros(n_sec, np.int64)
    ow10 = np.convolve(ow, np.ones(ALARM_WINDOW_S, np.int64))[:n_sec]
    f["max_ow10"] = int(ow10.max()) if ow10.size else 0
    f["first_high_s"] = int(sec_h.min()) if sec_h.size else None
    f["first_ow_s"] = int(sec_ow.min()) if sec_ow.size else None
    nz = np.nonzero(ow)[0]
    f["ow_series"] = [[int(s), int(ow[s])] for s in nz]            # sparse
    hs = np.bincount(sec_h, minlength=n_sec) if sec_h.size else np.zeros(n_sec, np.int64)
    nzh = np.nonzero(hs)[0]
    f["high_series"] = [[int(s), int(hs[s])] for s in nzh]
    f["t0_unix"] = t0
    return f


def _work(item):
    split, rel, path = item
    t = time.time()
    try:
        rec = run_features(Path(path))
        rec["error"] = None
    except Exception as e:
        rec = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=2)}
    p = rel.split("/")
    rec.update(split=split, rel_path=rel, cpu=p[0], ram=p[1], app=p[2], run=p[3],
               config=f"{p[0]}/{p[1]}", seconds=round(time.time() - t, 2), constants=CONST)
    if split == "mix":
        benign, _, ransom = p[2].rpartition("_")   # Firefox_Office_REvil -> 2 benign
        rec.update(benign_app=benign, ransom_app=ransom)
    return rec


def features_path(split):
    return RESULTS / f"overwrite_features_{split}.jsonl"


def cmd_extract(a):
    root = DATA / a.split
    out = features_path(a.split)
    if not root.is_dir():
        sys.exit(f"{root} not found. For mix, fetch it first:\n"
                 f"  cd {IMPL/'_shared_data'/'RanSMAP'}\n"
                 f"  git sparse-checkout add 'dataset/mix/**'\n"
                 f"  git checkout HEAD -- dataset/mix")
    runs = sorted(str(p.relative_to(root)) for p in root.glob("*/*/*/*") if p.is_dir())
    good = []
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if d.get("error") is None:
                good.append(l)
        out.write_text("".join(l + "\n" for l in good))
    done = {json.loads(l)["rel_path"] for l in good}
    todo = [r for r in runs if r not in done][: a.limit or None]
    log(f"{a.split}: {len(runs)} runs, {len(done)} done, {len(todo)} to do -> {out}")
    t0, nerr = time.time(), 0
    with open(out, "a") as fh, Pool(a.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(
                _work, [(a.split, r, str(root / r)) for r in todo]), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            nerr += rec["error"] is not None
            if rec["error"] or i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                log(f"[{i}/{len(todo)}] {rec['rel_path']} "
                    f"{'ERROR ' + rec['error'] if rec['error'] else ''} "
                    f"{el/60:.1f} min, eta {el/i*(len(todo)-i)/60:.1f} min")
    log(f"done: {len(todo)-nerr} ok, {nerr} failed (re-run to retry failures)")


# ------------------------------------------------------------------ evaluation
def load_rows(split):
    rows, skipped = [], []
    for l in features_path(split).read_text().splitlines():
        d = json.loads(l)
        if d.get("error"):
            skipped.append({"run": d["rel_path"], "why": d["error"]})
            continue
        d["y"] = (d["app"] in RANSOM) if split == "original" else True
        rows.append(d)
    return rows, skipped


def fit_share(tr):
    mp = st.median(r["share"] for r in tr if r["y"])
    mn = st.median(r["share"] for r in tr if not r["y"])
    return {"t": (mp + mn) / 2}


def fit_win(tr):
    mp = st.median(r["max_ow10"] for r in tr if r["y"])
    mn = st.median(r["max_ow10"] for r in tr if not r["y"])
    return {"t": math.sqrt(max(mp, 1) * max(mn, 1))}


def alarm_second(r, t):
    """First second at which OW10 >= t, or None."""
    win, acc = [], 0
    q = []                               # (sec, count) inside the window
    for s, c in r["ow_series"]:
        q.append((s, c))
        acc += c
        while q and q[0][0] <= s - ALARM_WINDOW_S:
            acc -= q.pop(0)[1]
        if acc >= t:
            return s
    return None


def windows(r):
    """Yield (s, high10, ow10) for every second with a high-entropy write."""
    ow = dict(map(tuple, r["ow_series"]))
    hi = dict(map(tuple, r["high_series"]))
    secs = sorted(hi)
    j = 0
    acc_h = acc_o = 0
    for i, s in enumerate(secs):
        acc_h += hi[s]
        acc_o += ow.get(s, 0)
        while secs[j] <= s - ALARM_WINDOW_S:
            acc_h -= hi[secs[j]]
            acc_o -= ow.get(secs[j], 0)
            j += 1
        yield s, acc_h, acc_o


def max_winshare(r):
    m = 0.0
    for _, h, o in windows(r):
        if h >= MIN_HIGH_PER_WINDOW:
            m = max(m, o / h)
    return m


def alarm_second_share(r, t):
    for s, h, o in windows(r):
        if h >= MIN_HIGH_PER_WINDOW and o / h >= t:
            return s
    return None


def fit_winshare(tr):
    mp = st.median(max_winshare(r) for r in tr if r["y"])
    mn = st.median(max_winshare(r) for r in tr if not r["y"])
    return {"t": (mp + mn) / 2}


FIT = {"R_share": None, "R_win": None, "R_winshare": None}   # filled below
ALARM = {"R_win": lambda r, t: alarm_second(r, t),
         "R_winshare": lambda r, t: alarm_second_share(r, t)}


def fires(rule, r, th):
    if rule == "R_share":
        return r["share"] > th["t"]
    return ALARM[rule](r, th["t"]) is not None


def high_blocks_before(r, s):
    return sum(c for x, c in r["high_series"] if x <= s)


def op_dict(tp, fn, fp, tn, threshold, detector):
    op = anchors.OperatingPoint(tpr=tp / (tp + fn), fpr=fp / (fp + tn),
                                n_pos=tp + fn, n_neg=fp + tn,
                                threshold=threshold, detector=detector)
    d = op.as_dict()
    d.update(tp=tp, fn=fn, fp=fp, tn=tn)
    try:
        d["confidence_at_prior_0.01"] = anchors.confidence_from_operating_point(op, 0.01)
    except anchors.AnchorError as e:
        d["confidence_at_prior_0.01"] = None
        d["refused"] = str(e)
    return d


def folds(rows, scheme):
    apps = sorted({r["app"] for r in rows})
    if scheme == "LOAO":
        for x in apps:
            yield x, [r for r in rows if r["app"] != x], [r for r in rows if r["app"] == x]
    elif scheme == "LOCO":
        for c in sorted({r["config"] for r in rows}):
            yield c, [r for r in rows if r["config"] != c], [r for r in rows if r["config"] == c]
    else:
        for x in apps:
            for c in sorted({r["cpu"] for r in rows}):
                yield (f"{x}|train {c}",
                       [r for r in rows if r["app"] != x and r["cpu"] == c],
                       [r for r in rows if r["app"] == x and r["cpu"] != c])


def evaluate_rule(rows, rule, scheme):
    fit = FIT[rule]
    tp = fn = fp = tn = 0
    per_app, lat, lat_onset, blocks, thr = {}, [], [], [], []
    for _, tr, te in folds(rows, scheme):
        th = fit(tr)
        thr.append(th["t"])
        for r in te:
            hit = fires(rule, r, th)
            k = per_app.setdefault(r["app"], [0, 0])
            k[0] += hit
            k[1] += 1
            if r["y"]:
                tp += hit; fn += not hit
                if hit and rule != "R_share":
                    s = ALARM[rule](r, th["t"])
                    lat.append(s)
                    lat_onset.append(s - r["first_ow_s"])
                    blocks.append(high_blocks_before(r, s))
            else:
                fp += hit; tn += not hit
    d = op_dict(tp, fn, fp, tn, threshold={"min": min(thr), "max": max(thr)},
                detector=f"T1486 {rule} ({scheme}, held-out)")
    d["per_app_flagged"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_app.items())}
    if lat:
        d["latency_s_from_run_start"] = {"median": st.median(lat), "max": max(lat)}
        d["latency_s_from_first_overwrite"] = {"median": st.median(lat_onset),
                                               "p90": sorted(lat_onset)[int(0.9 * len(lat_onset))],
                                               "max": max(lat_onset)}
        mb = [b * 4096 / 2 ** 20 for b in blocks]
        d["high_entropy_MiB_written_before_alarm"] = {
            "median": st.median(mb), "p90": sorted(mb)[int(0.9 * len(mb))], "max": max(mb),
            "note": "all high-entropy writes in the run up to the alarm second, so an "
                    "upper bound on what was encrypted before the alarm"}
    return d


FIT.update(R_share=fit_share, R_win=fit_win, R_winshare=fit_winshare)


def cmd_evaluate(a):
    rows, skipped = load_rows("original")
    out = {"n_runs": len(rows), "n_pos": sum(r["y"] for r in rows),
           "n_neg": sum(not r["y"] for r in rows), "skipped": skipped,
           "constants": CONST, "protocol": __doc__, "rules": {}}
    log(f"original: {out['n_runs']} runs ({out['n_pos']} ransomware, {out['n_neg']} other)")
    for rule in ("R_share", "R_win", "R_winshare"):
        out["rules"][rule] = {}
        for scheme in ("LOAO", "LOCO", "STRICT"):
            d = evaluate_rule(rows, rule, scheme)
            out["rules"][rule][scheme] = d
            extra = ""
            if "latency_s_from_first_overwrite" in d:
                extra = (f"  latency from first overwrite median "
                         f"{d['latency_s_from_first_overwrite']['median']} s, "
                         f"MiB before alarm median "
                         f"{d['high_entropy_MiB_written_before_alarm']['median']:.1f}")
            log(f"{rule:8s} {scheme:6s} TP {d['tp']}/{d['tp']+d['fn']}  FP {d['fp']}/"
                f"{d['fp']+d['tn']}  LR+ {'>=' if d['fpr_is_floored'] else ''}"
                f"{d['lr_plus'] or 0:.1f}  conf {'>=' if d['fpr_is_floored'] else ''}"
                f"{d['confidence_at_prior_0.01'] or 0:.3f}{extra}")
            log(f"         per app {d['per_app_flagged']}")
    (RESULTS / "overwrite_eval_original.json").write_text(json.dumps(out, indent=1))
    log("wrote results/overwrite_eval_original.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_freeze(a):
    dst = RESULTS / "overwrite_frozen.json"
    if dst.exists() and not a.force:
        sys.exit(f"{dst} exists; thresholds are frozen. --force only if you mean to "
                 f"re-freeze, and say why in the paper.")
    if features_path("mix").exists():
        sys.exit("mix features already exist: freezing now would not be a pre-registration")
    rows, _ = load_rows("original")
    fr = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
          "fitted_on": "all of dataset/original (no mix data present at freeze time)",
          "features_sha256": sha(features_path("original")),
          "R_share": fit_share(rows), "R_win": fit_win(rows),
          "R_winshare": fit_winshare(rows), "constants": CONST,
          "note": "R_win failed on original before freezing; R_winshare is post hoc, "
                  "so mix is its only untainted test"}
    dst.write_text(json.dumps(fr, indent=1))
    log(json.dumps(fr, indent=1))


def cmd_mix(a):
    fr = json.loads((RESULTS / "overwrite_frozen.json").read_text())
    rows, skipped = load_rows("mix")
    out = {"frozen": fr, "n_runs": len(rows), "skipped": skipped,
           "what_this_measures": "recall under concurrent benign load; every mix run "
                                 "contains ransomware, so no FPR can be measured here",
           "rules": {}}
    for rule in ("R_share", "R_win", "R_winshare"):
        by, lat = {}, []
        for r in rows:
            hit = fires(rule, r, fr[rule])
            k = by.setdefault(r["app"], [0, 0])
            k[0] += hit; k[1] += 1
            if hit and rule != "R_share":
                s = ALARM[rule](r, fr[rule]["t"])
                lat.append(s - (r["first_ow_s"] or 0))
        hit = sum(v[0] for v in by.values())
        n = sum(v[1] for v in by.values())
        by_benign, by_ransom = {}, {}
        for r in rows:
            h = fires(rule, r, fr[rule])
            for key, dd in ((r["benign_app"], by_benign), (r["ransom_app"], by_ransom)):
                v = dd.setdefault(key, [0, 0]); v[0] += h; v[1] += 1
        out["rules"][rule] = {
            "recall": hit / n if n else None, "hits": hit, "n": n,
            "recall_ci95": wilson(hit, n),
            "by_combo": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())},
            "by_benign_app": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_benign.items())},
            "by_ransomware": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_ransom.items())},
            "latency_s_from_first_overwrite": ({"median": st.median(lat), "max": max(lat)}
                                               if lat else None)}
        log(f"{rule}: recall {hit}/{n}  by benign {out['rules'][rule]['by_benign_app']}  "
            f"by ransomware {out['rules'][rule]['by_ransomware']}")
    (RESULTS / "overwrite_eval_mix.json").write_text(json.dumps(out, indent=1))
    log("wrote results/overwrite_eval_mix.json")


def wilson(k, n, z=1.96):
    if not n:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


# ------------------------------------------------------------------ hand-off
def cmd_emit(a):
    """Operating point for link A4 = R_win, LOAO, held out. Anchors = one per run
    the frozen R_win alarms on, subject = that real RanSMAP run (host kind: each run
    is one guest machine), emitted at the alarm time on the trace's own clock."""
    ev = json.loads((RESULTS / "overwrite_eval_original.json").read_text())
    if a.rule not in ("R_winshare", "R_share"):
        sys.exit("only R_winshare (online) or R_share (whole run) may feed A4; R_win failed")
    d = ev["rules"][a.rule][a.scheme]
    op = anchors.OperatingPoint(tpr=d["tpr"], fpr=d["fpr"], n_pos=d["n_pos"],
                                n_neg=d["n_neg"], threshold=None,
                                detector=f"T1486 high-entropy in-place overwrite {a.rule} "
                                         f"(SSD-Insider signal), {a.scheme} held-out, "
                                         f"real RanSMAP dataset/original")
    op_path = RESULTS / "operating_point_overwrite.json"
    op.save(op_path)
    extra = json.loads(op_path.read_text())
    extra.update(source="measured on real RanSMAP traces, held-out by application",
                 eval_file="results/overwrite_eval_original.json",
                 features_sha256=sha(features_path("original")),
                 caveats=CAVEATS + ([
                     "FPR 0 is floored by the rule of three (its 95% upper limit), so LR+ "
                     "and confidence are conservative lower bounds"] if op.is_floored() else [
                     f"FPR measured ({d['fp']}/{d['n_neg']} false alarms), not floored: "
                     f"LR+ and confidence are point estimates, not bounds"]))
    op_path.write_text(json.dumps(extra, indent=2, sort_keys=True))
    conf = anchors.confidence_from_operating_point(op, a.prior)
    log(f"operating point: tpr {op.tpr:.4f} fpr {op.fpr} (effective {op.effective_fpr():.5f})"
        f" LR+ {op.lr_plus():.2f}{' (lower bound)' if op.is_floored() else ''} "
        f"-> confidence {conf:.6f} at prior {a.prior}")

    fr = json.loads((RESULTS / "overwrite_frozen.json").read_text())
    rows, _ = load_rows("original")
    # each run is judged by the threshold of the LOAO fold that EXCLUDED its own
    # application, so every anchor's alarm decision is held out, like its confidence
    fold_t = {}
    for app, tr, _ in folds(rows, "LOAO"):
        fold_t[app] = (fit_winshare(tr) if a.rule == "R_winshare" else fit_share(tr))["t"]
    out = RESULTS / "anchors_a4_overwrite.jsonl"
    out.write_text("")
    store = anchors.AnchorStore(out)
    n = nfp = 0
    for r in rows:
        t_run = fold_t[r["app"]]
        if a.rule == "R_share":
            s = r["duration_s"] if r["share"] > t_run else None
        else:
            s = alarm_second_share(r, t_run)
        if s is None:
            continue
        t_alarm = r["t0_unix"] + s
        anc = anchors.Anchor.create(
            "A4", "host", f"ransmap:{r['rel_path']}", emitted_at=t_alarm,
            ttl_seconds=a.ttl, operating_point=op, prior=a.prior,
            evidence={"detection": f"{a.rule} alarm on the real trace", "rule": a.rule,
                      "group": f"ransmap:{r['rel_path']}",
                      "alarm_second_from_trace_start": s,
                      "first_overwrite_second": r["first_ow_s"],
                      "threshold_used": t_run,
                      "threshold_from": f"LOAO fold excluding {r['app']}",
                      "frozen_threshold_for_mix": fr[a.rule]["t"],
                      "as_true_positive": int(r["y"]), "as_false_positive": int(not r["y"]),
                      "application": r["app"], "trace": r["rel_path"],
                      "confidence_is_lower_bound": op.is_floored(),
                      "threshold_note": "alarm and confidence are both held out by "
                                        "application"},
            note="measured operating point; subject is a real RanSMAP guest. Binding it "
                 "to a stage-5 host is a separate, declared step (tools/rebind_anchors.py).")
        store.emit(anc)
        n += 1
        nfp += not r["y"]
    log(f"wrote {n} A4 anchors ({nfp} on non-ransomware runs) to {out}")


CAVEATS = [
    "SSD-Insider's signal (Baek et al. 2018), not a new detector",
    "out-of-place ransomware (new file + delete original) evades it by construction",
    "RanSMAP has no benign in-place encryptor or database workload; SSD-Insider's own "
    "false alarms came from wiping and database updates",
    "run level on one guest; no sessions, so per-session keying is untested",
    "unit of generalisation: 6 ransomware families, 6 other applications",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--split", choices=["original", "mix"], required=True)
    e.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    e.add_argument("--limit", type=int, default=0)
    sub.add_parser("evaluate")
    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")
    sub.add_parser("mix")
    m = sub.add_parser("emit")
    m.add_argument("--rule", default="R_winshare", choices=["R_winshare", "R_share"])
    m.add_argument("--scheme", default="LOAO", choices=["LOAO", "LOCO", "STRICT"])
    m.add_argument("--prior", type=float, default=0.01)
    m.add_argument("--ttl", type=float, default=3600.0)
    a = ap.parse_args()
    {"extract": cmd_extract, "evaluate": cmd_evaluate, "freeze": cmd_freeze,
     "mix": cmd_mix, "emit": cmd_emit}[a.cmd](a)


if __name__ == "__main__":
    main()
