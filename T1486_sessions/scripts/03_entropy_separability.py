#!/usr/bin/env python3
"""How far does entropy alone get you on real data? Measured on RanSMAP, no simulation.

This is a standalone result, not a check on the simulator. Nothing here is
generated: every number is computed from the entropy column of RanSMAP's
`ata_write.csv` files, over every row of every run, at block level.

The question it answers is the one the kit's content stage rests on. ERW-Radar and
Casino et al. both argue that an entropy threshold cannot separate encryption from
compression, which is why `detect.py` adds a chi-square test. That argument is
usually made qualitatively. Here it is a number: the ROC-AUC of block entropy as a
ransomware/benign discriminator on 27 million real 4 KB writes, with the overlap of
the two distributions and the rates at the threshold the detector actually uses.

Labels. `dataset/original/` runs have exactly one application under test on an
otherwise idle host, so the run's directory name is what was writing. These are
RUN-LEVEL labels: OS background writes occur in every run and cannot be separated
from the application's own, which is what the `Idle` runs measure and report
below. They are not a per-block process attribution, and the AUC inherits that
limit - it is "a block from a host running ransomware versus a block from a host
running this application", which is the strongest claim this dataset supports.

`dataset/mix/` runs cannot be used for this at all: a benign application and
ransomware write concurrently with no process id recorded, so any per-block label
would have to be inferred from entropy itself, and an AUC computed against
entropy-derived labels measures nothing. When ground truth is unavailable this
script says so and reports only the label-free contrasts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, log, save_json, wilson
import ransmap
from ransmap import (BENIGN_APPS, IDLE_APPS, RANSOMWARE_APPS, Hist, auc, ks,
                     overlap, rate_at_threshold, threshold_sweep)

# detect.py's content test, on RanSMAP's normalised scale
ENT_MIN_BITS = 7.5
ENT_MIN_NORM = ENT_MIN_BITS / ransmap.BITS_PER_SYMBOL          # 0.9375
PCTS = (1, 5, 25, 50, 75, 95, 99)


def describe(h: Hist, n_runs, label):
    q = h.quantiles()
    return {"label": label, "n_runs": n_runs, "n_blocks": h.n,
            "mean": h.mean, "sd": h.sd, "min": h.mn, "max": h.mx,
            "percentiles": {f"p{p:02d}": q[p] for p in PCTS},
            "mean_bits": None if h.mean is None else h.mean * ransmap.BITS_PER_SYMBOL,
            "frac_at_or_above_detector_ent_min": rate_at_threshold(h, ENT_MIN_NORM)}


def best_points(pos: Hist, neg: Hist):
    """The best achievable single-threshold operating points, with denominators."""
    sweep = threshold_sweep(pos, neg)
    if not sweep:
        return None
    youden = max(sweep, key=lambda r: r["youden_j"])
    out = {"n_pos_blocks": pos.n, "n_neg_blocks": neg.n,
           "best_youden": {"threshold": youden["t"],
                           "threshold_bits": youden["t"] * ransmap.BITS_PER_SYMBOL,
                           "tpr": wilson(youden["tp"], pos.n),
                           "fpr": wilson(youden["fp"], neg.n),
                           "youden_j": youden["youden_j"]}}
    for cap in (0.01, 0.05):
        ok = [r for r in sweep if r["fpr"] <= cap]
        best = max(ok, key=lambda r: r["tpr"]) if ok else None
        out[f"tpr_at_fpr_le_{cap:g}"] = (None if best is None else
                                         {"threshold": best["t"],
                                          "tpr": wilson(best["tp"], pos.n),
                                          "fpr": wilson(best["fp"], neg.n)})
    at = next((r for r in sweep if r["t"] >= ENT_MIN_NORM), None)
    out["at_detector_ent_min"] = (None if at is None else
                                  {"threshold": ENT_MIN_NORM,
                                   "threshold_bits": ENT_MIN_BITS,
                                   "tpr": wilson(at["tp"], pos.n),
                                   "fpr": wilson(at["fp"], neg.n)})
    return out


def compare(pos: Hist, neg: Hist, pos_label, neg_label):
    return {"positive": pos_label, "negative": neg_label,
            "auc": auc(pos, neg), "overlap_coefficient": overlap(pos, neg),
            "ks_distance": ks(pos, neg),
            "operating_points": best_points(pos, neg)}


def run_level_auc(pos_runs, neg_runs):
    """A coarser but cleaner unit: one mean entropy per RUN, which is the unit the
    labels are actually attached to. 10 runs per application, so the denominators
    are small and are reported rather than smoothed over."""
    p = sorted(r["hist"].mean for r in pos_runs if r["hist"].n)
    q = sorted(r["hist"].mean for r in neg_runs if r["hist"].n)
    if not p or not q:
        return None
    tot = sum(sum(1.0 if a > b else 0.5 if a == b else 0.0 for b in q) for a in p)
    return {"auc": tot / (len(p) * len(q)), "n_pos_runs": len(p), "n_neg_runs": len(q),
            "unit": "one run, scored by its mean block entropy"}


def main():
    runs, prov = ransmap.load_original_runs()
    out = {"schema": "t1486/separability/1",
           "question": ("ROC-AUC of 4 KB-block write entropy alone as a "
                        "ransomware/benign discriminator, on real RanSMAP traces"),
           "entropy_scale": ("RanSMAP column 5 is NORMALISED 0-1, not 0-8 bits; x8 for "
                             "bits. detect.py's ent_min=7.5 bits is 0.9375 here."),
           "detector_thresholds": {"ent_min_bits": ENT_MIN_BITS,
                                   "ent_min_normalised": ENT_MIN_NORM,
                                   "chi_max": 350.0},
           "label_provenance": prov}

    if not runs:
        log("NO GROUND TRUTH AVAILABLE.")
        log("dataset/original/ is not present and no entropy extract was found, so no")
        log("per-block ransomware/benign label exists. A block-level AUC is therefore")
        log("NOT COMPUTABLE: dataset/mix mixes both processes with no process id, and an")
        log("AUC against labels inferred from entropy would only restate the inference.")
        log("Reporting the label-free contrast between mix COMBOs instead.")
        fit = ransmap.load_profiles(verbose=False)
        combos = {}
        for combo, d in fit["mix"].items():
            combos[combo] = {k: v for k, v in d.items() if k != "runs"}
        out["status"] = "no_ground_truth"
        out["block_level_auc"] = None
        out["why_no_auc"] = ("dataset/original/ absent: no per-block label exists and a "
                             "label inferred from entropy would make the AUC circular")
        out["mix_combos"] = combos
        save_json(RESULTS / "separability.json", out)
        return

    by_app = {}
    for r in runs:
        by_app.setdefault(r["app"], []).append(r)

    def pooled(apps):
        h, n = Hist(), 0
        for a in apps:
            for r in by_app.get(a, []):
                h.merge(r["hist"])
                n += 1
        return h, n

    present_rans = [a for a in RANSOMWARE_APPS if a in by_app]
    present_benign = [a for a in BENIGN_APPS if a in by_app]
    present_idle = [a for a in IDLE_APPS if a in by_app]

    rans_h, rans_n = pooled(present_rans)
    ben_h, ben_n = pooled(present_benign)

    out["classes"] = {"ransomware": present_rans, "benign": present_benign,
                      "idle_reported_separately": present_idle}
    out["distributions"] = {
        a: describe(_merge([r["hist"] for r in rs]), len(rs), a)
        for a, rs in sorted(by_app.items())}
    out["distributions"]["ALL_ransomware"] = describe(rans_h, rans_n, "ALL_ransomware")
    out["distributions"]["ALL_benign"] = describe(ben_h, ben_n, "ALL_benign")

    out["pooling_note"] = (
        "'pooled' weights every BLOCK equally, so the applications that write most "
        "decide the result: AESCrypt and Zip are 88% of the benign blocks. The "
        "equal_weight variants give each application the same total weight instead. "
        "Neither is a measured base rate; both are reported so the headline cannot be "
        "read as one.")
    out["headline"] = compare(rans_h, ben_h, "ransomware (pooled, block-weighted)",
                              "benign (pooled, block-weighted)")
    out["headline"]["run_level"] = run_level_auc(
        [r for a in present_rans for r in by_app[a]],
        [r for a in present_benign for r in by_app[a]])

    rans_eq = _equal_weight({a: _merge([r["hist"] for r in by_app[a]])
                             for a in present_rans})
    ben_eq = _equal_weight({a: _merge([r["hist"] for r in by_app[a]])
                            for a in present_benign})
    out["headline_equal_weight"] = compare(
        rans_eq, ben_eq, "ransomware (equal weight per family)",
        "benign (equal weight per application)")
    out["headline_equal_weight"]["weighting"] = (
        "each application rescaled to the same total weight; see pooling_note")

    out["ransomware_vs_each_benign_app"] = {
        a: compare(rans_h, _merge([r["hist"] for r in by_app[a]]),
                   "ransomware (pooled)", a) for a in present_benign}
    out["each_family_vs_pooled_benign"] = {
        a: compare(_merge([r["hist"] for r in by_app[a]]), ben_h, a, "benign (pooled)")
        for a in present_rans}
    if present_idle:
        idle_h, idle_n = pooled(present_idle)
        out["os_background_only"] = {
            "note": ("Idle runs have no application under test, so this is the OS "
                     "background write distribution that contaminates every other run "
                     "and sets a floor on how clean any run-level label can be."),
            "distribution": describe(idle_h, idle_n, "Idle"),
            "ransomware_vs_idle": compare(rans_h, idle_h, "ransomware (pooled)", "Idle")}

    save_json(RESULTS / "separability.json", out)
    _print(out, by_app, present_rans, present_benign, present_idle)


def _merge(hists):
    h = Hist()
    for x in hists:
        h.merge(x)
    return h


def _equal_weight(hists_by_app, target=1_000_000):
    """Pool several applications giving each the SAME total weight.

    Pooling by block count lets the busiest applications decide the result: the
    benign set is 88% AESCrypt and Zip blocks simply because those two write more.
    How much each application contributes is an evaluator's choice, not a measured
    base rate, so every pooled number is reported both ways. Only relative weights
    matter to AUC, overlap and KS, so rescaling each application to a common total
    is the whole of it.
    """
    out = Hist()
    for app, h in hists_by_app.items():
        if not h.n:
            continue
        scale = target / h.n
        for b, c in h.counts.items():
            k = int(round(c * scale))
            if k:
                v = (b + 0.5) / ransmap.NBINS
                out.counts[b] = out.counts.get(b, 0) + k
                out.n += k
                out.s += k * v
                out.ss += k * v * v
        out.mn = h.mn if out.mn is None else min(out.mn, h.mn)
        out.mx = h.mx if out.mx is None else max(out.mx, h.mx)
    return out


def _print(out, by_app, rans, benign, idle):
    def f(x, w=7, d=4):
        return " " * (w - 1) + "-" if x is None else f"{x:{w}.{d}f}"

    log("")
    log("=" * 100)
    log("ENTROPY ALONE AS A RANSOMWARE DISCRIMINATOR - REAL RanSMAP WRITE TRACES")
    log("=" * 100)
    p = out["label_provenance"]
    log(f"labels      : {p.get('label_source')}  ({p.get('n_runs')} runs, one "
        f"application per run)")
    ep = (p.get("extract_provenance") or {})
    if ep:
        log(f"source      : {ep.get('dataset', '')}")
        log(f"commit      : {ep.get('git_commit', '-')}   pattern "
            f"{ep.get('sparse_pattern', '-')}")
    log(f"scale       : entropy normalised 0-1 (x8 for bits). detect.py's ent_min "
        f"7.5 bits = {ENT_MIN_NORM}")
    log("")
    log("--- per-application distribution of 4 KB write-block entropy ---")
    log(f"  {'application':12s} {'kind':11s} {'runs':>4s} {'blocks':>10s} {'mean':>7s} "
        f"{'sd':>7s} {'p05':>7s} {'p50':>7s} {'p95':>7s} {'>=0.9375':>9s}")
    order = [(a, "ransomware") for a in rans] + [(a, "benign") for a in benign] \
        + [(a, "idle") for a in idle]
    for a, kind in order:
        d = out["distributions"][a]
        r = d["frac_at_or_above_detector_ent_min"]
        log(f"  {a:12s} {kind:11s} {d['n_runs']:4d} {d['n_blocks']:10d} "
            f"{f(d['mean'])} {f(d['sd'])} {f(d['percentiles']['p05'])} "
            f"{f(d['percentiles']['p50'])} {f(d['percentiles']['p95'])} "
            f"{r['rate']*100:8.2f}%")
    for name in ("ALL_ransomware", "ALL_benign"):
        d = out["distributions"][name]
        r = d["frac_at_or_above_detector_ent_min"]
        log(f"  {name:12s} {'pooled':11s} {d['n_runs']:4d} {d['n_blocks']:10d} "
            f"{f(d['mean'])} {f(d['sd'])} {f(d['percentiles']['p05'])} "
            f"{f(d['percentiles']['p50'])} {f(d['percentiles']['p95'])} "
            f"{r['rate']*100:8.2f}%")

    h = out["headline"]
    op = h["operating_points"]
    log("")
    log("--- HEADLINE: entropy alone, pooled ransomware vs pooled benign ---")
    log(f"  blocks            {op['n_pos_blocks']} ransomware / {op['n_neg_blocks']} benign")
    log(f"  ROC-AUC           {h['auc']:.4f}        (0.5 = useless, 1.0 = perfect)")
    log(f"  overlap coeff.    {h['overlap_coefficient']:.4f}        "
        f"(shared area of the two densities)")
    log(f"  KS distance       {h['ks_distance']:.4f}")
    y = op["best_youden"]
    log(f"  best threshold    >= {y['threshold']:.4f} ({y['threshold_bits']:.3f} bits): "
        f"TPR {y['tpr']['rate']*100:.2f}% ({y['tpr']['k']}/{y['tpr']['n']})  "
        f"FPR {y['fpr']['rate']*100:.2f}% ({y['fpr']['k']}/{y['fpr']['n']})")
    for cap in ("0.01", "0.05"):
        z = op[f"tpr_at_fpr_le_{cap}"]
        if z:
            log(f"  at FPR <= {cap:>4s}     >= {z['threshold']:.4f}: "
                f"TPR {z['tpr']['rate']*100:6.2f}% ({z['tpr']['k']}/{z['tpr']['n']})  "
                f"FPR {z['fpr']['rate']*100:.3f}% ({z['fpr']['k']}/{z['fpr']['n']})")
    z = op["at_detector_ent_min"]
    if z:
        log(f"  at detect.py's    >= {z['threshold']:.4f} ({z['threshold_bits']} bits): "
            f"TPR {z['tpr']['rate']*100:6.2f}% ({z['tpr']['k']}/{z['tpr']['n']})  "
            f"FPR {z['fpr']['rate']*100:6.2f}% ({z['fpr']['k']}/{z['fpr']['n']})")
    rl = h.get("run_level")
    if rl:
        log(f"  run-level AUC     {rl['auc']:.4f}        "
            f"({rl['n_pos_runs']} vs {rl['n_neg_runs']} runs, scored by mean entropy)")
    e = out["headline_equal_weight"]
    ey = e["operating_points"]["best_youden"]
    log("")
    log("--- same comparison, each application weighted EQUALLY rather than by blocks ---")
    log(f"  ROC-AUC           {e['auc']:.4f}   overlap {e['overlap_coefficient']:.4f}   "
        f"KS {e['ks_distance']:.4f}")
    log(f"  best threshold    >= {ey['threshold']:.4f}: TPR {ey['tpr']['rate']*100:.2f}%  "
        f"FPR {ey['fpr']['rate']*100:.2f}%")
    ez = e["operating_points"]["at_detector_ent_min"]
    if ez:
        log(f"  at detect.py's    >= {ez['threshold']:.4f}: "
            f"TPR {ez['tpr']['rate']*100:6.2f}%  FPR {ez['fpr']['rate']*100:6.2f}%")
    log(f"  {out['pooling_note']}")

    log("")
    log("--- pooled ransomware vs EACH benign application ---")
    log(f"  {'benign app':12s} {'AUC':>7s} {'overlap':>8s} {'KS':>7s} "
        f"{'best t':>7s} {'TPR@best':>9s} {'FPR@best':>9s} {'n benign blocks':>16s}")
    for a, c in out["ransomware_vs_each_benign_app"].items():
        y = c["operating_points"]["best_youden"]
        log(f"  {a:12s} {c['auc']:7.4f} {c['overlap_coefficient']:8.4f} "
            f"{c['ks_distance']:7.4f} {y['threshold']:7.4f} "
            f"{y['tpr']['rate']*100:8.2f}% {y['fpr']['rate']*100:8.2f}% "
            f"{c['operating_points']['n_neg_blocks']:16d}")

    log("")
    log("--- EACH ransomware family vs pooled benign ---")
    log(f"  {'family':12s} {'AUC':>7s} {'overlap':>8s} {'KS':>7s} "
        f"{'TPR@0.9375':>11s} {'FPR@0.9375':>11s} {'n blocks':>11s}")
    for a, c in out["each_family_vs_pooled_benign"].items():
        z = c["operating_points"]["at_detector_ent_min"]
        log(f"  {a:12s} {c['auc']:7.4f} {c['overlap_coefficient']:8.4f} "
            f"{c['ks_distance']:7.4f} {z['tpr']['rate']*100:10.2f}% "
            f"{z['fpr']['rate']*100:10.2f}% "
            f"{c['operating_points']['n_pos_blocks']:11d}")

    if "os_background_only" in out:
        o = out["os_background_only"]
        d = o["distribution"]
        log("")
        log("--- OS background only (Idle runs: no application under test) ---")
        log(f"  {d['n_runs']} runs, {d['n_blocks']} blocks, mean {d['mean']:.4f}, "
            f"p95 {d['percentiles']['p95']:.4f}")
        log(f"  ransomware vs idle AUC {o['ransomware_vs_idle']['auc']:.4f}")
        log(f"  {o['note']}")

    a = h["auc"]
    log("")
    log("--- what this means for the kit ---")
    log(f"  Entropy alone reaches AUC {a:.4f} at block level ({e['auc']:.4f} with every")
    log("  application weighted equally). The benign applications")
    log("  that overlap the ransomware are the ones that produce high-entropy output for")
    log("  legitimate reasons: compression (Zip) and user-driven encryption (AESCrypt).")
    log("  This is the measurement behind detect.py's second test, and it is also why a")
    log("  payload model that invents its own entropy cannot be used to evaluate that")
    log("  test: the separation it reports is a property of the generator.")
    log("")
    log("  Caveat carried from the labels: these are RUN-level labels, so a 'ransomware'")
    log("  block may be an OS background write on a host where ransomware was running,")
    log("  and vice versa. The Idle row bounds how much of the overlap that explains.")
    log("")
    log("next: python3 scripts/01_run_experiment.py --payload-model ransmap")


if __name__ == "__main__":
    main()
