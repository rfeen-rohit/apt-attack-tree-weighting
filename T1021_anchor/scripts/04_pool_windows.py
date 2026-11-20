"""Pool two or more disjoint evaluation windows into one operating point.

A single window gives a tpr over 5-10 attack paths, which is a point estimate with an
interval so wide it cannot be distinguished from any other window's. Two disjoint
windows of the SAME detector at the SAME budget can have their counts added: the
denominators are independent samples of the same corpus, so the pooled rate is a
better estimate than either, and its interval is narrower.

What this does NOT do: average the two tpr values. 1/10 and 1/5 average to 0.15, which
is a number no experiment produced. The counts are pooled (2 of 15) and the rate is
recomputed from them.

Refusals are inherited, not hidden: if the pooled LR+ is at or below 1, anchors.py
raises and no operating point is written, exactly as for a single window.

    python3 scripts/04_pool_windows.py --tags lanl_d8_b1000 lanl_d26_b1000 --out-tag pooled
"""
import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from common import anchors                                      # noqa: E402

RESULTS = ROOT / "results"
NEEDED = ("redteam_total_paths", "redteam_detected_paths",
          "non_redteam_logons", "alerts_not_redteam")


def wilson(k, n, z=1.96):
    """Wilson score interval. Correct at k=0 and k=n, unlike the normal approximation."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="experiment tags to pool. They must be DISJOINT windows of the "
                         "same detector at the same budget; the script checks both.")
    ap.add_argument("--out-tag", default="pooled")
    ap.add_argument("--prior", type=float, default=0.01)
    a = ap.parse_args()

    runs = []
    for t in a.tags:
        p = RESULTS / f"experiment_{t}.json"
        if not p.exists():
            raise SystemExit(f"{p} not found - run 02_run_experiment.py --tag {t} first")
        R = json.loads(p.read_text())
        base = R["results"]["none"]
        missing = [k for k in NEEDED if base.get(k) is None]
        if missing:
            raise SystemExit(f"{p.name} is missing {missing}; it predates the fields this "
                             f"script pools and cannot be used.")
        runs.append({
            "tag": t,
            "budget": R["config"].get("budget"),
            "window": (R.get("window") or {}).get("from_time"),
            "to": (R.get("window") or {}).get("to_time"),
            "days": (R.get("window") or {}).get("days"),
            **{k: base[k] for k in NEEDED},
            "logons": R["data"]["n_rows"] if "n_rows" in R.get("data", {}) else None,
        })

    # --- the two checks that decide whether pooling is legitimate at all
    budgets = {r["budget"] for r in runs}
    if len(budgets) != 1:
        raise SystemExit(f"REFUSING to pool: the runs use different budgets {sorted(budgets)}. "
                         f"A pooled rate across two different operating points is not an "
                         f"operating point.")
    spans = sorted((r["window"], r["to"], r["tag"]) for r in runs)
    for (f1, t1, n1), (f2, t2, n2) in zip(spans, spans[1:]):
        if None in (f1, t1, f2, t2):
            print(f"  WARNING: {n1} or {n2} records no window, so overlap cannot be checked")
        elif f2 < t1:
            raise SystemExit(f"REFUSING to pool: windows {n1} (t={f1}..{t1}) and {n2} "
                             f"(t={f2}..{t2}) OVERLAP. Pooling counts from overlapping "
                             f"windows double-counts the shared logons.")

    n_pos = sum(r["redteam_total_paths"] for r in runs)
    det = sum(r["redteam_detected_paths"] for r in runs)
    n_neg = sum(r["non_redteam_logons"] for r in runs)
    fp = sum(r["alerts_not_redteam"] for r in runs)
    tpr, fpr = det / n_pos, fp / n_neg

    print(f"pooling {len(runs)} window(s) at budget {budgets.pop()}/day\n")
    print(f"  {'tag':<22}{'window':>22}{'days':>8}{'paths':>9}{'caught':>8}"
          f"{'benign logons':>16}{'false alerts':>14}")
    for r in runs:
        span = "{}..{}".format(r["window"], r["to"])
        print(f"  {r['tag']:<22}{span:>22}"
              f"{r['days']:>8}{r['redteam_total_paths']:>9}{r['redteam_detected_paths']:>8}"
              f"{r['non_redteam_logons']:>16,}{r['alerts_not_redteam']:>14,}")
    print(f"  {'POOLED':<22}{'':>22}{'':>8}{n_pos:>9}{det:>8}{n_neg:>16,}{fp:>14,}")

    lo, hi = wilson(det, n_pos)
    print(f"\n  tpr {tpr:.4f}  (Wilson 95% CI {lo:.3f} - {hi:.3f}, over {n_pos} paths)")
    print(f"  fpr {fpr:.6g}  ({fp:,} / {n_neg:,})")

    op = anchors.OperatingPoint(
        tpr=tpr, fpr=fpr, n_pos=n_pos, n_neg=n_neg,
        detector=(f"T1021 budgeted anomaly, pooled over {len(runs)} disjoint windows "
                  f"({', '.join(r['tag'] for r in runs)})"))
    try:
        print(f"  LR+ {op.lr_plus():.3f}")
        conf = anchors.confidence_from_operating_point(op, a.prior)
        print(f"  at prior {a.prior} an A3 anchor from the pooled point carries "
              f"confidence {conf:.4f}")
    except anchors.AnchorError as e:
        raise SystemExit(f"\nREFUSED, and the refusal stands for the pooled point too: {e}")

    out = RESULTS / f"operating_point_{a.out_tag}.json"
    op.save(out)
    side = RESULTS / f"pooled_{a.out_tag}.json"
    side.write_text(json.dumps({
        "windows": runs, "n_pos": n_pos, "detected": det, "n_neg": n_neg,
        "false_alerts": fp, "tpr": tpr, "tpr_wilson95": [lo, hi], "fpr": fpr,
        "lr_plus": op.lr_plus(), "prior": a.prior,
        "note": ("counts pooled, rate recomputed - the per-window tpr values were NOT "
                 "averaged. Windows were checked for budget equality and non-overlap."),
    }, indent=2))
    print(f"\nwrote {out.name} and {side.name}")


if __name__ == "__main__":
    main()
