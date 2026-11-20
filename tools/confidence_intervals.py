#!/usr/bin/env python3
"""Anchor confidence and its interval, as reported in Table II of the paper.

For each emitter the confidence is the posterior of equation (1),

    LR+ = TPR / FPR,    c = LR+ * pi / (LR+ * pi + 1 - pi),

at a prior pi of 0.01. The bracket uses the Wilson 95% interval of TPR and of
FPR, taking the low TPR with the high FPR for the lower bound and the reverse
for the upper bound. An emitter with LR+ <= 1 is refused.

    python3 tools/confidence_intervals.py            # from the repository root
"""
import json
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PI = 0.01
EMITTERS = [
    ("A2", "T1190 web shells", "evidence/T1190_operating_point_linear_fixed.json"),
    ("A3", "T1021 lateral movement", "evidence/T1021_operating_point_pooled.json"),
    ("A4", "T1486 overwrites", "evidence/T1486_operating_point_overwrite.json"),
]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def post(lr):
    return lr * PI / (lr * PI + 1 - PI)


def main():
    print("%-4s %-24s %6s %9s %5s %8s  %s" % ("link", "emitter", "TPR", "FPR", "n+", "LR+", "c [95% range]"))
    for link, name, rel in EMITTERS:
        op = json.loads((ROOT / rel).read_text())
        tpr, fpr, npos, nneg = op["tpr"], op["fpr"], op["n_pos"], op["n_neg"]
        if tpr == 0 or fpr == 0 or tpr / fpr <= 1:
            print("%-4s %-24s %6.3f %9.1e %5d %8s  refused" % (link, name, tpr, fpr, npos, "-"))
            continue
        lr = tpr / fpr
        tlo, thi = wilson(tpr * npos, npos)
        flo, fhi = wilson(fpr * nneg, nneg)
        lo, hi = post(tlo / fhi), post(thi / flo)
        print("%-4s %-24s %6.3f %9.1e %5d %8.0f  %.2f [%.2f, %.2f]" % (link, name, tpr, fpr, npos, lr, post(lr), lo, hi))


if __name__ == "__main__":
    main()
