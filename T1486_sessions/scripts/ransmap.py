#!/usr/bin/env python3
"""Drive the T1486 payload model from REAL measured storage traces (RanSMAP).

Why this module exists
----------------------
`scripts/workload.py` used to invent its payloads. A benign "archive" write was
`zlib.compress()` of a loop over eleven English words, and the ransomware wrote
`rng.randbytes()`. Both are caricatures: dictionary-compressed English is far more
compressible than real compressed output, and `randbytes` is closer to uniform
than any real cipher stream seen through a block device. The detector's content
stage (Shannon entropy plus chi-square against uniform) therefore separated the
two almost perfectly, and the resulting numbers measured the generator, not the
detector.

This module replaces the invented entropy with an empirical distribution measured
on RanSMAP (Hirano et al.), the public storage-access dataset the kit's README
already names as the obvious next step.

The entropy scale, which is the easiest thing to get wrong
----------------------------------------------------------
RanSMAP's `ata_write.csv` column 5 is entropy **normalised to 0-1**, not 0-8
bits. `scripts/detect.py` and `scripts/common.py` work in BITS (0-8). Everything
in this module is in the RanSMAP normalised scale unless a name says `_bits`, and
`BITS_PER_SYMBOL` is the only conversion factor. The detector's default
`ent_min=7.5` bits is **0.9375** on this scale.

What can and cannot be labelled
-------------------------------
`dataset/mix/<cpu>/<ram>/<COMBO>/<run>/ata_write.csv` is block level and single
host: one row per 4 KB block written, no filenames, no renames, no sessions and
**no process id**. A mix run has a benign application and ransomware writing
concurrently, so an individual block in a mix run CANNOT be labelled benign or
malicious. `load_profiles` therefore prefers `dataset/original/`, where exactly
one application is under test per run, so the run name is what was writing.

Two provenance modes, and the code says which one produced a profile:

  `original_runs`  run-level ground truth from `dataset/original/`. Preferred.
  `mix_otsu`       APPROXIMATION. Each mix run's entropy histogram is split into
                   two components by the threshold that minimises within-group
                   variance (Otsu), the low component called benign and the high
                   one ransomware. This is a fitted guess, not ground truth, and
                   is labelled as such in the code, the JSON and the console.

The honest limit on the chi-square side
---------------------------------------
RanSMAP records entropy per block and nothing about the byte histogram, so the
**entropy** side of the content test can be driven by real data and the
**chi-square** side cannot. `synthesise_block` produces bytes that hit a measured
entropy target; their chi-square against uniform is then a property of the
synthesis family (a truncated-exponential symbol distribution), not a measured
property of real ransomware. `--build` measures and records that entropy/chi-square
curve so a reader can see exactly what the synthesis implies, but it must not be
read as evidence about real ciphertext.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import random
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

try:                                     # optional: only speeds things up
    import numpy as _np
except Exception:                        # pragma: no cover
    _np = None

HERE = Path(__file__).resolve().parent
KIT = HERE.parent                                      # T1486_sessions
IMPL = KIT.parent                                      # Implementation
DEFAULT_RANSMAP_ROOT = IMPL / "_shared_data" / "RanSMAP"
DEFAULT_EXTRACT = (IMPL / "_shared_data" / "RanSMAP_original_entropy"
                   / "ransmap_original_entropy.json.gz")
PROFILES_PATH = KIT / "data" / "entropy_profiles.json"
MIX_CACHE_PATH = KIT / "data" / "mix_histograms.json.gz"

BITS_PER_SYMBOL = 8.0          # the only entropy-scale conversion in the kit
NBINS = 10000                  # histogram resolution: bin width 1e-4 on the 0-1 scale
N_QUANTILES = 101              # the persisted quantile grid, p = 0.00 .. 1.00

# RanSMAP's own application names. The six ransomware families and the benign
# applications are kept apart here and nowhere else, so a reader can check the
# class assignment in one place.
RANSOMWARE_APPS = ("Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry")
BENIGN_APPS = ("AESCrypt", "Firefox", "Office", "Zip")
IDLE_APPS = ("Idle",)          # no application under test: OS background writes only

MIX_COMBOS = ("AESCrypt_Conti", "AESCrypt_REvil", "Firefox_Conti",
              "Firefox_Office_REvil", "Firefox_REvil", "Office_REvil",
              "SDelete_Conti", "SDelete_REvil", "Zip_Conti", "Zip_REvil")


def log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# histograms: the one internal representation for an entropy distribution
# --------------------------------------------------------------------------- #
class Hist:
    """A sparse histogram of normalised entropy over NBINS bins of [0, 1].

    Sparse because a run touches a few thousand of the 10000 bins at most, and
    because it has to survive a round trip through JSON. Exact counts, exact
    first and second moments and the exact min and max are all kept, so every
    number this module reports carries its own denominator.
    """

    __slots__ = ("counts", "n", "s", "ss", "mn", "mx")

    def __init__(self):
        self.counts: dict[int, int] = {}
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.mn = None
        self.mx = None

    # -- construction ------------------------------------------------------ #
    def add_values(self, vals):
        if not len(vals):
            return
        if _np is not None:
            e = _np.asarray(vals, dtype=_np.float64)
            self.n += int(e.size)
            self.s += float(e.sum())
            self.ss += float((e * e).sum())
            lo, hi = float(e.min()), float(e.max())
            idx = _np.clip((e * NBINS).astype(_np.int64), 0, NBINS - 1)
            for b, c in zip(*_np.unique(idx, return_counts=True)):
                b = int(b)
                self.counts[b] = self.counts.get(b, 0) + int(c)
        else:
            lo, hi = min(vals), max(vals)
            for v in vals:
                self.n += 1
                self.s += v
                self.ss += v * v
                b = int(v * NBINS)
                b = 0 if b < 0 else (NBINS - 1 if b >= NBINS else b)
                self.counts[b] = self.counts.get(b, 0) + 1
        self.mn = lo if self.mn is None else min(self.mn, lo)
        self.mx = hi if self.mx is None else max(self.mx, hi)

    def merge(self, other: "Hist") -> "Hist":
        for b, c in other.counts.items():
            self.counts[b] = self.counts.get(b, 0) + c
        self.n += other.n
        self.s += other.s
        self.ss += other.ss
        if other.mn is not None:
            self.mn = other.mn if self.mn is None else min(self.mn, other.mn)
            self.mx = other.mx if self.mx is None else max(self.mx, other.mx)
        return self

    # -- summaries --------------------------------------------------------- #
    @property
    def mean(self):
        return None if not self.n else self.s / self.n

    @property
    def sd(self):
        if not self.n:
            return None
        m = self.s / self.n
        return math.sqrt(max(0.0, self.ss / self.n - m * m))

    def _cum(self):
        bins = sorted(self.counts)
        cum, c = [], 0
        for b in bins:
            c += self.counts[b]
            cum.append(c)
        return bins, cum

    def quantiles(self, k=N_QUANTILES):
        """Sorted empirical quantiles at p = 0, 1/(k-1), ... 1.

        Interior quantiles land on a bin centre, so they carry the histogram's
        1e-4 resolution; p=0 and p=1 use the exact min and max that were
        recorded while streaming, so the range is never narrowed by binning.
        """
        if not self.n:
            return [0.0] * k
        bins, cum = self._cum()
        out = []
        for i in range(k):
            if i == 0:
                out.append(self.mn)
                continue
            if i == k - 1:
                out.append(self.mx)
                continue
            j = min(bisect.bisect_left(cum, i / (k - 1) * self.n), len(bins) - 1)
            v = (bins[j] + 0.5) / NBINS
            out.append(min(max(v, self.mn), self.mx))
        return out

    def to_json(self):
        return {"n": self.n, "sum": self.s, "sumsq": self.ss, "min": self.mn,
                "max": self.mx, "nbins": NBINS,
                "hist": [[b, self.counts[b]] for b in sorted(self.counts)]}

    @staticmethod
    def from_json(d) -> "Hist":
        h = Hist()
        h.n = int(d["n"])
        h.s = float(d["sum"])
        h.ss = float(d["sumsq"])
        h.mn = d.get("min")
        h.mx = d.get("max")
        h.counts = {int(b): int(c) for b, c in d["hist"]}
        return h


# --------------------------------------------------------------------------- #
# two-sample statistics, computed exactly from the histograms
# --------------------------------------------------------------------------- #
def auc(pos: Hist, neg: Hist):
    """ROC-AUC of entropy as a score, Mann-Whitney U with the tie correction.

    Ties inside a bin get the usual half credit, which is what makes this the
    probability that a random positive scores above a random negative rather
    than an optimistic version of it.
    """
    if not pos.n or not neg.n:
        return None
    below, tot = 0, 0.0
    for b in sorted(set(pos.counts) | set(neg.counts)):
        p, q = pos.counts.get(b, 0), neg.counts.get(b, 0)
        tot += p * (below + 0.5 * q)
        below += q
    return tot / (pos.n * neg.n)


def overlap(pos: Hist, neg: Hist):
    """Overlapping coefficient: the shared area of the two densities, 0 to 1.

    1.0 means the two distributions are indistinguishable at this resolution.
    """
    if not pos.n or not neg.n:
        return None
    return sum(min(pos.counts.get(b, 0) / pos.n, neg.counts.get(b, 0) / neg.n)
               for b in set(pos.counts) | set(neg.counts))


def ks(pos: Hist, neg: Hist):
    """Kolmogorov-Smirnov distance between the two empirical CDFs."""
    if not pos.n or not neg.n:
        return None
    cp = cq = 0
    best = 0.0
    for b in sorted(set(pos.counts) | set(neg.counts)):
        cp += pos.counts.get(b, 0)
        cq += neg.counts.get(b, 0)
        best = max(best, abs(cp / pos.n - cq / neg.n))
    return best


def threshold_sweep(pos: Hist, neg: Hist):
    """Every achievable operating point of the rule `entropy >= t`.

    Returns a list of {t, tpr, fpr, tp, fp, n_pos, n_neg, youden_j}, one per bin
    boundary that either class occupies, so TPR at a fixed FPR and the best
    Youden point can both be read off with their denominators intact.
    """
    if not pos.n or not neg.n:
        return []
    bins = sorted(set(pos.counts) | set(neg.counts))
    tp, fp = pos.n, neg.n            # t at or below the lowest bin: everything alerts
    rows = []
    for b in bins:
        t = b / NBINS
        rows.append({"t": round(t, 6), "tp": tp, "fp": fp,
                     "tpr": tp / pos.n, "fpr": fp / neg.n,
                     "n_pos": pos.n, "n_neg": neg.n,
                     "youden_j": tp / pos.n - fp / neg.n})
        tp -= pos.counts.get(b, 0)
        fp -= neg.counts.get(b, 0)
    return rows


def rate_at_threshold(h: Hist, t: float):
    """Fraction of blocks with entropy >= t, and the numerator and denominator."""
    if not h.n:
        return {"rate": None, "k": 0, "n": 0}
    b0 = int(t * NBINS)
    k = sum(c for b, c in h.counts.items() if b >= b0)
    return {"rate": k / h.n, "k": k, "n": h.n}


def otsu_split(h: Hist):
    """Split a histogram into two components at the minimum within-group variance.

    This is Otsu's method. It is used ONLY in the `mix_otsu` fallback, where no
    per-block label exists, and the number it returns is a fitted guess about
    which blocks came from the ransomware. Nothing downstream may treat it as
    ground truth.
    """
    if not h.n:
        return None
    bins = sorted(h.counts)
    w0 = s0 = ss0 = 0.0
    wt = float(h.n)
    st, sst = h.s, h.ss
    best = None
    for b in bins:                      # threshold = the top edge of bin b
        c = h.counts[b]
        v = (b + 0.5) / NBINS
        w0 += c
        s0 += c * v
        ss0 += c * v * v
        w1 = wt - w0
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = s0 / w0
        var0 = max(0.0, ss0 / w0 - m0 * m0)
        s1, ss1 = st - s0, sst - ss0
        m1 = s1 / w1
        var1 = max(0.0, ss1 / w1 - m1 * m1)
        within = (w0 * var0 + w1 * var1) / wt
        if best is None or within < best[0]:
            best = (within, (b + 1) / NBINS, w0, w1, m0, m1)
    if best is None:
        return None
    within, t, w0, w1, m0, m1 = best
    return {"threshold": round(t, 6), "within_group_variance": within,
            "n_low": int(w0), "n_high": int(w1),
            "share_high": w1 / wt, "mean_low": m0, "mean_high": m1}


def split_at(h: Hist, t: float):
    """Two Hists, below and at-or-above `t`. Moments are recomputed from the bins,
    so a component's mean and sd are bin-resolution accurate rather than exact."""
    b0 = int(t * NBINS)
    lo, hi = Hist(), Hist()
    for b, c in h.counts.items():
        tgt = hi if b >= b0 else lo
        v = (b + 0.5) / NBINS
        tgt.counts[b] = tgt.counts.get(b, 0) + c
        tgt.n += c
        tgt.s += c * v
        tgt.ss += c * v * v
    for part in (lo, hi):
        if part.counts:
            part.mn = min(part.counts) / NBINS
            part.mx = (max(part.counts) + 1) / NBINS
    return lo, hi


# --------------------------------------------------------------------------- #
# reading RanSMAP
# --------------------------------------------------------------------------- #
def _hist_from_csv(path: Path, chunk_bytes=32 << 20):
    """Stream one ata_write.csv into a Hist. Every row is read; nothing is sampled.

    Row format: unix_sec, unix_nsec, LBA, size, entropy, unused. Column 5 is the
    normalised entropy. Short or unparsable rows are counted and reported rather
    than silently dropped.
    """
    h, bad, sizes = Hist(), 0, Counter()
    with path.open("rb") as fh:
        while True:
            lines = fh.readlines(chunk_bytes)
            if not lines:
                break
            vals = []
            for ln in lines:
                f = ln.split(b",")
                if len(f) < 5:
                    bad += 1
                    continue
                try:
                    vals.append(float(f[4]))
                except ValueError:
                    bad += 1
                    continue
                sizes[f[3].strip()] += 1
            h.add_values(vals)
    return h, bad, {k.decode(): v for k, v in sizes.most_common(8)}


def discover_mix_runs(ransmap_root: Path, combos=None):
    """[(combo, path)] for every mix run present, filtered to `combos` if given."""
    base = Path(ransmap_root) / "dataset" / "mix"
    out = []
    for p in sorted(base.rglob("ata_write.csv")):
        combo = p.parent.parent.name
        if combos and combo not in combos:
            continue
        out.append((combo, p))
    return out


def discover_original_runs(ransmap_root: Path):
    """[(app, path)] for every single-application run present under dataset/original."""
    base = Path(ransmap_root) / "dataset" / "original"
    return [(p.parent.parent.name, p) for p in sorted(base.rglob("ata_write.csv"))]


def split_combo(combo: str):
    """'Firefox_Office_REvil' -> (['Firefox', 'Office'], 'REvil')."""
    parts = combo.split("_")
    return parts[:-1], parts[-1]


def load_original_extract(path=DEFAULT_EXTRACT):
    """Per-run Hists for dataset/original from the precomputed entropy extract.

    The extract holds a full-data histogram per run (every row of every
    ata_write.csv, 10000 bins) plus the exact moments, built because
    dataset/original is ~1.8 GB for this hardware while only the entropy
    distribution is needed here. It is NOT a subsample. Its own `provenance`
    block travels into entropy_profiles.json unchanged.
    """
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        d = json.load(fh)
    runs = []
    for r in d["runs"]:
        h = Hist()
        h.n = int(r["n_blocks"])
        h.s = float(r["sum"])
        h.ss = float(r["sumsq"])
        h.mn, h.mx = r.get("min"), r.get("max")
        h.counts = {int(b): int(c) for b, c in r["hist"]}
        runs.append({"app": r["app"], "run": r["run"], "rel_path": r["rel_path"],
                     "cpu": r.get("cpu"), "ram": r.get("ram"),
                     "sizes": r.get("sizes", {}), "hist": h})
    return {"runs": runs, "provenance": d.get("provenance", {}),
            "source": str(path)}


def load_original_runs(ransmap_root=DEFAULT_RANSMAP_ROOT, extract=DEFAULT_EXTRACT):
    """Ground-truth per-run histograms, from real CSVs if present, else the extract.

    Returns (runs, provenance) or (None, why_not).
    """
    csvs = discover_original_runs(ransmap_root)
    if csvs:
        runs, bad_total = [], 0
        for app, p in csvs:
            h, bad, sizes = _hist_from_csv(p)
            bad_total += bad
            runs.append({"app": app, "run": p.parent.name,
                         "rel_path": str(p.relative_to(Path(ransmap_root) / "dataset"
                                                       / "original")),
                         "sizes": sizes, "hist": h})
        return runs, {"label_source": "original_csv",
                      "what": "dataset/original ata_write.csv read in full on this machine",
                      "n_runs": len(runs), "unparsable_rows": bad_total,
                      "root": str(Path(ransmap_root) / "dataset" / "original")}
    ext = load_original_extract(extract) if extract else None
    if ext:
        return ext["runs"], {"label_source": "original_extract",
                             "what": ("full-data per-run entropy histograms of "
                                      "dataset/original ata_write.csv"),
                             "extract_file": ext["source"],
                             "extract_provenance": ext["provenance"],
                             "n_runs": len(ext["runs"])}
    return None, {"label_source": None,
                  "why_not": (f"no dataset/original under {ransmap_root} and no extract "
                              f"at {extract}")}


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #
def _profile(name, kind, hist: Hist, n_runs, sources, label_source, note=""):
    return {"name": name, "kind": kind, "label_source": label_source,
            "n_blocks": hist.n, "n_runs": n_runs,
            "mean": hist.mean, "sd": hist.sd, "min": hist.mn, "max": hist.mx,
            "quantiles_p": [i / (N_QUANTILES - 1) for i in range(N_QUANTILES)],
            "quantiles": hist.quantiles(),
            "sources": sources, "note": note,
            "scale": "normalised entropy 0-1 as shipped by RanSMAP; x8 for bits"}


def load_profiles(root=DEFAULT_RANSMAP_ROOT, combos=None, extract=DEFAULT_EXTRACT,
                  use_mix_cache=True, verbose=True):
    """Fit one empirical entropy distribution per class from real RanSMAP traces.

    `root` is the RanSMAP checkout (the directory containing `dataset/`).
    `combos` restricts which mix COMBOs are read; None means all present.

    The mix runs are always walked, because they are what the kit's scenario
    mirrors and because they are what the `mix_otsu` fallback needs. The CLASS
    LABELS, however, come from `dataset/original` whenever it is available, since
    a mix run has a benign application and ransomware writing at the same time
    with no process id to tell them apart. When only `mix/` is present the
    fallback splits each mix run by Otsu and says loudly that it is doing so.
    """
    root = Path(root)
    t0 = time.time()

    # ---- the mix runs: real, unlabelled, one histogram per run
    mix_runs = discover_mix_runs(root, combos)
    cache = None
    if use_mix_cache and MIX_CACHE_PATH.exists():
        try:
            with gzip.open(MIX_CACHE_PATH, "rt") as fh:
                c = json.load(fh)
            if c.get("nbins") == NBINS:
                cache = {r["rel_path"]: r for r in c["runs"]}
        except Exception:
            cache = None
    mix = []
    bad_rows = 0
    for i, (combo, p) in enumerate(mix_runs, 1):
        rel = str(p.relative_to(root))
        hit = cache.get(rel) if cache else None
        if hit and hit.get("mtime") == int(p.stat().st_mtime) and hit.get("bytes") == p.stat().st_size:
            h, bad, sizes = Hist.from_json(hit["hist"]), hit.get("bad", 0), hit.get("sizes", {})
        else:
            h, bad, sizes = _hist_from_csv(p)
            if verbose:
                log(f"  [{i}/{len(mix_runs)}] {combo}/{p.parent.name} "
                    f"{h.n} blocks  mean {h.mean:.4f}")
        bad_rows += bad
        mix.append({"combo": combo, "run": p.parent.name, "rel_path": rel,
                    "sizes": sizes, "bad": bad, "hist": h,
                    "mtime": int(p.stat().st_mtime), "bytes": p.stat().st_size})
    if use_mix_cache and mix:
        MIX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(MIX_CACHE_PATH, "wt") as fh:
            json.dump({"nbins": NBINS, "runs": [
                {"rel_path": m["rel_path"], "mtime": m["mtime"], "bytes": m["bytes"],
                 "bad": m["bad"], "sizes": m["sizes"], "hist": m["hist"].to_json()}
                for m in mix]}, fh)

    # ---- the labels
    orig, orig_prov = load_original_runs(root, extract)
    profiles, prov = {}, {}

    if orig:
        by_app = {}
        for r in orig:
            by_app.setdefault(r["app"], []).append(r)
        for app, rs in sorted(by_app.items()):
            h = Hist()
            for r in rs:
                h.merge(r["hist"])
            kind = ("ransomware" if app in RANSOMWARE_APPS else
                    "idle" if app in IDLE_APPS else "benign")
            profiles[app] = _profile(
                app, kind, h, len(rs), [r["rel_path"] for r in rs], "original_runs",
                note=("run-level ground truth: one application under test per run on an "
                      "otherwise idle host. NOT a per-block process attribution - OS "
                      "background writes are present in every run, which the Idle "
                      "profile quantifies."))
        for pooled, apps, kind in (("ransomware", RANSOMWARE_APPS, "ransomware"),
                                   ("benign", BENIGN_APPS, "benign")):
            h, srcs, nr = Hist(), [], 0
            for app in apps:
                for r in by_app.get(app, []):
                    h.merge(r["hist"])
                    srcs.append(r["rel_path"])
                    nr += 1
            if h.n:
                profiles[pooled] = _profile(
                    pooled, kind, h, nr, srcs, "original_runs",
                    note=f"pooled over {', '.join(a for a in apps if a in by_app)}; "
                         f"equal weight per BLOCK, so the busier applications dominate")
        prov = {"label_source": "original_runs", "is_ground_truth": True,
                "detail": orig_prov,
                "how_labels_were_obtained": (
                    "dataset/original runs each have exactly one application under test, "
                    "so the directory name is what was writing. Labels are RUN-level, not "
                    "per-block: OS background writes occur in every run and are not "
                    "separable from the application's own, which is what the Idle runs "
                    "measure."),
                "apps_found": sorted(by_app),
                "missing_vs_mix": sorted({a for c in MIX_COMBOS
                                          for a in split_combo(c)[0]} - set(by_app))}
    else:
        # ---------------- the documented APPROXIMATION ----------------
        log("")
        log("!!! dataset/original/ IS NOT PRESENT AND NO EXTRACT WAS FOUND.")
        log("!!! Falling back to a two-component Otsu split of the MIX runs.")
        log("!!! The resulting 'benign' and 'ransomware' profiles are a FITTED GUESS")
        log("!!! about which blocks came from which process, not ground truth. Every")
        log("!!! profile below is tagged label_source=mix_otsu for that reason.")
        log("")
        lo_all, hi_all, splits = Hist(), Hist(), []
        for m in mix:
            sp = otsu_split(m["hist"])
            if sp is None:
                continue
            lo, hi = split_at(m["hist"], sp["threshold"])
            lo_all.merge(lo)
            hi_all.merge(hi)
            splits.append({"combo": m["combo"], "run": m["run"], **sp})
        note = ("APPROXIMATION, not ground truth: component of a two-group Otsu split "
                "(minimum within-group variance) of a MIX run in which a benign "
                "application and ransomware wrote concurrently with no process id "
                "recorded. Treat as a fitted guess.")
        profiles["benign"] = _profile("benign", "benign", lo_all, len(splits),
                                      [m["rel_path"] for m in mix], "mix_otsu", note)
        profiles["ransomware"] = _profile("ransomware", "ransomware", hi_all, len(splits),
                                          [m["rel_path"] for m in mix], "mix_otsu", note)
        prov = {"label_source": "mix_otsu", "is_ground_truth": False,
                "detail": orig_prov,
                "how_labels_were_obtained": (
                    "APPROXIMATION. For each mix run the entropy histogram was split in "
                    "two at the threshold minimising within-group variance (Otsu); the "
                    "low component was called benign and the high component ransomware. "
                    "No per-block process attribution exists in dataset/mix, so this is "
                    "a fitted guess and is labelled as such everywhere it appears."),
                "per_run_splits": splits}

    # ---- the mix runs themselves, reported either way
    mix_summary = {}
    for m in mix:
        benign_apps, fam = split_combo(m["combo"])
        d = mix_summary.setdefault(m["combo"], {
            "benign_apps": benign_apps, "ransomware_family": fam,
            "n_runs": 0, "n_blocks": 0, "runs": [], "hist": Hist()})
        d["n_runs"] += 1
        d["n_blocks"] += m["hist"].n
        d["runs"].append(m["rel_path"])
        d["hist"].merge(m["hist"])
    mix_out = {}
    for combo, d in sorted(mix_summary.items()):
        h = d.pop("hist")
        sp = otsu_split(h)
        d.update({"mean": h.mean, "sd": h.sd, "min": h.mn, "max": h.mx,
                  "quantiles": h.quantiles(11),
                  "otsu_two_component_fit": sp,
                  "otsu_note": ("a descriptive decomposition of a run that contains BOTH "
                                "processes; the share_high is not a measured ransomware "
                                "write share")})
        mix_out[combo] = d

    expected_mix_runs = 100
    found = len(mix_runs)
    return {
        "profiles": profiles,
        "mix": mix_out,
        "provenance": {
            **prov,
            "ransmap_root": str(root),
            "entropy_scale": ("RanSMAP column 5 is entropy NORMALISED to 0-1, not 0-8 "
                              "bits. Multiply by 8 for bits. detect.py's ent_min=7.5 "
                              "bits is 0.9375 here."),
            "hist_bins": NBINS,
            "quantile_grid": N_QUANTILES,
            "mix_runs_found": found,
            "mix_runs_expected": expected_mix_runs,
            "mix_runs_complete": found >= expected_mix_runs,
            "mix_combos_found": sorted({c for c, _ in mix_runs}),
            "mix_combos_missing": sorted(set(MIX_COMBOS) - {c for c, _ in mix_runs}),
            "mix_unparsable_rows": bad_rows,
            "chi_square_caveat": (
                "RanSMAP records entropy per block and nothing about the byte histogram, "
                "so the chi-square half of the content test CANNOT be driven by this "
                "data. synthesise_block hits a measured entropy target; the chi-square "
                "of the bytes it produces is a property of the synthesis family, not a "
                "measured property of real ransomware."),
            "built_in_seconds": round(time.time() - t0, 1),
        },
    }


def save_profiles(fit, path=PROFILES_PATH, synthesis_check=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"schema": "t1486/entropy_profiles/1",
           "provenance": fit["provenance"],
           "profiles": fit["profiles"],
           "mix_runs": fit["mix"]}
    if synthesis_check is not None:
        out["synthesis_check"] = synthesis_check
    path.write_text(json.dumps(out, indent=2))
    return path


def read_profiles(path=PROFILES_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no entropy profiles at {path}. Build them first:\n"
            f"    python3 scripts/ransmap.py --build")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# sampling and synthesis
# --------------------------------------------------------------------------- #
def sample_entropy(profile, rng, n=1):
    """Draw `n` normalised entropies from a profile by inverse-transform sampling.

    The profile's persisted quantiles ARE the inverse CDF on a grid of
    N_QUANTILES points; a uniform draw is mapped through it with linear
    interpolation between neighbouring grid points, so the result is the
    empirical distribution smoothed only at the 1/100 quantile scale.
    """
    q = profile["quantiles"] if isinstance(profile, dict) else list(profile)
    k = len(q) - 1
    out = []
    for _ in range(n):
        x = rng.random() * k
        i = int(x)
        if i >= k:
            out.append(q[k])
            continue
        f = x - i
        out.append(q[i] + f * (q[i + 1] - q[i]))
    return out


def entropy_bits(b: bytes) -> float:
    """Shannon entropy over 8-bit symbols, in bits. Same definition as common.py."""
    if not b:
        return 0.0
    c = Counter(b)
    n = len(b)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def entropy_norm(b: bytes) -> float:
    """Shannon entropy on RanSMAP's scale: bits / 8, so 0 to 1."""
    return entropy_bits(b) / BITS_PER_SYMBOL


def chi_square_uniform(b: bytes) -> float:
    """Chi-square of the byte distribution against uniform. Same as common.py."""
    if not b:
        return 0.0
    c = Counter(b)
    exp = len(b) / 256.0
    return sum((c.get(i, 0) - exp) ** 2 / exp for i in range(256))


def _counts_entropy_bits(counts, n):
    if n <= 0:
        return 0.0
    return math.log2(n) - sum(c * math.log2(c) for c in counts if c > 0) / n


def _counts_chi2(counts, n):
    exp = n / 256.0
    return sum((c - exp) ** 2 / exp for c in counts)


def _probs(lam):
    """Truncated-exponential weights over the 256 byte values.

    lam = 0 is uniform (entropy 8 bits); as lam grows the alphabet collapses
    onto the low byte values and the entropy falls smoothly to 0. One continuous
    knob covering the whole achievable range is what makes the search below a
    plain bisection.
    """
    if lam <= 0.0:
        return [1.0 / 256.0] * 256
    w = [math.exp(-lam * i) for i in range(256)]
    s = sum(w)
    return [x / s for x in w]


def _expected_bits(lam):
    return -sum(p * math.log2(p) for p in _probs(lam) if p > 0.0)


LAM_MAX = 40.0


def _lam_for_expected_bits(target_bits):
    """Bisect the skew parameter against the EXPECTED entropy of the alphabet."""
    if target_bits >= _expected_bits(0.0):
        return 0.0
    lo, hi = 0.0, LAM_MAX
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _expected_bits(mid) > target_bits:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sample_counts(lam, size, rng):
    """An i.i.d. draw of `size` bytes from the alphabet, as a 256-long count vector."""
    p = _probs(lam)
    if _np is not None:
        g = _np.random.default_rng(rng.getrandbits(63))
        return [int(x) for x in g.multinomial(size, p)]
    cw, acc = [], 0.0
    for x in p:
        acc += x
        cw.append(acc)
    cw[-1] = 1.0
    counts = [0] * 256
    for v in rng.choices(range(256), cum_weights=cw, k=size):
        counts[v] += 1
    return counts


def _exact_counts(target_bits, size):
    """Construct a byte multiset whose entropy equals the target as closely as the
    block size allows, by bisecting the same skew parameter against the EXACT
    entropy of the rounded counts.

    Needed only above the finite-sample ceiling: a 4096-byte i.i.d. uniform draw
    measures about 7.955 bits (0.9944 normalised), while real RanSMAP blocks reach
    0.9966 and above. Those blocks are more evenly balanced than chance, so they
    cannot be produced by i.i.d. sampling and the multiset has to be built.
    """
    def counts_for(lam):
        p = _probs(lam)
        raw = [x * size for x in p]
        counts = [int(r) for r in raw]
        rem = size - sum(counts)
        order = sorted(range(256), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order[:rem]:
            counts[i] += 1
        return counts

    top = counts_for(0.0)
    if target_bits >= _counts_entropy_bits(top, size):
        return top
    lo, hi = 0.0, LAM_MAX
    best = top
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        c = counts_for(mid)
        h = _counts_entropy_bits(c, size)
        if abs(h - target_bits) < abs(_counts_entropy_bits(best, size) - target_bits):
            best = c
        if h > target_bits:
            lo = mid
        else:
            hi = mid
    return best


def _bytes_from_counts(counts, rng):
    buf = bytearray()
    for v, c in enumerate(counts):
        if c:
            buf += bytes([v]) * c
    rng.shuffle(buf)
    return bytes(buf)


def _synthesise(entropy_target, size, rng, tol=0.005, max_iter=28):
    """Bytes hitting a normalised-entropy target, with a record of how it went.

    Stage 1  bisect the skew parameter against the alphabet's EXPECTED entropy,
             to get a bracket cheaply (256 operations per evaluation, no bytes).
    Stage 2  bisect against the MEASURED entropy of an actual i.i.d. draw, which
             is what the detector will see, and stop inside `tol`. Sampling is
             used rather than exact construction because a sampled block has a
             realistic chi-square (about 255 for a near-uniform draw, i.e. the
             degrees of freedom) whereas a perfectly balanced one has chi-square 0.
    Stage 3  only if stage 2 cannot reach the target - which happens above the
             finite-sample ceiling of roughly 0.9944 for 4096 bytes - construct
             the multiset exactly and flag it.

    `tol` is in normalised units, so the default 0.005 is 0.04 bits.
    """
    size = max(1, int(size))
    tgt = min(max(float(entropy_target), 0.0), 1.0)
    tgt_bits = tgt * BITS_PER_SYMBOL

    lo, hi = 0.0, LAM_MAX
    seed = _lam_for_expected_bits(tgt_bits)
    best = None
    for it in range(max_iter):
        lam = seed if it == 0 else 0.5 * (lo + hi)
        counts = _sample_counts(lam, size, rng)
        h = _counts_entropy_bits(counts, size) / BITS_PER_SYMBOL
        err = h - tgt
        if best is None or abs(err) < abs(best[1] - tgt):
            best = (counts, h, lam, it + 1)
        if abs(err) <= tol:
            break
        if err > 0:
            lo = lam
        else:
            hi = lam
    counts, h, lam, iters = best
    method = "iid_sample"
    if abs(h - tgt) > tol:
        counts = _exact_counts(tgt_bits, size)
        h = _counts_entropy_bits(counts, size) / BITS_PER_SYMBOL
        method = "exact_multiset"
    data = _bytes_from_counts(counts, rng)
    return data, {"target": tgt, "realised": h, "error": h - tgt,
                  "chi_square": _counts_chi2(counts, size), "size": size,
                  "lam": lam, "iterations": iters, "method": method,
                  "within_tolerance": abs(h - tgt) <= tol}


def synthesise_block(entropy_target, size, rng):
    """Bytes of length `size` whose Shannon entropy is close to `entropy_target`.

    `entropy_target` is on RanSMAP's NORMALISED 0-1 scale (x8 for bits), because
    that is the scale the profiles are fitted on. See `_synthesise` for the method
    and `synthesis_report` for the measured entropy/chi-square behaviour.

    Caveat that has to travel with every use: the entropy is matched to real
    measured data, the chi-square is not. RanSMAP stores no byte histograms, so
    the chi-square of these bytes is a consequence of the truncated-exponential
    synthesis family and must not be read as evidence about real ciphertext.
    """
    return _synthesise(entropy_target, size, rng)[0]


# --------------------------------------------------------------------------- #
# the payload model workload.py consumes
# --------------------------------------------------------------------------- #
# Which RanSMAP class stands in for which generated session. The office and
# archive mappings are the obvious ones; bulk_copy is pooled benign because a
# copy reproduces whatever was already on disk rather than any one application's
# output, and that mixture is exactly what the pooled profile is.
SESSION_CLASS = {
    "office": ("Office", "benign", "ransomware"),
    "bulk_copy": ("benign", "ransomware"),
    "archive": ("Zip", "benign"),
    "ransomware": ("ransomware",),
}

SAMPLE_BYTES = 4096          # what detect.py actually looks at, and RanSMAP's block size


class RanSMAPPayloads:
    """Payload bytes drawn from the measured RanSMAP entropy distributions.

    One block per write. The detector samples `content[:4096]` and RanSMAP's unit
    is the 4 KB block, so a write of any declared size is realised as a single
    block of at most SAMPLE_BYTES bytes whose entropy is drawn from the matching
    class. `bytes_written` still carries the full declared size, so the
    behavioural stage, the window counts and the timing are untouched.

    A pool of blocks is built once per class at construction, at evenly spaced
    quantile levels, because searching for a skew parameter per write is far too
    slow for hundreds of runs. At write time a level is drawn by
    inverse-transform sampling and the pooled block for it is passed through a
    random byte-permutation table. A permutation of the alphabet is a bijection
    on byte values, so it leaves the frequency multiset - and therefore the
    entropy and the chi-square - EXACTLY unchanged while making the bytes
    differ. The realised entropy distribution is thus the profile's, quantised
    to `pool` levels.
    """

    name = "ransmap"

    def __init__(self, profiles=None, pool=256, pool_seed=20240229, n_tables=64,
                 profiles_path=PROFILES_PATH):
        doc = profiles if profiles is not None else read_profiles(profiles_path)
        self.profiles = doc["profiles"]
        self.provenance = doc.get("provenance", {})
        self.profiles_path = str(profiles_path)
        self.pool_size = int(pool)
        self.pool_seed = int(pool_seed)
        self._pools = {}
        prng = random.Random(pool_seed)
        self._tables = [bytes(prng.sample(range(256), 256)) for _ in range(n_tables)]
        self.stats = {}

    def profile_for(self, kind):
        for name in SESSION_CLASS.get(kind, (kind,)):
            if name in self.profiles and self.profiles[name]["n_blocks"]:
                return name, self.profiles[name]
        raise KeyError(f"no RanSMAP profile for session kind {kind!r}; have "
                       f"{sorted(self.profiles)}")

    def _pool(self, kind):
        if kind in self._pools:
            return self._pools[kind]
        name, prof = self.profile_for(kind)
        # zlib.crc32, not hash(): str hashing is salted per process (PYTHONHASHSEED),
        # which would make the pool - and therefore every run's content - differ
        # between invocations of the same command.
        prng = random.Random(self.pool_seed ^ zlib.crc32(name.encode()))
        blocks, info = [], []
        for i in range(self.pool_size):
            # the (i+0.5)/pool quantile level: mid-bucket, so neither tail is
            # over-represented by the quantisation
            p = (i + 0.5) / self.pool_size
            q = prof["quantiles"]
            x = p * (len(q) - 1)
            j = int(x)
            tgt = q[j] if j >= len(q) - 1 else q[j] + (x - j) * (q[j + 1] - q[j])
            b, meta = _synthesise(tgt, SAMPLE_BYTES, prng)
            blocks.append(b)
            info.append(meta)
        n_exact = sum(1 for m in info if m["method"] == "exact_multiset")
        self.stats[kind] = {
            "profile": name, "pool": self.pool_size,
            "blocks_needing_exact_construction": n_exact,
            "mean_abs_entropy_error": sum(abs(m["error"]) for m in info) / len(info),
            "max_abs_entropy_error": max(abs(m["error"]) for m in info),
            "realised_entropy_mean": sum(m["realised"] for m in info) / len(info),
            "chi_square_mean": sum(m["chi_square"] for m in info) / len(info),
            "chi_square_min": min(m["chi_square"] for m in info),
            "chi_square_max": max(m["chi_square"] for m in info),
        }
        self._pools[kind] = (blocks, info)
        return self._pools[kind]

    def blob(self, kind, rng, size):
        blocks, _ = self._pool(kind)
        i = min(int(rng.random() * len(blocks)), len(blocks) - 1)
        b = blocks[i]
        if size < len(b):
            b = b[:max(1, int(size))]
        return b.translate(self._tables[rng.randrange(len(self._tables))])

    def describe(self):
        return {"payload_model": self.name, "profiles_file": self.profiles_path,
                "pool_size": self.pool_size, "pool_seed": self.pool_seed,
                "session_class_map": {k: v[0] for k, v in SESSION_CLASS.items()},
                "block_bytes": SAMPLE_BYTES,
                "label_source": self.provenance.get("label_source"),
                "is_ground_truth": self.provenance.get("is_ground_truth"),
                "pool_stats": self.stats,
                "caveat": ("entropy is drawn from measured RanSMAP blocks; chi-square is "
                           "NOT measured data - see provenance.chi_square_caveat")}


_CACHE = {}


def payloads(pool=256, pool_seed=20240229, profiles_path=PROFILES_PATH):
    """A process-wide RanSMAPPayloads, so the block pool is built once per run set."""
    key = (str(profiles_path), pool, pool_seed)
    if key not in _CACHE:
        _CACHE[key] = RanSMAPPayloads(pool=pool, pool_seed=pool_seed,
                                      profiles_path=profiles_path)
    return _CACHE[key]


# --------------------------------------------------------------------------- #
# the synthesis check that goes into entropy_profiles.json
# --------------------------------------------------------------------------- #
def synthesis_report(targets=None, size=SAMPLE_BYTES, reps=3, seed=7):
    """Measure what synthesise_block actually produces, entropy AND chi-square.

    This is the evidence for the limitation in the module docstring: entropy is
    matched to the real data, chi-square is whatever the synthesis family
    implies. The table lets a reader see the implied chi-square at every entropy
    the profiles will ask for, instead of taking the caveat on trust.
    """
    if targets is None:
        targets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                   0.9375, 0.95, 0.97, 0.99, 0.9944, 0.9966, 1.0]
    rng = random.Random(seed)
    rows = []
    for t in targets:
        ms = [_synthesise(t, size, rng)[1] for _ in range(reps)]
        rows.append({
            "target": t,
            "target_bits": round(t * BITS_PER_SYMBOL, 4),
            "realised_mean": sum(m["realised"] for m in ms) / reps,
            "realised_bits_mean": sum(m["realised"] for m in ms) / reps * BITS_PER_SYMBOL,
            "abs_error_mean": sum(abs(m["error"]) for m in ms) / reps,
            "chi_square_mean": sum(m["chi_square"] for m in ms) / reps,
            "method": ms[0]["method"],
            "within_tolerance": all(m["within_tolerance"] for m in ms),
            "reps": reps, "block_bytes": size})
    return {"note": ("entropy is matched to measured RanSMAP data; the chi-square column "
                     "is a CONSEQUENCE of the truncated-exponential synthesis family and "
                     "is not measured data. detect.py's defaults are ent_min 7.5 bits "
                     "(0.9375 here) and chi_max 350."),
            "detector_defaults": {"ent_min_bits": 7.5, "ent_min_normalised": 0.9375,
                                  "chi_max": 350.0},
            "rows": rows}


def round_trip_report(doc, kinds=None, n=4000, seed=99):
    """Does the payload model actually reproduce the measured distribution?

    Two stages are checked separately, because they fail differently:
      SAMPLED  inverse-transform draws from the persisted quantiles. This should
               match the real quantiles almost exactly; if it does not, the
               quantile grid or the sampler is wrong.
      BYTES    the entropy actually measured on the bytes the workload emits.
               This inherits a known, bounded bias at the top of the range: an
               i.i.d. draw of 4096 bytes over 256 symbols measures about 0.9953
               at best, because a random block is necessarily less balanced than
               a perfectly even one, while real RanSMAP blocks reach 0.9966 and
               above. Blocks asked for above that ceiling come back a little
               low - about 0.0015 normalised, 0.012 bits.

    That bias is accepted rather than removed on purpose. It could be removed by
    constructing the byte multiset exactly, but an exactly balanced block has a
    chi-square against uniform of 0, which is far LESS realistic than the ~255 an
    i.i.d. draw gives, and chi-square is the other half of the test. Getting the
    last 0.0015 of entropy would therefore cost the credibility of the statistic
    next to it. The ceiling is recorded here so the trade is visible.
    """
    import statistics
    profiles = doc["profiles"]
    kinds = kinds or [(k, SESSION_CLASS[k][0]) for k in
                      ("office", "bulk_copy", "archive", "ransomware")]
    model = RanSMAPPayloads(profiles=doc)
    rng = random.Random(seed)
    ceiling = _synthesise(1.0, SAMPLE_BYTES, random.Random(1))[1]["realised"]
    rows = []
    for kind, pname in kinds:
        if pname not in profiles:
            continue
        prof = profiles[pname]
        sampled = sorted(sample_entropy(prof, rng, n))
        realised = sorted(entropy_norm(model.blob(kind, rng, SAMPLE_BYTES))
                          for _ in range(n))

        def pick(seq, q):
            return seq[min(int(q * (len(seq) - 1)), len(seq) - 1)]

        rows.append({
            "session_kind": kind, "profile": pname, "n_sampled": n,
            "real": {"mean": prof["mean"],
                     **{f"p{q:02d}": prof["quantiles"][q] for q in (5, 25, 50, 75, 95)}},
            "sampled": {"mean": statistics.mean(sampled),
                        **{f"p{q:02d}": pick(sampled, q / 100)
                           for q in (5, 25, 50, 75, 95)}},
            "realised_bytes": {"mean": statistics.mean(realised),
                               **{f"p{q:02d}": pick(realised, q / 100)
                                  for q in (5, 25, 50, 75, 95)}},
            "mean_abs_shift_bytes_vs_real": abs(statistics.mean(realised) - prof["mean"]),
        })
    return {"note": ("SAMPLED checks the quantile sampler against the measured data; "
                     "BYTES checks the entropy actually realised in the emitted bytes."),
            "iid_entropy_ceiling_4096B": ceiling,
            "ceiling_note": ("the highest normalised entropy an i.i.d. 4096-byte draw "
                             "reaches. Real RanSMAP blocks exceed it (up to 0.9999), so "
                             "requests above it come back about 0.0015 low. Not removed: "
                             "see round_trip_report's docstring."),
            "block_bytes": SAMPLE_BYTES, "rows": rows}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true",
                    help="fit the profiles from RanSMAP and write data/entropy_profiles.json")
    ap.add_argument("--root", default=str(DEFAULT_RANSMAP_ROOT),
                    help="the RanSMAP checkout (the directory containing dataset/)")
    ap.add_argument("--extract", default=str(DEFAULT_EXTRACT),
                    help="full-data entropy histogram extract of dataset/original")
    ap.add_argument("--combos", nargs="*", default=None,
                    help="restrict to these mix COMBOs (default: all present)")
    ap.add_argument("--out", default=str(PROFILES_PATH))
    ap.add_argument("--no-cache", action="store_true",
                    help="re-read every mix CSV instead of using data/mix_histograms.json.gz")
    ap.add_argument("--force-mix-otsu", action="store_true",
                    help="ignore dataset/original and exercise the APPROXIMATION path")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not a.build:
        ap.error("nothing to do: pass --build")

    root = Path(a.root)
    log(f"RanSMAP root {root}")
    n_mix = len(discover_mix_runs(root, a.combos))
    log(f"mix runs with ata_write.csv: {n_mix}")
    if n_mix < 100:
        log(f"  NOTE: {n_mix} of the expected ~100 mix runs are present. The missing "
            f"ones are the runs that ship ata_write as a .zip rather than a .csv; "
            f"fetch_data.sh has a 'ransmap_zip' step for them.")
    fit = load_profiles(root, a.combos, extract=(None if a.force_mix_otsu else a.extract),
                        use_mix_cache=not a.no_cache, verbose=not a.quiet)
    if a.force_mix_otsu:
        fit["provenance"]["forced"] = "--force-mix-otsu was passed"

    log("\n--- synthesis check: what synthesise_block produces ---")
    log(f"  {'target':>7s} {'bits':>7s} {'realised':>9s} {'abs err':>8s} "
        f"{'chi2':>9s}  method")
    rep = synthesis_report()
    for r in rep["rows"]:
        log(f"  {r['target']:7.4f} {r['target_bits']:7.4f} {r['realised_mean']:9.4f} "
            f"{r['abs_error_mean']:8.5f} {r['chi_square_mean']:9.1f}  {r['method']}")
    log("  chi-square is a property of the synthesis, NOT measured RanSMAP data.")

    prov = fit["provenance"]
    log(f"\n--- label provenance: {prov['label_source']} "
        f"(ground truth: {prov.get('is_ground_truth')}) ---")
    log(f"  {prov['how_labels_were_obtained']}")

    log(f"\n--- fitted profiles (entropy normalised 0-1; x8 for bits) ---")
    log(f"  {'class':12s} {'kind':11s} {'runs':>5s} {'blocks':>11s} {'mean':>7s} "
        f"{'sd':>7s} {'p05':>7s} {'p50':>7s} {'p95':>7s}  labels")
    for name, p in sorted(fit["profiles"].items(),
                          key=lambda kv: (kv[1]["kind"], kv[0])):
        q = p["quantiles"]
        log(f"  {name:12s} {p['kind']:11s} {p['n_runs']:5d} {p['n_blocks']:11d} "
            f"{p['mean']:7.4f} {p['sd']:7.4f} {q[5]:7.4f} {q[50]:7.4f} {q[95]:7.4f}"
            f"  {p['label_source']}")

    log(f"\n--- mix runs (both processes writing; descriptive only) ---")
    log(f"  {'combo':22s} {'runs':>5s} {'blocks':>10s} {'mean':>7s} {'otsu t':>7s} "
        f"{'hi share':>9s}")
    for combo, d in fit["mix"].items():
        sp = d.get("otsu_two_component_fit") or {}
        log(f"  {combo:22s} {d['n_runs']:5d} {d['n_blocks']:10d} {d['mean']:7.4f} "
            f"{sp.get('threshold', float('nan')):7.4f} {sp.get('share_high', float('nan')):9.4f}")

    rt = round_trip_report({"profiles": fit["profiles"],
                            "provenance": fit["provenance"]})
    log(f"\n--- round trip: does the payload model reproduce the measured data? ---")
    log(f"  i.i.d. entropy ceiling for a {rt['block_bytes']}-byte block: "
        f"{rt['iid_entropy_ceiling_4096B']:.4f}")
    log(f"  {'profile':10s} {'source':8s} {'mean':>7s} {'p05':>7s} {'p50':>7s} {'p95':>7s}")
    for r in rt["rows"]:
        for src, key in (("REAL", "real"), ("SAMPLED", "sampled"),
                         ("BYTES", "realised_bytes")):
            d = r[key]
            log(f"  {(r['profile'] if src == 'REAL' else ''):10s} {src:8s} "
                f"{d['mean']:7.4f} {d['p05']:7.4f} {d['p50']:7.4f} {d['p95']:7.4f}")

    fit["provenance"]["round_trip"] = {
        "iid_entropy_ceiling_4096B": rt["iid_entropy_ceiling_4096B"],
        "max_mean_abs_shift": max((r["mean_abs_shift_bytes_vs_real"] for r in rt["rows"]),
                                  default=None)}
    p = save_profiles(fit, a.out, synthesis_check=rep)
    # the round trip belongs in the cited file too, not only on the console
    doc = json.loads(Path(p).read_text())
    doc["round_trip_check"] = rt
    Path(p).write_text(json.dumps(doc, indent=2))
    log(f"\nwrote {p}")
    log("next: python3 scripts/03_entropy_separability.py")


if __name__ == "__main__":
    main()
