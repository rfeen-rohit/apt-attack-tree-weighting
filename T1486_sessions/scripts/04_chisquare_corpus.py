#!/usr/bin/env python3
"""Does the chi-square test rescue the separation entropy alone could not make?

The T1486 design (paper 8) confirms a behavioural ransomware flag with a CONTENT
test taken from Casino et al.: entropy alone cannot separate encrypted from
compressed data, but a chi-square statistic against a uniform byte distribution
can, because ciphertext sits much closer to uniform than compressed data does.
`detect.py` implements exactly that, as `ent >= 7.5 bits AND chi <= 350.0` over a
SAMPLE_BYTES = 4096 sample of what was written.

Half of that claim is already measured. `03_entropy_separability.py` shows on
RanSMAP that entropy alone is near chance (block-level ROC-AUC 0.444; AESCrypt
0.335, Zip 0.491, Office 0.770, Firefox 0.832), which is Casino's premise
confirmed on real data. The other half cannot be measured there at all: RanSMAP
stores one entropy scalar per 4 KB block and NOT the bytes, and a 256-bin
histogram cannot be recovered from a scalar. So the remedy has never been tested
on the kit's own data.

This script tests it on real bytes.

No ransomware is needed, and none is used. Ciphertext is ciphertext: AES-256
output is AES-256 output whether Conti produced it or `openssl` did, because the
whole point of a block cipher is that its output is indistinguishable from random
to anything that does not hold the key. So the positive class is generated locally
with /usr/bin/openssl, from real files, with a key and IV derived from a recorded
seed, and it is genuinely representative of what a ransomware write looks like to
a byte-histogram test. What CANNOT be generated is the negative class, so every
negative here is a real file off the disk: JPEG, PNG, PDF, the ZIP-container
Office formats, ZIP, gzip, plus zlib-compressed text as the pure-compression case.
Plain text and source code are carried as a negative CONTROL, to show the test
behaves sensibly at the low end rather than only at the hard end.

The measurement that matters is the BLOCK-SIZE SWEEP. Chi-square over 256 bins
needs enough samples per bin: at 4096 bytes there are only 16 expected counts per
bin. The null does not care - the statistic is ~chi2 with 255 df, mean 255,
sd ~22.6, at any block size - but the POWER does, because the non-centrality grows
linearly with the block length while the noise does not. So 1 KB .. whole-file are
all measured, and the headline question is whether the kit's own SAMPLE_BYTES is
large enough.

First-block and interior-block results are reported separately. A file's first
block carries its header - JPEG's FFD8, PNG's magic, PDF's %PDF, ZIP's PK - which
is a giveaway a detector sampling mid-stream would never see. The interior blocks
are the honest test.

Entropy convention: BITS, 0..8, to match `detect.py`. RanSMAP's column is
normalised 0..1 and `03_entropy_separability.py` works on that scale; the two are
BITS = 8 * NORMALISED and must not be compared without the factor.

    python3 scripts/04_chisquare_corpus.py --corpus local
    python3 scripts/04_chisquare_corpus.py --corpus govdocs1 --govdocs-thread 0
    python3 scripts/04_chisquare_corpus.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Same two-step path dance detect.py does: this kit's scripts/common.py must keep
# shadowing the Implementation/common package, so the shared dir is APPENDED.
sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from common import RESULTS, log, save_json, wilson          # noqa: E402
from anchors import OperatingPoint                           # noqa: E402
import detect                                                # noqa: E402

KIT_ROOT = Path(__file__).resolve().parents[1]               # T1486_sessions/
IMPL_ROOT = KIT_ROOT.parent                                  # Implementation/
PROJECT_ROOT = IMPL_ROOT.parent                              # Minor Project/
SHARED = IMPL_ROOT / "_shared_data"

# ---------------------------------------------------------------- the kit's own numbers
# Read off detect.py rather than retyped, so this script cannot drift from the thing
# it is evaluating. If detect.py's defaults change, these change with them.
KIT_SAMPLE_BYTES = detect.SAMPLE_BYTES
_sig = __import__("inspect").signature(detect.content_says_encrypted).parameters
KIT_CHI_MAX = float(_sig["chi_max"].default)
KIT_ENT_MIN = float(_sig["ent_min"].default)
BITS_PER_SYMBOL = 8.0

BLOCK_SIZES = (1024, 4096, 16384, 65536, 262144)             # plus "whole"
CHUNK = 1024                                                  # histogram granularity

# ------------------------------------------------------------------------ classes
POSITIVE_TYPES = ("aes_ctr", "aes_cbc")
NEG_HIGH_TYPES = ("jpeg", "png", "gif", "pdf", "docx", "xlsx", "pptx", "zip",
                  "gzip", "zlib_text")
NEG_CTRL_TYPES = ("text", "source")

CLASS_OF_TYPE = {}
for _t in POSITIVE_TYPES:
    CLASS_OF_TYPE[_t] = "encrypted"
for _t in NEG_HIGH_TYPES:
    CLASS_OF_TYPE[_t] = "benign_high_entropy"
for _t in NEG_CTRL_TYPES:
    CLASS_OF_TYPE[_t] = "benign_control"

MAGIC = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF8", "gif"),
    (b"%PDF", "pdf"),
    (b"\x1f\x8b", "gzip"),
)
ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
OOXML_EXT = {".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx",
             ".docm": "docx", ".xlsm": "xlsx", ".pptm": "pptx"}
TEXT_EXT = {".txt", ".md", ".tex", ".csv", ".json", ".jsonl", ".html", ".xml",
            ".yml", ".yaml", ".cls", ".bst", ".bib", ".log", ".aux", ".ini"}
SOURCE_EXT = {".py", ".c", ".cc", ".cpp", ".h", ".hpp", ".js", ".css", ".scss",
              ".sh", ".php", ".twig", ".rb", ".go", ".java"}
# Files that are already in the corpus but whose bytes are the kit's own output, or
# which are not real documents at all. Excluded so the negative class is real data.
SKIP_EXT = {".pyc", ".pack", ".idx", ".rev", ".promisor", ".sample", ".service",
            ".synctex", ".out", ".bak", ".evtx"}

# The stated exclusions for --corpus local.
LOCAL_EXCLUDE_REL = ("Implementation/T1190_aggregation/data/raw",
                     "Implementation/_shared_data")

# Directories UNDER Implementation/ that hold what the kits themselves produced.
# They are skipped for two reasons. The first is provenance: a negative class that
# contains this experiment's own intermediate files is not a corpus of real files.
# The second is worse, and was a real defect - this script writes
# results/chisquare_blocks_<tag>.jsonl INTO the tree it walks, so without this the
# second run of the script measures the first run's output and no two runs agree.
GENERATED_DIR_NAMES = ("results", "work", "data")
# Nearly every real JPEG, PNG and ZIP in the tree lives UNDER the first of those -
# it is the scraped web-application corpus T1190 downloaded. --include-web-media
# re-admits only its media and document files (never its 16k .php/.css/.js source),
# because without them the hard-negative classes have 5 files between them.
WEB_MEDIA_TYPES = ("jpeg", "png", "gif", "pdf", "zip", "docx", "xlsx", "pptx")

GOVDOCS1_BASE = "https://digitalcorpora.org/corpora/file-corpora/files/govdocs1/zipfiles"
GOVDOCS1_CITE = ("Garfinkel, Farrell, Roussev and Dinolt, 'Bringing science to digital "
                 "forensics with standardized forensic corpora', DFRWS 2009")


# ====================================================================== classify
def sniff_type(path: Path, head: bytes) -> str | None:
    """The file's type, from its magic bytes where it has any, else its extension.

    Magic wins, because a .bin that is really a JPEG is a JPEG for this purpose and
    a mislabelled file in the negative class would be a silent error.
    """
    ext = path.suffix.lower()
    if ext in SKIP_EXT:
        return None
    for sig, name in MAGIC:
        if head.startswith(sig):
            return name
    if any(head.startswith(s) for s in ZIP_MAGIC):
        return OOXML_EXT.get(ext, "zip")
    if ext in OOXML_EXT:          # an OOXML file that is not a zip is not usable
        return None
    if ext in TEXT_EXT:
        return "text"
    if ext in SOURCE_EXT:
        return "source"
    return None


def iter_candidates(root: Path, excluded: tuple[str, ...], *,
                    generated_root: Path | None = None):
    """Every file under `root`, skipping excluded subtrees and generated output.

    `generated_root` is the tree under which GENERATED_DIR_NAMES are pruned; it is
    Implementation/ for a project walk and None for a corpus directory that this
    project did not produce, such as an unpacked govdocs1 thread.
    """
    ex = tuple(str(root / e) for e in excluded)
    gen = None if generated_root is None else str(generated_root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d != "__pycache__"]
        if gen is not None and (dirpath == gen or dirpath.startswith(gen + os.sep)):
            dirnames[:] = [d for d in dirnames if d not in GENERATED_DIR_NAMES]
        if any(dirpath == e or dirpath.startswith(e + os.sep) for e in ex):
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.startswith("."):
                continue
            yield Path(dirpath) / fn


def collect(root: Path, excluded, min_bytes, per_type_cap, extra_media_root=None,
            media_cap=0, generated_root=None):
    """Group real files under `root` by sniffed type, capped per type.

    The cap exists so one abundant type cannot decide the result on its own: with
    104 PDFs and 3 PNGs an uncapped pool would be a PDF experiment wearing a
    file-corpus label. Selection within a type is by sorted path, so it is
    deterministic and does not depend on directory order.
    """
    by_type = defaultdict(list)
    for p in iter_candidates(root, excluded, generated_root=generated_root):
        try:
            if p.is_symlink() or not p.is_file():
                continue
            size = p.stat().st_size
            if size < min_bytes:
                continue
            with p.open("rb") as f:
                head = f.read(16)
        except OSError:
            continue
        t = sniff_type(p, head)
        if t is None:
            continue
        by_type[t].append(p)

    if extra_media_root is not None and media_cap > 0:
        extra = defaultdict(list)
        for p in iter_candidates(extra_media_root, ()):
            try:
                if p.is_symlink() or not p.is_file() or p.stat().st_size < min_bytes:
                    continue
                with p.open("rb") as f:
                    head = f.read(16)
            except OSError:
                continue
            t = sniff_type(p, head)
            if t in WEB_MEDIA_TYPES:
                extra[t].append(p)
        for t, ps in extra.items():
            by_type[t].extend(sorted(ps)[:media_cap])

    return {t: sorted(set(ps))[:per_type_cap] for t, ps in sorted(by_type.items())}


# ====================================================================== govdocs1
def govdocs1_thread_url(thread: int, base: str = GOVDOCS1_BASE) -> str:
    if not isinstance(thread, int) or isinstance(thread, bool):
        raise ValueError(f"thread must be an int, got {thread!r}")
    if not 0 <= thread <= 999:
        raise ValueError(f"govdocs1 thread {thread} outside 0..999")
    return f"{base.rstrip('/')}/{thread:03d}.zip"


def govdocs1_prepare(thread: int, dest: Path, *, base: str = GOVDOCS1_BASE,
                     fetch: bool = False) -> Path | None:
    """Make one govdocs1 thread available locally, and say what is missing if not.

    Returns the directory of unpacked files, or None with printed instructions. One
    thread is about 1 GB of real files harvested from .gov domains, so it is never
    downloaded as a side effect of running an experiment: --govdocs-fetch has to be
    asked for, and on a machine with no outbound network the printed curl line is
    the whole of what the user has to do.
    """
    url = govdocs1_thread_url(thread, base)
    dest = Path(dest)
    zip_path = dest / f"{thread:03d}.zip"
    out_dir = dest / f"{thread:03d}"

    if out_dir.is_dir() and any(out_dir.rglob("*")):
        log(f"govdocs1 thread {thread:03d}: already unpacked at {out_dir}")
        return out_dir

    if not zip_path.exists() and fetch:
        dest.mkdir(parents=True, exist_ok=True)
        tmp = zip_path.with_suffix(".zip.part")
        log(f"govdocs1 thread {thread:03d}: downloading {url} (about 1 GB)")
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(zip_path)
            log(f"  saved {zip_path} ({zip_path.stat().st_size} bytes)")
        except (urllib.error.URLError, OSError) as exc:
            # A machine with no outbound network is the normal case here, not an
            # error to crash on: fall through to the printed instructions.
            tmp.unlink(missing_ok=True)
            log(f"  download failed: {exc}")
            log("  this machine may have no outbound network. Falling back to "
                "instructions.")

    if not zip_path.exists():
        log("")
        log(f"govdocs1 thread {thread:03d} is not here yet. It is the standardized")
        log(f"forensic corpus - {GOVDOCS1_CITE} - and is the citable")
        log("negative class for the paper. One thread is about 1 GB, so this script")
        log("will not pull it down on its own. Run ONE of:")
        log("")
        log(f"    mkdir -p '{dest}'")
        log(f"    curl -L --fail -o '{zip_path}' '{url}'")
        log("")
        log("  or, equivalently:")
        log("")
        here = Path(__file__).resolve()
        log(f"    python3 {here.parent.name}/{here.name} --corpus govdocs1 "
            f"--govdocs-thread {thread} --govdocs-fetch")
        log("")
        log("then re-run this script with --corpus govdocs1. Thread numbers are")
        log("000..999; thread 0 is the default and is enough for this experiment.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"govdocs1 thread {thread:03d}: unpacking {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        for n in names:
            # Flattened and sanitised: a corpus zip is untrusted input and must not
            # be able to write outside out_dir through a .. or absolute member name.
            safe = Path(n).name
            if not safe or safe.startswith("."):
                continue
            with z.open(n) as src, (out_dir / safe).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    log(f"  unpacked {len(names)} members")
    return out_dir


# =================================================================== ciphertext
def derive_key_iv(seed: int, tag: str, mode: str) -> tuple[str, str]:
    """A reproducible 256-bit key and 128-bit IV. Recorded in the config so that
    anyone can regenerate byte-identical ciphertext from the seed alone."""
    k = hashlib.sha256(f"{seed}|{tag}|{mode}|key".encode()).hexdigest()
    iv = hashlib.sha256(f"{seed}|{tag}|{mode}|iv".encode()).hexdigest()[:32]
    return k, iv


def make_ciphertext(sources, work: Path, seed: int, openssl: str = "/usr/bin/openssl"):
    """Encrypt each source file with AES-256-CTR and AES-256-CBC via openssl.

    Both modes are kept distinguishable: CTR is a stream over a counter and CBC is
    chained blocks, and a reviewer is entitled to ask whether the content test is
    reading a property of one mode rather than of encryption.
    """
    work.mkdir(parents=True, exist_ok=True)
    made, failed = [], []
    for i, src in enumerate(sources):
        tag = f"{i:04d}-{src.name}"
        for mode, cipher in (("aes_ctr", "aes-256-ctr"), ("aes_cbc", "aes-256-cbc")):
            key, iv = derive_key_iv(seed, tag, mode)
            out = work / f"{mode}_{i:04d}{src.suffix or ''}.enc"
            cmd = [openssl, "enc", f"-{cipher}", "-K", key, "-iv", iv,
                   "-in", str(src), "-out", str(out)]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0 or not out.exists() or out.stat().st_size < CHUNK:
                failed.append({"src": str(src), "mode": mode,
                               "rc": r.returncode,
                               "stderr": r.stderr.decode("utf8", "replace")[:200]})
                continue
            made.append((out, mode, src))
    return made, failed


def make_zlib_text(text_files, work: Path, n_blobs: int, target_in: int):
    """zlib-compressed English and source code: the pure-compression negative.

    This is the only DERIVED negative in the script and it is labelled as such. It
    is here because it is the case Casino et al.'s argument is actually about -
    deflate output at the highest level, with no container, no header and no
    embedded raw bitmap - and the corpus's own .zip files are mostly already
    compressed media repackaged, which is an easier case.
    """
    work.mkdir(parents=True, exist_ok=True)
    buf, out = bytearray(), []
    idx = 0

    def drain(force=False):
        """Emit blobs while there is enough input left to make a usable one.

        A blob shorter than one CHUNK yields no measurable block at any block size,
        so highly repetitive input is accumulated until the compressed result is at
        least a chunk long rather than written out as a file nothing can use.
        """
        nonlocal idx
        while idx < n_blobs and (force or len(buf) >= target_in):
            take = min(len(buf), target_in) if not force else len(buf)
            if take == 0:
                return
            blob = zlib.compress(bytes(buf[:take]), 9)
            while len(blob) < CHUNK and take < len(buf):
                take = min(len(buf), take * 2)
                blob = zlib.compress(bytes(buf[:take]), 9)
            if len(blob) < CHUNK:
                return                      # wait for more input, or give up at force
            del buf[:take]
            q = work / f"zlib_text_{idx:02d}.z"
            q.write_bytes(blob)
            out.append(q)
            idx += 1

    for p in text_files:
        try:
            buf += p.read_bytes()
        except OSError:
            continue
        drain()
        if idx >= n_blobs:
            break
    drain(force=True)
    return out


# ======================================================================== blocks
_LOG2 = math.log2


def ent_chi_from_counts(counts, n):
    """Shannon entropy in BITS and the chi-square against uniform, from a histogram.

    Identical in value to common.shannon_entropy / common.chi_square_uniform - the
    self-test asserts that - but computed from a 256-bin histogram so that the
    histograms of adjacent 1 KB chunks can be summed to get any larger aligned
    block without re-reading the bytes. That is what makes a six-point block-size
    sweep over the whole corpus affordable.
    """
    ent = 0.0
    exp = n / 256.0
    ss = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * _LOG2(p)
        d = c - exp
        ss += d * d
    return ent, ss / exp


def chunk_histograms(data: bytes):
    """One 256-bin histogram per aligned 1 KB chunk, plus the trailing remainder."""
    n_full = len(data) // CHUNK
    hists = []
    for i in range(n_full):
        h = [0] * 256
        for k, v in Counter(data[i * CHUNK:(i + 1) * CHUNK]).items():
            h[k] = v
        hists.append(h)
    rest = data[n_full * CHUNK:]
    tail = None
    if rest:
        tail = [0] * 256
        for k, v in Counter(rest).items():
            tail[k] = v
    return hists, tail, len(rest)


def pick_indices(n_avail: int, cap: int):
    """Up to `cap` block indices, evenly spread, index 0 always included.

    Taking the first `cap` blocks instead would make every small-block result a
    measurement of file headers, which is exactly the artefact the first-block /
    interior-block split exists to expose.
    """
    if cap <= 0 or n_avail <= 0:
        return []
    if n_avail <= cap:
        return list(range(n_avail))
    step = n_avail / cap
    return sorted({min(n_avail - 1, int(i * step)) for i in range(cap)})


def blocks_for_file(path: Path, file_type: str, cls: str, split_key: str,
                    max_bytes: int, max_blocks: int, source: str | None = None):
    """Every measured block of one file, at every block size and whole-file."""
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes + 1)
    except OSError:
        return []
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    if len(data) < CHUNK:
        return []

    hists, tail, tail_len = chunk_histograms(data)
    n_chunks = len(hists)
    rows = []
    rel = str(path)

    for bs in BLOCK_SIZES:
        k = bs // CHUNK
        n_avail = n_chunks // k
        if n_avail < 1:
            continue
        for bi in pick_indices(n_avail, max_blocks):
            acc = [0] * 256
            for c in range(bi * k, (bi + 1) * k):
                h = hists[c]
                for v in range(256):
                    acc[v] += h[v]
            ent, chi = ent_chi_from_counts(acc, bs)
            rows.append({"file": rel, "source": source, "file_type": file_type,
                         "class": cls, "split_key": split_key,
                         "block_size": bs, "block_index": bi,
                         "is_first_block": bi == 0, "block_bytes": bs,
                         "entropy_bits": round(ent, 6), "chi_square": round(chi, 3)})

    acc = [0] * 256
    for h in hists:
        for v in range(256):
            acc[v] += h[v]
    if tail is not None:
        for v in range(256):
            acc[v] += tail[v]
    total = n_chunks * CHUNK + tail_len
    ent, chi = ent_chi_from_counts(acc, total)
    rows.append({"file": rel, "source": source, "file_type": file_type, "class": cls,
                 "split_key": split_key, "block_size": "whole", "block_index": 0,
                 "is_first_block": True, "block_bytes": total,
                 "entropy_bits": round(ent, 6), "chi_square": round(chi, 3),
                 "truncated": truncated})
    return rows


# ======================================================================= metrics
def roc_auc(pos, neg):
    """ROC-AUC by the Mann-Whitney identity, with ties counted as half.

    Ties matter here: at 1 KB there are only 1024 samples over 256 bins, so entropy
    takes a limited number of distinct values and an AUC that ignored ties would be
    optimistic.
    """
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return None
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda x: x[0])
    N = n1 + n0
    rank_sum = 0.0
    i = 0
    while i < N:
        j = i
        while j + 1 < N and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0                      # 1-based ranks i+1 .. j+1
        for k in range(i, j + 1):
            if merged[k][1] == 1:
                rank_sum += avg
        i = j + 1
    u = rank_sum - n1 * (n1 + 1) / 2.0
    return u / (n1 * n0)


def roc_points(pos, neg):
    """(threshold, tp, fp) at every distinct score; the decision is score >= t.

    The list starts at the ROC origin - a threshold above every observed score, so
    tp = fp = 0. Without it a question like "TPR at FPR <= 0" has no answer at all
    on data where the classes touch, and the honest answer is "zero", not "none".
    """
    ev = sorted(set(pos) | set(neg), reverse=True)
    if ev:
        ev = [ev[0] + 1.0] + ev
    sp = sorted(pos, reverse=True)
    sn = sorted(neg, reverse=True)
    out = []
    i = j = 0
    for t in ev:
        while i < len(sp) and sp[i] >= t:
            i += 1
        while j < len(sn) and sn[j] >= t:
            j += 1
        out.append((t, i, j))
    return out


def best_at_fpr(pos, neg, cap):
    """The highest TPR reachable with FPR at or below `cap`, and where."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return None
    best = None
    for t, tp, fp in roc_points(pos, neg):
        if fp / n0 > cap:
            continue
        tpr = tp / n1
        if best is None or tpr > best["tpr"]["rate"]:
            best = {"threshold": t, "tpr": wilson(tp, n1), "fpr": wilson(fp, n0)}
    return best


def quantiles(xs, qs=(0.25, 0.5, 0.75)):
    if not xs:
        return {f"p{int(q*100)}": None for q in qs}
    s = sorted(xs)
    out = {}
    for q in qs:
        pos = q * (len(s) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(s) - 1)
        out[f"p{int(q*100)}"] = s[lo] + (s[hi] - s[lo]) * (pos - lo)
    return out


def describe(xs):
    q = quantiles(xs)
    return {"n": len(xs), "median": q["p50"], "p25": q["p25"], "p75": q["p75"],
            "iqr": None if q["p75"] is None else q["p75"] - q["p25"],
            "min": min(xs) if xs else None, "max": max(xs) if xs else None,
            "mean": (sum(xs) / len(xs)) if xs else None}


def fit_logistic(feats, labels, *, iters=3000, lr=0.5):
    """Two-feature logistic regression by gradient descent, class-balanced.

    No sklearn on this machine and none wanted: the point of the combined rule is
    only to answer whether entropy adds anything to chi-square, and a linear rule
    in two features is the smallest thing that can answer it. Class balancing is
    not cosmetic - the negative class here outnumbers the positive several times
    over, and an unbalanced fit would optimise for saying 'benign'.
    """
    n = len(labels)
    if n == 0 or not (0 < sum(labels) < n):
        return None
    d = len(feats[0])
    mu = [sum(f[k] for f in feats) / n for k in range(d)]
    sd = []
    for k in range(d):
        v = sum((f[k] - mu[k]) ** 2 for f in feats) / n
        sd.append(math.sqrt(v) if v > 1e-18 else 1.0)
    Z = [[(f[k] - mu[k]) / sd[k] for k in range(d)] for f in feats]
    n1 = sum(labels)
    n0 = n - n1
    w1, w0 = 0.5 / n1, 0.5 / n0
    wts = [w1 if y else w0 for y in labels]

    w = [0.0] * d
    b = 0.0
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for zi, y, wt in zip(Z, labels, wts):
            z = b + sum(w[k] * zi[k] for k in range(d))
            z = 30.0 if z > 30 else (-30.0 if z < -30 else z)
            e = 1.0 / (1.0 + math.exp(-z)) - y
            ewt = e * wt
            for k in range(d):
                gw[k] += ewt * zi[k]
            gb += ewt
        for k in range(d):
            w[k] -= lr * gw[k]
        b -= lr * gb
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def logistic_score(model, f):
    z = model["b"]
    for k, wk in enumerate(model["w"]):
        z += wk * (f[k] - model["mu"][k]) / model["sd"][k]
    return z


# ====================================================================== analysis
def f_ent(r):
    return r["entropy_bits"]


def f_chi(r):
    """Chi-square turned into a score where HIGHER means more encrypted.

    The log is monotone, so it changes no AUC and no ROC point; it is there because
    the raw statistic runs from 255 to tens of millions and a linear rule over the
    raw value would be dominated by the few enormous ones.
    """
    return -math.log10(r["chi_square"] + 1.0)


def split_of(r):
    """Calibration or evaluation, decided by the ORIGIN file, never by the block.

    Blocks of one file are not independent, and a ciphertext is keyed to the
    plaintext it came from, so both must land on the same side of the split or the
    held-out numbers are not held out.
    """
    h = hashlib.sha1(r["split_key"].encode("utf8", "replace")).hexdigest()
    return "cal" if int(h[:8], 16) % 2 == 0 else "eval"


def shipped_rule(rows, ent_min, chi_max):
    """TPR/FPR of `ent >= ent_min AND chi <= chi_max`, the rule detect.py runs."""
    pos = [r for r in rows if r["class"] == "encrypted"]
    neg = [r for r in rows if r["class"] != "encrypted"]
    fire = lambda r: r["entropy_bits"] >= ent_min and r["chi_square"] <= chi_max
    tp = sum(1 for r in pos if fire(r))
    fp = sum(1 for r in neg if fire(r))
    out = {"ent_min": ent_min, "chi_max": chi_max,
           "tpr": wilson(tp, len(pos)), "fpr": wilson(fp, len(neg))}
    by_type = {}
    for t in sorted({r["file_type"] for r in neg}):
        sub = [r for r in neg if r["file_type"] == t]
        by_type[t] = wilson(sum(1 for r in sub if fire(r)), len(sub))
    out["fpr_by_type"] = by_type
    return out


def calibrate_rule(cal_rows, fpr_cap=0.01):
    """Best (ent_min, chi_max) on the calibration half, at FPR <= cap.

    A 2-D grid rather than anything cleverer, because the rule being calibrated IS
    a 2-D threshold pair and the paper has to be able to print the grid.
    """
    pos = [r for r in cal_rows if r["class"] == "encrypted"]
    neg = [r for r in cal_rows if r["class"] != "encrypted"]
    if not pos or not neg:
        return None
    ent_grid = [6.0 + 0.05 * i for i in range(41)]            # 6.00 .. 8.00
    chi_grid = [200.0 + 25.0 * i for i in range(33)]          # 200 .. 1000
    chi_grid += [1200.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0]
    best = None
    for cm in chi_grid:
        pos_c = [r["entropy_bits"] for r in pos if r["chi_square"] <= cm]
        neg_c = [r["entropy_bits"] for r in neg if r["chi_square"] <= cm]
        if not pos_c:
            continue
        pos_c.sort()
        neg_c.sort()
        for em in ent_grid:
            fp = sum(1 for v in neg_c if v >= em)
            if fp / len(neg) > fpr_cap:
                continue
            tp = sum(1 for v in pos_c if v >= em)
            tpr = tp / len(pos)
            if best is None or tpr > best["tpr_cal"]:
                best = {"ent_min": round(em, 4), "chi_max": cm, "tpr_cal": tpr,
                        "fpr_cal": fp / len(neg)}
    return best


def one_block_size(rows, label, fpr_caps=(0.01, 0.05)):
    """Everything the sweep reports for one block size."""
    pos = [r for r in rows if r["class"] == "encrypted"]
    neg = [r for r in rows if r["class"] != "encrypted"]
    if not pos or not neg:
        return None
    cal = [r for r in rows if split_of(r) == "cal"]
    ev = [r for r in rows if split_of(r) == "eval"]

    model = fit_logistic([[f_ent(r), f_chi(r)] for r in cal],
                         [1 if r["class"] == "encrypted" else 0 for r in cal])

    out = {"block_size": label, "n_pos": len(pos), "n_neg": len(neg),
           "n_cal": len(cal), "n_eval": len(ev),
           "n_neg_high_entropy": sum(1 for r in neg
                                     if r["class"] == "benign_high_entropy"),
           "n_neg_control": sum(1 for r in neg if r["class"] == "benign_control"),
           "features": {}}

    def feature_block(fn, subset, note):
        p = [fn(r) for r in subset if r["class"] == "encrypted"]
        q = [fn(r) for r in subset if r["class"] != "encrypted"]
        d = {"auc": roc_auc(p, q), "n_pos": len(p), "n_neg": len(q), "basis": note}
        for cap in fpr_caps:
            d[f"at_fpr_le_{cap:g}"] = best_at_fpr(p, q, cap)
        return d

    out["features"]["entropy_bits"] = feature_block(
        f_ent, rows, "all blocks; no fitting needed")
    out["features"]["chi_square"] = feature_block(
        f_chi, rows, "all blocks; no fitting needed")
    if model is not None:
        sc = lambda r: logistic_score(model, [f_ent(r), f_chi(r)])
        out["features"]["combined"] = feature_block(
            sc, ev, "held-out half; logistic weights fitted on the other half")
        out["features"]["combined"]["model"] = {
            "form": "w0 + w1*z(entropy_bits) + w2*z(-log10(chi+1))",
            "w": [round(x, 5) for x in model["w"]], "b": round(model["b"], 5),
            "mu": [round(x, 5) for x in model["mu"]],
            "sd": [round(x, 5) for x in model["sd"]]}
        out["features"]["combined_in_sample"] = feature_block(
            sc, rows, "all blocks, fitted on half of them: IN-SAMPLE, "
                                  "reported only for comparability with the two above")

    out["distributions"] = {}
    for cls in ("encrypted", "benign_high_entropy", "benign_control"):
        sub = [r for r in rows if r["class"] == cls]
        out["distributions"][cls] = {
            "entropy_bits": describe([f_ent(r) for r in sub]),
            "chi_square": describe([r["chi_square"] for r in sub])}
    for ft in sorted({r["file_type"] for r in rows}):
        sub = [r for r in rows if r["file_type"] == ft]
        out["distributions"][f"type:{ft}"] = {
            "entropy_bits": describe([f_ent(r) for r in sub]),
            "chi_square": describe([r["chi_square"] for r in sub])}

    out["shipped_rule"] = shipped_rule(rows, KIT_ENT_MIN, KIT_CHI_MAX)
    cal_best = calibrate_rule(cal, 0.01)
    if cal_best is not None:
        ev_eval = shipped_rule(ev, cal_best["ent_min"], cal_best["chi_max"])
        cal_best["tpr_eval"] = ev_eval["tpr"]
        cal_best["fpr_eval"] = ev_eval["fpr"]
    out["calibrated_rule"] = cal_best

    if label != "whole":
        first = [r for r in rows if r["is_first_block"]]
        inner = [r for r in rows if not r["is_first_block"]]
        out["first_vs_interior"] = {}
        for nm, sub in (("first_block", first), ("interior_blocks", inner)):
            if not sub or not any(r["class"] == "encrypted" for r in sub) \
                    or not any(r["class"] != "encrypted" for r in sub):
                out["first_vs_interior"][nm] = None
                continue
            out["first_vs_interior"][nm] = {
                "n_pos": sum(1 for r in sub if r["class"] == "encrypted"),
                "n_neg": sum(1 for r in sub if r["class"] != "encrypted"),
                "auc_entropy": roc_auc([f_ent(r) for r in sub if r["class"] == "encrypted"],
                                       [f_ent(r) for r in sub if r["class"] != "encrypted"]),
                "auc_chi_square": roc_auc([f_chi(r) for r in sub if r["class"] == "encrypted"],
                                          [f_chi(r) for r in sub if r["class"] != "encrypted"]),
                "shipped_rule": shipped_rule(sub, KIT_ENT_MIN, KIT_CHI_MAX)}
    return out


# ======================================================================= console
def fmt(v, w=7, p=4, pct=False):
    if v is None:
        return " " * (w - 1) + "-"
    if pct:
        return f"{v*100:{w}.2f}"
    return f"{v:{w}.{p}f}"


def tpr_of(d, cap):
    x = d.get(f"at_fpr_le_{cap:g}")
    return None if x is None else x["tpr"]["rate"]


def print_tables(out):
    sw = out["sweep"]
    log("")
    log("=" * 96)
    log("BLOCK-SIZE SWEEP  (entropy in BITS, 0-8, as detect.py uses it)")
    log("=" * 96)
    log(f"  {'block':>9s} {'n_pos':>7s} {'n_neg':>7s} | {'AUC ent':>8s} {'AUC chi':>8s} "
        f"{'AUC both':>8s} | {'ent@1%':>7s} {'chi@1%':>7s} {'both@1%':>7s} | "
        f"{'ent@5%':>7s} {'chi@5%':>7s} {'both@5%':>7s}")
    log(f"  {'-'*9} {'-'*7} {'-'*7} + {'-'*8} {'-'*8} {'-'*8} + "
        f"{'-'*7} {'-'*7} {'-'*7} + {'-'*7} {'-'*7} {'-'*7}")
    for s in sw:
        f = s["features"]
        c = f.get("combined", {})
        log(f"  {str(s['block_size']):>9s} {s['n_pos']:7d} {s['n_neg']:7d} | "
            f"{fmt(f['entropy_bits']['auc'], 8)} {fmt(f['chi_square']['auc'], 8)} "
            f"{fmt(c.get('auc'), 8)} | "
            f"{fmt(tpr_of(f['entropy_bits'],0.01),7,pct=True)} "
            f"{fmt(tpr_of(f['chi_square'],0.01),7,pct=True)} "
            f"{fmt(tpr_of(c,0.01),7,pct=True)} | "
            f"{fmt(tpr_of(f['entropy_bits'],0.05),7,pct=True)} "
            f"{fmt(tpr_of(f['chi_square'],0.05),7,pct=True)} "
            f"{fmt(tpr_of(c,0.05),7,pct=True)}")
    log("  TPR columns are percentages at the stated FPR ceiling. AUC ent / AUC chi are")
    log("  over all blocks (neither needs fitting); AUC both and both@ are on the")
    log("  HELD-OUT half, with the two logistic weights fitted on the other half,")
    log("  split by origin file so no file appears on both sides.")

    log("")
    log("=" * 96)
    log("CLASS SEPARATION  (median [p25-p75])")
    log("=" * 96)
    log(f"  {'block':>9s} | {'entropy bits: encrypted':>26s} {'high-entropy benign':>24s} "
        f"| {'chi-square: encrypted':>24s} {'high-entropy benign':>26s}")
    for s in sw:
        d = s["distributions"]
        e, h = d["encrypted"], d["benign_high_entropy"]
        log(f"  {str(s['block_size']):>9s} | "
            f"{e['entropy_bits']['median']:9.4f} [{e['entropy_bits']['p25']:6.3f}-"
            f"{e['entropy_bits']['p75']:6.3f}] "
            f"{h['entropy_bits']['median']:9.4f} [{h['entropy_bits']['p25']:6.3f}-"
            f"{h['entropy_bits']['p75']:6.3f}] | "
            f"{e['chi_square']['median']:9.1f} [{e['chi_square']['p25']:7.1f}-"
            f"{e['chi_square']['p75']:7.1f}] "
            f"{h['chi_square']['median']:11.1f} [{h['chi_square']['p25']:9.1f}-"
            f"{h['chi_square']['p75']:9.1f}]")
    log("  For uniform bytes the chi-square statistic is ~chi2 with 255 df: mean 255,")
    log("  sd 22.6, at EVERY block size. So the encrypted column should sit near 255")
    log("  throughout, and what changes with block size is the benign column.")

    log("")
    log("=" * 96)
    log(f"THE RULE detect.py ACTUALLY RUNS:  ent >= {KIT_ENT_MIN} bits AND "
        f"chi <= {KIT_CHI_MAX}")
    log("=" * 96)
    log(f"  {'block':>9s} | {'TPR':>8s} {'FPR':>8s} | {'best ent_min':>12s} "
        f"{'best chi_max':>12s} {'TPR(eval)':>10s} {'FPR(eval)':>10s}")
    for s in sw:
        r = s["shipped_rule"]
        c = s["calibrated_rule"]
        line = (f"  {str(s['block_size']):>9s} | {r['tpr']['rate']*100:7.2f}% "
                f"{r['fpr']['rate']*100:7.2f}% | ")
        if c is None:
            line += f"{'-':>12s} {'-':>12s} {'-':>10s} {'-':>10s}"
        else:
            line += (f"{c['ent_min']:12.2f} {c['chi_max']:12.0f} "
                     f"{c['tpr_eval']['rate']*100:9.2f}% {c['fpr_eval']['rate']*100:9.2f}%")
        log(line)
    log("  Left: the shipped constants, over all blocks. Right: the (ent_min, chi_max)")
    log("  pair that maximises TPR at FPR <= 1% on the calibration half, scored on the")
    log("  held-out half.")

    log("")
    log("=" * 96)
    log("FIRST BLOCK vs INTERIOR BLOCKS  (the header is a giveaway; interior is honest)")
    log("=" * 96)
    log(f"  {'block':>9s} | {'first: AUC ent':>14s} {'AUC chi':>9s} {'FPR':>8s} "
        f"| {'interior: AUC ent':>18s} {'AUC chi':>9s} {'FPR':>8s}")
    for s in sw:
        fv = s.get("first_vs_interior")
        if not fv:
            continue
        a, b = fv.get("first_block"), fv.get("interior_blocks")
        if a is None or b is None:
            continue
        log(f"  {str(s['block_size']):>9s} | {fmt(a['auc_entropy'],14)} "
            f"{fmt(a['auc_chi_square'],9)} "
            f"{a['shipped_rule']['fpr']['rate']*100:7.2f}% | "
            f"{fmt(b['auc_entropy'],18)} {fmt(b['auc_chi_square'],9)} "
            f"{b['shipped_rule']['fpr']['rate']*100:7.2f}%")

    log("")
    log("=" * 96
        )
    log("PER-TYPE CHI-SQUARE MEDIAN  (how far from uniform each real format sits)")
    log("=" * 96)
    ref = [s for s in sw if s["block_size"] == KIT_SAMPLE_BYTES]
    big = [s for s in sw if s["block_size"] == 262144]
    types = sorted({k[5:] for s in sw for k in s["distributions"] if k.startswith("type:")})
    log(f"  {'file type':>12s} | {'at '+str(KIT_SAMPLE_BYTES)+' B':>26s} "
        f"{'FPR':>8s} | {'at 262144 B':>26s} {'FPR':>8s}")
    for t in types:
        cells = []
        for grp in (ref, big):
            if not grp:
                cells.append((None, None))
                continue
            s = grp[0]
            d = s["distributions"].get(f"type:{t}")
            fp = s["shipped_rule"]["fpr_by_type"].get(t)
            cells.append((None if d is None else d["chi_square"]["median"],
                          None if fp is None else fp["rate"]))
        a, b = cells
        la = "-" if a[0] is None else f"{a[0]:26.1f}"
        lb = "-" if b[0] is None else f"{b[0]:26.1f}"
        fa = "-" if a[1] is None else f"{a[1]*100:7.2f}%"
        fb = "-" if b[1] is None else f"{b[1]*100:7.2f}%"
        log(f"  {t:>12s} | {la} {fa:>8s} | {lb} {fb:>8s}")
    log("  FPR here is the shipped rule firing on that type - for an encrypted type")
    log("  it is really the TPR, and is printed in the same column on purpose.")


def print_verdict(out):
    sw = out["sweep"]
    log("")
    log("=" * 96)
    log("WHAT THIS MEANS FOR THE DESIGN")
    log("=" * 96)
    get = lambda bs: next((s for s in sw if s["block_size"] == bs), None)
    kit = get(KIT_SAMPLE_BYTES)
    small = get(1024)
    best = max((s for s in sw if s["features"].get("combined")),
               key=lambda s: (tpr_of(s["features"]["combined"], 0.01) or -1.0),
               default=None)
    if kit:
        ea = kit["features"]["entropy_bits"]["auc"]
        ca = kit["features"]["chi_square"]["auc"]
        log(f"  At the kit's own SAMPLE_BYTES = {KIT_SAMPLE_BYTES}:")
        log(f"    entropy alone     AUC {ea:.4f},  TPR at FPR<=1% "
            f"{fmt(tpr_of(kit['features']['entropy_bits'],0.01),0,pct=True)}%")
        log(f"    chi-square alone  AUC {ca:.4f},  TPR at FPR<=1% "
            f"{fmt(tpr_of(kit['features']['chi_square'],0.01),0,pct=True)}%")
        r = kit["shipped_rule"]
        log(f"    the shipped rule  TPR {r['tpr']['rate']*100:.2f}% at FPR "
            f"{r['fpr']['rate']*100:.2f}%  (n_pos {r['tpr']['n']}, n_neg {r['fpr']['n']})")
    big = get(262144)
    if small and kit and big:
        log(f"  Block size: chi-square AUC goes "
            f"{small['features']['chi_square']['auc']:.4f} at 1 KB -> "
            f"{kit['features']['chi_square']['auc']:.4f} at 4 KB -> "
            f"{big['features']['chi_square']['auc']:.4f} at 256 KB, while entropy "
            f"goes {small['features']['entropy_bits']['auc']:.4f} -> "
            f"{kit['features']['entropy_bits']['auc']:.4f} -> "
            f"{big['features']['entropy_bits']['auc']:.4f}.")
    if best:
        log(f"  Best combined operating point at FPR<=1% is at block size "
            f"{best['block_size']}: TPR "
            f"{fmt(tpr_of(best['features']['combined'],0.01),0,pct=True)}%.")
    log("  Convention: entropy is in BITS here (0-8). 03_entropy_separability.py")
    log("  reports RanSMAP's normalised 0-1 column; BITS = 8 x NORMALISED.")
    log("  The positive class is openssl AES-256 ciphertext of real files, not")
    log("  ransomware output, and the script's docstring says why that is the same")
    log("  thing for a byte-histogram test. The negative class is entirely real files.")


# ========================================================================== main
def build_parser():
    p = argparse.ArgumentParser(
        description="Chi-square vs entropy on a real file corpus, for the T1486 "
                    "content test.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", choices=("local", "govdocs1"), default="local",
                   help="local: walk the project tree for files already on disk. "
                        "govdocs1: the standardized forensic corpus (one ~1 GB "
                        "thread, downloaded only when asked).")
    p.add_argument("--corpus-root", default=None,
                   help="override the directory --corpus local walks "
                        f"(default {PROJECT_ROOT})")
    p.add_argument("--govdocs-thread", type=int, default=0,
                   help="which govdocs1 thread, 0..999 (default 0)")
    p.add_argument("--govdocs-url", default=GOVDOCS1_BASE,
                   help="base URL for govdocs1 zipfiles")
    p.add_argument("--govdocs-fetch", action="store_true",
                   help="actually download the thread (about 1 GB). Without this "
                        "the script prints the curl line and stops.")
    p.add_argument("--include-web-media", action="store_true",
                   help="re-admit the media and document files under "
                        "Implementation/T1190_aggregation/data/raw, which --corpus "
                        "local otherwise excludes. Without it the tree has 5 real "
                        "images in total and the hard-negative classes are empty.")
    p.add_argument("--n-encrypt", type=int, default=50,
                   help="how many real files to encrypt (x2 modes) (default 50)")
    p.add_argument("--per-type-cap", type=int, default=60,
                   help="max files per file type, so one abundant type cannot "
                        "decide the result (default 60)")
    p.add_argument("--media-cap", type=int, default=60,
                   help="max files per type re-admitted by --include-web-media")
    p.add_argument("--max-bytes-per-file", type=int, default=8 * 1024 * 1024)
    p.add_argument("--max-blocks-per-file", type=int, default=48,
                   help="per file per block size, spread evenly (default 48)")
    p.add_argument("--min-bytes", type=int, default=CHUNK,
                   help="skip files smaller than this (default 1024)")
    p.add_argument("--zlib-blobs", type=int, default=8)
    p.add_argument("--zlib-input-bytes", type=int, default=1024 * 1024)
    p.add_argument("--seed", type=int, default=20261004,
                   help="seed the AES key and IV of every ciphertext derive from")
    p.add_argument("--openssl", default="/usr/bin/openssl")
    p.add_argument("--tag", default=None, help="results filename suffix")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--keep-work", action="store_true",
                   help="keep the generated ciphertext instead of deleting it")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        return selftest()

    t_start = time.time()
    tag = args.tag or (f"govdocs1_{args.govdocs_thread:03d}"
                       if args.corpus == "govdocs1" else
                       ("local_media" if args.include_web_media else "local"))

    if args.corpus == "govdocs1":
        try:
            govdocs1_thread_url(args.govdocs_thread, args.govdocs_url)
        except ValueError as exc:
            log(f"error: {exc}")
            return 2
        root = govdocs1_prepare(args.govdocs_thread, SHARED / "govdocs1",
                                base=args.govdocs_url, fetch=args.govdocs_fetch)
        if root is None:
            return 2
        excluded = ()
        media_root, media_cap = None, 0
    else:
        root = Path(args.corpus_root) if args.corpus_root else PROJECT_ROOT
        excluded = LOCAL_EXCLUDE_REL
        media_root = (root / LOCAL_EXCLUDE_REL[0]) if args.include_web_media else None
        media_cap = args.media_cap if args.include_web_media else 0

    log(f"T1486 content test on a real file corpus - corpus={args.corpus} tag={tag}")
    log(f"  root: {root}")
    if excluded:
        log(f"  excluded: {', '.join(excluded)}")
    if media_root is not None:
        log(f"  re-admitted (media/documents only, <= {media_cap} per type): {media_root}")

    by_type = collect(root, excluded, args.min_bytes, args.per_type_cap,
                      extra_media_root=media_root, media_cap=media_cap,
                      generated_root=IMPL_ROOT if args.corpus == "local" else None)
    real_counts = {t: len(ps) for t, ps in by_type.items()}
    log(f"  real files found by type: {real_counts}")
    if not by_type:
        log("  nothing usable found. Nothing to measure.")
        return 2

    work = Path(args.work_dir) if args.work_dir else (KIT_ROOT / "work" / f"chisq_{tag}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------- the derived negative
    text_pool = [p for t in ("text", "source") for p in by_type.get(t, [])]
    zblobs = make_zlib_text(sorted(text_pool, key=lambda p: -p.stat().st_size),
                            work, args.zlib_blobs, args.zlib_input_bytes)
    log(f"  zlib-compressed text blobs: {len(zblobs)}")

    # ------------------------------------------------------- the positive class
    # Round-robin across types, largest file first within each type, so the
    # plaintexts behind the ciphertext are a cross-section of the corpus and not
    # 50 PDFs. The ciphertext does not depend on the plaintext's statistics, but a
    # reviewer should not have to take that on trust from the selection.
    ordered = {t: sorted(ps, key=lambda p: -p.stat().st_size)
               for t, ps in by_type.items()}
    sources, i = [], 0
    while len(sources) < args.n_encrypt:
        added = False
        for t in sorted(ordered):
            if i < len(ordered[t]) and len(sources) < args.n_encrypt:
                sources.append(ordered[t][i])
                added = True
        if not added:
            break
        i += 1
    made, failed = make_ciphertext(sources, work, args.seed, args.openssl)
    log(f"  ciphertext files: {len(made)} from {len(sources)} plaintexts "
        f"({len(failed)} failures)")
    if failed:
        for f in failed[:5]:
            log(f"    FAILED {f['mode']} {f['src']}: rc={f['rc']} {f['stderr']}")
    if not made:
        log("  no ciphertext could be generated; the positive class is empty.")
        return 2

    # ------------------------------------------------------------------ blocks
    log("  reading blocks ...")
    rows = []
    for t, ps in by_type.items():
        for p in ps:
            rows += blocks_for_file(p, t, CLASS_OF_TYPE[t], str(p),
                                    args.max_bytes_per_file, args.max_blocks_per_file)
    for q in zblobs:
        rows += blocks_for_file(q, "zlib_text", CLASS_OF_TYPE["zlib_text"], str(q),
                                args.max_bytes_per_file, args.max_blocks_per_file)
    for out_path, mode, src in made:
        rows += blocks_for_file(out_path, mode, CLASS_OF_TYPE[mode], str(src),
                                args.max_bytes_per_file, args.max_blocks_per_file,
                                source=str(src))
    log(f"  {len(rows)} blocks over {len({r['file'] for r in rows})} files "
        f"in {time.time()-t_start:.1f}s")

    # ----------------------------------------------------------------- analysis
    sweep = []
    for bs in list(BLOCK_SIZES) + ["whole"]:
        sub = [r for r in rows if r["block_size"] == bs]
        s = one_block_size(sub, bs)
        if s is not None:
            sweep.append(s)

    file_counts = defaultdict(int)
    block_counts = defaultdict(int)
    for r in rows:
        if r["block_size"] == "whole":
            file_counts[r["file_type"]] += 1
        block_counts[r["file_type"]] += 1

    osv = subprocess.run([args.openssl, "version"], capture_output=True)
    out = {
        "experiment": "T1486 content test: chi-square vs entropy on a real file corpus",
        "config": {
            "corpus": args.corpus, "tag": tag, "root": str(root),
            "excluded": list(excluded),
            "generated_dirs_pruned_under_Implementation": list(GENERATED_DIR_NAMES),
            "include_web_media": bool(args.include_web_media),
            "govdocs1": {"thread": args.govdocs_thread,
                         "url": govdocs1_thread_url(args.govdocs_thread, args.govdocs_url),
                         "citation": GOVDOCS1_CITE},
            "block_sizes": list(BLOCK_SIZES) + ["whole"],
            "histogram_chunk_bytes": CHUNK,
            "max_bytes_per_file": args.max_bytes_per_file,
            "max_blocks_per_file": args.max_blocks_per_file,
            "per_type_cap": args.per_type_cap, "media_cap": media_cap,
            "min_bytes": args.min_bytes,
            "n_encrypt_requested": args.n_encrypt,
            "zlib_blobs": args.zlib_blobs,
            "zlib_input_bytes": args.zlib_input_bytes,
            "seed": args.seed,
            "key_iv_derivation": "key = sha256('<seed>|<index>-<basename>|<mode>|key'), "
                                 "iv = sha256('<seed>|<index>-<basename>|<mode>|iv')[:32]; "
                                 "openssl enc -aes-256-{ctr,cbc} -K <key> -iv <iv>, so "
                                 "the ciphertext is byte-identical on a re-run",
            "openssl": osv.stdout.decode("utf8", "replace").strip(),
            "entropy_convention": "BITS, 0-8, as detect.py uses. RanSMAP's column and "
                                  "03_entropy_separability.py are NORMALISED 0-1; "
                                  "BITS = 8 x NORMALISED.",
            "kit_constants": {"SAMPLE_BYTES": KIT_SAMPLE_BYTES,
                              "ent_min": KIT_ENT_MIN, "chi_max": KIT_CHI_MAX,
                              "read_from": "detect.py, not retyped"},
        },
        "provenance": {
            "positive_class": "AES-256-CTR and AES-256-CBC ciphertext of real files, "
                              "generated locally with openssl. No ransomware is involved. "
                              "A block cipher's output is indistinguishable from random "
                              "without the key, so for a byte-histogram test openssl "
                              "ciphertext and ransomware ciphertext are the same object.",
            "negative_class": "entirely real files off the disk, except zlib_text, which "
                              "is zlib level-9 compression of the corpus's own text and "
                              "source files and is labelled as derived.",
            "negative_control": "plain text and source code, carried to show the test "
                                "behaves at the low end as well as the hard end.",
            "known_limits": [
                "The plaintexts behind the ciphertext are drawn from the same corpus as "
                "the negatives, so file SIZE distributions are correlated across classes. "
                "Nothing in either feature depends on size at fixed block size.",
                "zlib_text is the only generated negative.",
                "A govdocs1 run is the citable version; a local run is the same code on "
                "whatever happens to be on this machine.",
            ],
        },
        "corpus": {"files_by_type": dict(sorted(file_counts.items())),
                   "blocks_by_type": dict(sorted(block_counts.items())),
                   "real_files_discovered_by_type": real_counts,
                   "classes": {c: sorted(t for t in CLASS_OF_TYPE if CLASS_OF_TYPE[t] == c)
                               for c in ("encrypted", "benign_high_entropy",
                                         "benign_control")},
                   "ciphertext_failures": failed},
        "sweep": sweep,
        "runtime_seconds": round(time.time() - t_start, 2),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    jl = RESULTS / f"chisquare_blocks_{tag}.jsonl"
    with jl.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in
                                ("file", "file_type", "class", "block_size",
                                 "block_index", "is_first_block", "block_bytes",
                                 "entropy_bits", "chi_square")},
                               separators=(",", ":")) + "\n")
    log(f"wrote {jl}  ({len(rows)} rows)")
    save_json(RESULTS / f"chisquare_corpus_{tag}.json", out)

    best = max((s for s in sweep if s["features"].get("combined")),
               key=lambda s: (tpr_of(s["features"]["combined"], 0.01) or -1.0),
               default=None)
    if best is not None:
        c = best["features"]["combined"]["at_fpr_le_0.01"]
        if c is not None:
            op = OperatingPoint(
                tpr=c["tpr"]["rate"], fpr=c["fpr"]["rate"],
                n_pos=c["tpr"]["n"], n_neg=c["fpr"]["n"],
                threshold=round(c["threshold"], 6),
                detector=(f"T1486 content test, combined entropy+chi-square logistic "
                          f"rule at block size {best['block_size']} bytes, held-out "
                          f"half, corpus={tag}"))
            op.save(RESULTS / f"operating_point_chisq_{tag}.json")
            log(f"wrote {RESULTS / f'operating_point_chisq_{tag}.json'}")

    print_tables(out)
    print_verdict(out)

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
        log(f"  removed generated ciphertext at {work} (--keep-work to retain)")
    return 0


# ===================================================================== self-test
def selftest():
    import random
    from common import chi_square_uniform, shannon_entropy

    rng = random.Random(7)
    blob = bytes(rng.randrange(256) for _ in range(8192))
    h = [0] * 256
    for k, v in Counter(blob).items():
        h[k] = v
    e, c = ent_chi_from_counts(h, len(blob))
    assert abs(e - shannon_entropy(blob)) < 1e-9, (e, shannon_entropy(blob))
    assert abs(c - chi_square_uniform(blob)) < 1e-6, (c, chi_square_uniform(blob))

    # summing adjacent 1 KB histograms must equal the histogram of the join
    hists, tail, tl = chunk_histograms(blob)
    assert len(hists) == 8 and tail is None and tl == 0
    acc = [sum(hists[i][v] for i in range(4)) for v in range(256)]
    e2, c2 = ent_chi_from_counts(acc, 4096)
    assert abs(e2 - shannon_entropy(blob[:4096])) < 1e-9
    assert abs(c2 - chi_square_uniform(blob[:4096])) < 1e-6

    # uniform bytes: chi-square has mean 255 at any block size
    for bs in (1024, 4096, 65536):
        vals = []
        for _ in range(40):
            b = os.urandom(bs)
            hh = [0] * 256
            for k, v in Counter(b).items():
                hh[k] = v
            vals.append(ent_chi_from_counts(hh, bs)[1])
        m = sum(vals) / len(vals)
        assert 180 < m < 340, f"uniform chi-square mean {m} at block {bs}"

    assert roc_auc([1, 2, 3], [-1, 0]) == 1.0
    assert roc_auc([-1, 0], [1, 2, 3]) == 0.0
    assert roc_auc([1, 1], [1, 1]) == 0.5, "ties must score 0.5, not 1.0"
    # pos 2 beats neg 1; pos 2 loses to neg 3; pos 3 beats neg 1; pos 3 ties neg 3
    assert abs(roc_auc([2, 3], [1, 3]) - 0.625) < 1e-12
    assert roc_auc([], [1]) is None

    b = best_at_fpr([5, 6, 7], [1, 2, 3, 4], 0.0)
    assert b["tpr"]["rate"] == 1.0 and b["fpr"]["rate"] == 0.0
    z = best_at_fpr([1, 2], [1, 2], 0.0)
    assert z["tpr"]["rate"] == 0.0 and z["fpr"]["rate"] == 0.0, z
    assert best_at_fpr([1, 2], [1, 2], 0.5)["tpr"]["rate"] == 0.5

    assert pick_indices(3, 10) == [0, 1, 2]
    pi = pick_indices(100, 5)
    assert pi[0] == 0 and len(pi) == 5 and max(pi) < 100
    assert pick_indices(0, 5) == [] and pick_indices(10, 0) == []

    q = quantiles([1, 2, 3, 4, 5])
    assert q["p50"] == 3 and q["p25"] == 2 and q["p75"] == 4

    m = fit_logistic([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [4.9, 5.1]],
                     [0, 0, 1, 1], iters=800)
    assert m is not None
    s0 = logistic_score(m, [0.0, 0.0])
    s1 = logistic_score(m, [5.0, 5.0])
    assert s1 > s0, "the fit must order a separable problem correctly"
    assert fit_logistic([[1.0, 1.0]], [1]) is None, "one class is not fittable"

    assert sniff_type(Path("x.bin"), b"\xff\xd8\xff\xe0junk") == "jpeg"
    assert sniff_type(Path("x.docx"), b"PK\x03\x04junk") == "docx"
    assert sniff_type(Path("x.zip"), b"PK\x03\x04junk") == "zip"
    assert sniff_type(Path("x.pdf"), b"%PDF-1.7") == "pdf"
    assert sniff_type(Path("x.docx"), b"not a zip at all") is None
    assert sniff_type(Path("x.py"), b"import os\n") == "source"
    assert sniff_type(Path("x.md"), b"# title\n") == "text"
    assert sniff_type(Path("x.pyc"), b"\x6f\x0d\x0d\x0a") is None
    assert set(CLASS_OF_TYPE) == set(POSITIVE_TYPES + NEG_HIGH_TYPES + NEG_CTRL_TYPES)

    assert govdocs1_thread_url(0).endswith("/000.zip")
    assert govdocs1_thread_url(7).endswith("/007.zip")
    assert govdocs1_thread_url(999).endswith("/999.zip")
    assert govdocs1_thread_url(3, "http://x/y/") == "http://x/y/003.zip"
    for bad in (-1, 1000, 1.5, True, "0"):
        try:
            govdocs1_thread_url(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"govdocs1 thread {bad!r} should have been refused")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert govdocs1_prepare(0, d / "gd", fetch=False) is None, \
            "a missing thread must print instructions and return None, never download"
        assert not (d / "gd" / "000.zip").exists()
        zp = d / "gd2" / "005.zip"
        zp.parent.mkdir(parents=True)
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("005/doc1.txt", "hello " * 500)
            z.writestr("005/img.jpg", b"\xff\xd8\xff\xe0" + os.urandom(4096))
            z.writestr("../escape.txt", "no")
        got = govdocs1_prepare(5, d / "gd2", fetch=False)
        assert got == d / "gd2" / "005" and got.is_dir()
        names = sorted(p.name for p in got.iterdir())
        assert names == ["doc1.txt", "escape.txt", "img.jpg"], names
        assert not (d / "gd2" / "escape.txt").exists(), "zip traversal was not contained"
        assert govdocs1_prepare(5, d / "gd2", fetch=False) == got, "re-run must be a no-op"

        k1, i1 = derive_key_iv(1, "a", "aes_ctr")
        k2, i2 = derive_key_iv(1, "a", "aes_ctr")
        k3, _ = derive_key_iv(2, "a", "aes_ctr")
        k4, _ = derive_key_iv(1, "a", "aes_cbc")
        assert (k1, i1) == (k2, i2) and len(k1) == 64 and len(i1) == 32
        assert k1 != k3 and k1 != k4

        src = d / "plain.txt"
        src.write_bytes(b"the quick brown fox " * 2000)
        made, failed = make_ciphertext([src], d / "w", 1234)
        assert not failed and len(made) == 2, (made, failed)
        modes = sorted(m for _, m, _ in made)
        assert modes == ["aes_cbc", "aes_ctr"]
        for out_p, mode, _ in made:
            rr = blocks_for_file(out_p, mode, "encrypted", str(src), 1 << 20, 8)
            whole = [r for r in rr if r["block_size"] == "whole"][0]
            assert whole["entropy_bits"] > 7.9, whole
            assert whole["chi_square"] < 400, whole
        made2, _ = make_ciphertext([src], d / "w2", 1234)
        assert (d / "w" / made[0][0].name).read_bytes() == \
               (d / "w2" / made2[0][0].name).read_bytes(), "ciphertext must be reproducible"

        # Varied text, not a repeated phrase: a repeated phrase compresses to a few
        # hundred bytes and would exercise only the too-small-blob path.
        words = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randrange(3, 11)))
                 for _ in range(600)]
        varied = (" ".join(rng.choice(words) for _ in range(60000))).encode()
        src2 = d / "varied.txt"
        src2.write_bytes(varied)
        zb = make_zlib_text([src2], d / "z", 2, 128 * 1024)
        assert zb, "zlib negative must be produced"
        rr = blocks_for_file(zb[0], "zlib_text", "benign_high_entropy", str(zb[0]),
                             1 << 20, 8)
        assert rr, "a zlib blob must yield measurable blocks"
        w = [r for r in rr if r["block_size"] == "whole"][0]
        assert 7.0 < w["entropy_bits"] <= 8.0, w
        assert w["chi_square"] > 0.0, w
        # Deliberately NOT asserted: that compressed text sits further from uniform
        # than ciphertext does. On this input it does not - raw deflate output of
        # low-redundancy text lands within a few tens of a chi-square of uniform.
        # Whether Casino et al.'s separation survives on real compressed formats is
        # the question this script measures, so it must not be a test assumption.
        # a degenerate, hugely compressible input must yield nothing, not a file too
        # small to measure
        rep = d / "rep.txt"
        rep.write_bytes(b"the quick brown fox " * 2000)
        assert make_zlib_text([rep], d / "z2", 2, 16 * 1024) == []

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        impl = d / "Implementation"
        (impl / "K_sessions" / "results").mkdir(parents=True)
        (impl / "K_sessions" / "scripts").mkdir(parents=True)
        (impl / "K_sessions" / "data").mkdir(parents=True)
        (d / "Paper" / "results").mkdir(parents=True)
        (impl / "K_sessions" / "results" / "out.jsonl").write_text("x" * 4096)
        (impl / "K_sessions" / "data" / "derived.json").write_text("x" * 4096)
        (impl / "K_sessions" / "scripts" / "keep.py").write_text("y" * 4096)
        (d / "Paper" / "results" / "paper.txt").write_text("z" * 4096)
        seen = {p.name for p in iter_candidates(d, (), generated_root=impl)}
        assert "keep.py" in seen and "paper.txt" in seen, seen
        assert "out.jsonl" not in seen, "the script's own output must not be corpus"
        assert "derived.json" not in seen, "kit-derived data must not be corpus"
        assert {p.name for p in iter_candidates(d, (), generated_root=None)} >= \
            {"out.jsonl", "derived.json"}, "pruning must be opt-in by tree"

    log("04_chisquare_corpus.py: all self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
