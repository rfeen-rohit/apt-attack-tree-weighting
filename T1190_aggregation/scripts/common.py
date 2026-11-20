"""Shared helpers for the T1190 aggregation experiment."""
import hashlib, json, math, os, re, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

PHP_EXT = {".php", ".php3", ".php4", ".php5", ".php7", ".phtml", ".inc"}
OTHER_EXT = {".jsp", ".jspx", ".asp", ".aspx", ".ashx", ".asmx"}
CODE_EXT = PHP_EXT | OTHER_EXT


def log(msg):
    print(msg, flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def resolve(path) -> Path:
    """Turn a stored path into one that exists HERE.

    index.jsonl used to store absolute paths. The same corpus folder can be mounted
    at two different absolute paths (for example on the host and inside a VM), so an
    index built under one is unreadable under the other. Paths are now stored
    RELATIVE to the kit root; this function
    accepts either, and repairs a stale absolute path by re-anchoring everything from
    the "data/" component onwards onto this kit's own root.
    """
    p = Path(path)
    if not p.is_absolute():
        return ROOT / p
    if p.exists():
        return p
    parts = p.parts
    if "data" in parts:                      # re-anchor a foreign absolute path
        i = len(parts) - 1 - parts[::-1].index("data")
        cand = ROOT.joinpath(*parts[i:])
        if cand.exists():
            return cand
    return p                                  # let the caller report it missing


def relpath(path) -> str:
    """Store paths relative to the kit root, so the index travels with the folder."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_text(path) -> str:
    b = resolve(path).read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def lang_of(path) -> str:
    e = Path(path).suffix.lower()
    if e in PHP_EXT:
        return "php"
    if e in (".jsp", ".jspx"):
        return "jsp"
    if e in (".asp",):
        return "asp"
    if e in (".aspx", ".ashx", ".asmx"):
        return "aspx"
    return "other"


_WS = re.compile(r"\s+")


def normalise_for_dedup(text: str) -> str:
    """Normalised form used for near-duplicate detection (Hannousse et al.)."""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"/\*.*?\*/", " ", t, flags=re.S)
    t = re.sub(r"(?m)(//|#).*$", " ", t)
    t = _WS.sub(" ", t)
    return t.strip().lower()


def token_set(norm_text: str, n=5):
    """Shingle set for Jaccard near-duplicate detection."""
    toks = norm_text.split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _jsonl_lines(f, rows):
    """One compact JSON object per line, every non-finite float turned into null."""
    for r in rows:
        f.write(json.dumps(json_finite(r), allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        _jsonl_lines(f, rows)
    log(f"wrote {path}  ({len(rows)} rows)")


def truncate_jsonl(path):
    """Start an append-built jsonl file empty, so a re-run never mixes two runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        _jsonl_lines(f, rows)
    log(f"appended {len(rows)} rows to {path}")


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def json_finite(obj):
    """Replace every non-finite float (NaN, +-inf) with None, recursively.

    Bare NaN is accepted by Python's json module but is not valid JSON, and jq,
    pandas and R all refuse to read it, so the results files must carry null.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_finite(v) for v in obj]
    return obj


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_finite(obj), indent=2, allow_nan=False))
    log(f"wrote {path}")


# ---------------------------------------------------------------- near-duplicates
import hashlib as _hl

MINHASH_K = 64


def bottom_k_sketch(shingles, k=MINHASH_K):
    """Bottom-k sketch: the k smallest 64-bit hashes of the shingle set.

    Compact (k integers per file) and gives an unbiased Jaccard estimate, so
    de-duplication stays linear in memory and cheap to compare.
    """
    hs = sorted({int.from_bytes(_hl.blake2b(s.encode("utf-8", "replace"),
                                            digest_size=8).digest(), "big")
                 for s in shingles})
    return tuple(hs[:k])


def sketch_jaccard(a, b, k=MINHASH_K):
    if not a or not b:
        return 0.0
    merged = sorted(set(a) | set(b))[:k]
    if not merged:
        return 0.0
    sa, sb = set(a), set(b)
    return sum(1 for h in merged if h in sa and h in sb) / len(merged)
