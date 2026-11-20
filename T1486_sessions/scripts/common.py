"""Shared helpers for the T1486 ransomware-detection experiment."""
import json, math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
RESULTS = ROOT / "results"


def log(msg):
    print(msg, flush=True)


def shannon_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    c = Counter(b)
    n = len(b)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def chi_square_uniform(b: bytes) -> float:
    """Chi-square of the byte distribution against uniform.

    Encrypted data is close to uniform; compressed data keeps small but measurable
    structure. This is the statistic Casino et al. use to tell the two apart, and
    the one ERW-Radar uses to confirm that a write really is encryption.
    """
    if not b:
        return 0.0
    c = Counter(b)
    exp = len(b) / 256.0
    return sum((c.get(i, 0) - exp) ** 2 / exp for i in range(256))


Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """Wilson score interval for a binomial proportion, 95% by default.

    Written out here because scipy and statsmodels are not installed, and wanted
    because the normal approximation is useless at exactly the values that matter:
    it reports zero width at 0% and at 100%, which is where a false-alarm rate sits.
    Returned with k and n, so no rate is ever reported without its denominator.
    """
    k, n = int(k), int(n)
    if n <= 0:
        return {"rate": None, "lo": None, "hi": None, "k": k, "n": n}
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"rate": round(p, 6), "lo": round(max(0.0, centre - half), 6),
            "hi": round(min(1.0, centre + half), 6), "k": k, "n": n}


def _json_safe(o):
    """Replace non-finite floats with None.

    A NaN or an infinity reaches json.dumps as the bare tokens NaN and Infinity,
    which are not JSON and which a strict parser refuses, so a single empty mean
    could make a whole results file unreadable. Non-finite values are written as
    null instead, which every parser accepts and which reads as "no value".
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False))
    log(f"wrote {p}")
