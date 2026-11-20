"""Shared helpers for the T1070 log-tampering experiment. Standard library only."""
import hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
RESULTS = ROOT / "results"


def log(msg):
    print(msg, flush=True)


def canonical(rec):
    """The bytes a record's identity is computed over: everything except the chain."""
    return json.dumps({k: v for k, v in rec.items()
                       if k != "chain" and not k.startswith("_")},
                      sort_keys=True, separators=(",", ":")).encode()


def content_hash(rec):
    """Identity of a record by its CONTENT, not its sequence number.

    The sequence number is excluded on purpose: the tool Fox-IT analysed renumbers
    the records that follow a deletion, so anything keyed on seq can be rewritten.
    """
    return hashlib.sha256(
        json.dumps({k: v for k, v in rec.items()
                    if k not in ("chain", "seq") and not k.startswith("_")},
                   sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _json_safe(o):
    """Replace non-finite floats with None, recursively.

    json.dumps writes NaN, Infinity and -Infinity by default. Those are not JSON:
    every strict parser rejects the file, which would make a whole run unreadable
    because one mean was computed over an empty list. null is the honest value for
    a statistic that does not exist.
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
    # allow_nan=False so a non-finite value that slipped past _json_safe is an error
    # here rather than an unparseable results file discovered weeks later.
    p.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False))
    log(f"wrote {p}")
