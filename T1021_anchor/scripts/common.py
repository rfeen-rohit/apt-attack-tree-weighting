"""Shared helpers for the T1021 lateral-movement experiment."""
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def log(msg):
    print(msg, flush=True)


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


def read_jsonl(path):
    p = Path(path)
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {p} ({len(rows)} rows)")


class JsonlWriter:
    """Append rows to a JSONL file as they are produced, counting them.

    write_jsonl() takes a list, which means the caller has to have built the list
    first. 02_run_experiment.py used to accumulate one dict per logon per collected
    setting and write them all at the end - three to five full copies of the log held
    in RAM for the sole purpose of being serialised later. This writes each row as it
    is produced and holds nothing, so the per-logon file costs disk rather than memory,
    and it survives an interruption as the prefix that was actually computed.

    The bytes written are identical to write_jsonl's for the same rows in the same
    order: json.dumps(row) + "\n", nothing else.
    """

    def __init__(self, path, buffering=1 << 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", buffering=buffering)
        self.rows = 0

    def write(self, row):
        self._f.write(json.dumps(row) + "\n")
        self.rows += 1

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
            log(f"wrote {self.path} ({self.rows} rows)")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
