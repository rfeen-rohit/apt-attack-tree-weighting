"""Shared helpers for the T1593 decoy experiment. Standard library only."""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
PLAN = ROOT / "decoys.json"
EVENT_LOG = LOGS / "events.jsonl"


# Account names that every credential-stuffing list already contains. An attempt on one
# of these is guessing, not evidence that our planted content was read, so it grades as a
# probe. svc_backup is in this list because it used to be the decoy account itself:
# grading it as a use made the headline number a count of ordinary stuffing bots.
#
# The list lives here, rather than in the grader, because both ends need it. The server
# has to decide whether a submitted name was generic AT THE MOMENT OF RECORDING: the name
# itself is third-party data and is never written to the event log, so the grader can only
# read the verdict.
GENERIC_ACCOUNT_NAMES = frozenset({
    "svc_backup", "svc-backup", "backup", "backupadmin", "svc_admin", "svc_service",
    "administrator", "admin", "administrateur", "root", "guest", "test", "user",
    "operator", "oracle", "postgres", "mysql", "sa", "ftp", "ftpuser", "www",
    "www-data", "nginx", "apache", "tomcat", "jenkins", "git", "deploy", "ubuntu",
    "pi", "support", "service", "sysadmin", "manager", "monitor", "nagios", "zabbix",
})

# Every string written to the event log is capped at this many characters, and the number
# of query keys kept is capped as well. See Handler.record().
MAX_FIELD = 256
MAX_QUERY_KEYS = 20
MAX_QUERY_VALUES = 3


def length_bucket(n):
    """A coarse length band, so a submitted value can be described without keeping it."""
    for hi, name in ((0, "0"), (8, "1-8"), (16, "9-16"), (32, "17-32"),
                     (64, "33-64"), (128, "65-128")):
        if n <= hi:
            return name
    return "129+"


def looks_like_email(s):
    """Shape only: has the value the form of an address? The value itself is not kept."""
    s = s or ""
    return "@" in s and "." in s.rsplit("@", 1)[-1]


def log(msg):
    print(msg, flush=True)


def load_plan():
    if not PLAN.exists():
        raise SystemExit(f"{PLAN} not found - run scripts/00_make_decoys.py first")
    return json.loads(PLAN.read_text())


def save_plan(plan):
    PLAN.write_text(json.dumps(plan, indent=2))
    log(f"wrote {PLAN}")


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def append_jsonl(path, row):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
