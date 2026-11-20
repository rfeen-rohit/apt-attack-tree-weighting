"""Read REAL Windows Security records out of the EVTX-to-MITRE-Attack samples.

Why this module exists. `loggen.py` can invent a log - six event ids, seven
usernames, five hostnames, a timestamp every 1/rate seconds - and that is a fine
harness, but it is not a Windows log, and the kit's most interesting failure mode
has to be *asked for* in a synthetic log (`--collision-rate`) rather than observed.
Records that are byte-identical apart from their record number are something a real
Security log produces on its own. This module supplies them from real records so the
multiset comparison is tested against the thing it exists for instead of against a
flag.

The corpus is mdecrevoisier/EVTX-to-MITRE-Attack (CC0), 270+ real `.evtx` samples
mapped to ATT&CK, under `_shared_data/EVTX-to-MITRE-Attack/`.

SECURITY CHANNEL ONLY. Every sample file is filtered to `<Channel>Security`. Mixing
in PowerShell or Sysmon Operational records would put two different schemas through
one record-identity function, and two records that are "the same" would stop meaning
one thing.

## The record shape, and why the extra fields carry an underscore

`common.content_hash` hashes every key except `chain`, `seq` and anything starting
with `_`. So the identity of a record here is exactly:

    ts, host, channel, event_id, user, src_host, msg

and nothing else. Provenance - which file a record came from, its real
EventRecordID, its real timestamp, its provider - is kept, but under `_`-prefixed
names, because a record is supposed to be identifiable by what it says about the
host, not by where this script happened to read it. Putting `_record_id` in the hash
would make every record in the corpus unique and quietly destroy the exact property
the kit is built to test.

## The 1102 record is held separately

`clear_whole_log` needs the event Windows writes when a log is cleared. The corpus
has a real one, in `T1070.001-Clear Windows event logs/ID104-1102-Event log
cleared.evtx`, and `load_clear_record()` returns it. Clearing events are *excluded*
from the sampled population (see `EXCLUDED_EVENT_IDS`): a stray 1102 inside the
ordinary population would make `clearing_rules` alert on scenarios where nothing was
cleared, which is a false positive manufactured by the corpus rather than measured.

## Dependency

`python-evtx` (pure Python). pip cannot reach PyPI from this machine, so a copy is
vendored at `_shared_data/python-evtx/`; an installed `Evtx` package is preferred if
one is present. See `--summary` output for which was used.

CLI:
    python3 scripts/evtx_corpus.py --summary
"""
import argparse
import datetime
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPL = HERE.parent.parent                     # the repository root
CORPUS = IMPL / "_shared_data" / "EVTX-to-MITRE-Attack"
VENDORED_EVTX = IMPL / "_shared_data" / "python-evtx"

# Relative to CORPUS. Named explicitly rather than globbed so that the population is
# reproducible: a later `git pull` of the corpus adding files cannot silently change
# what this kit measured.
SOURCES = [
    "TA0008-Lateral Movement/T1021.002-SMB Windows Admin Shares/"
    "ID4688,4697,5140-5145 PSexec remote execution + admin share.evtx",
    "TA0008-Lateral Movement/T1021.002-SMB Windows Admin Shares/"
    "ID4688-4648 Lateral movement with net use.evtx",
    "TA0008-Lateral Movement/T1021.002-SMB Windows Admin Shares/"
    "ID4688-Network share manipulation via commandline.evtx",
    "TA0008-Lateral Movement/T1021.002-SMB Windows Admin Shares/"
    "ID5140-ADMIN$ share connection with Golden ticket.evtx",
    "TA0008-Lateral Movement/T1021.002-SMB Windows Admin Shares/"
    "ID5142- New file share created.evtx",
    "TA0008-Lateral Movement/T1021.002-SMB Windows Admin Shares/"
    "ID5142-5143-Mimispool print share created and modified.evtx",
    "TA0008-Lateral Movement/T1021.002-SMB Windows Admin Shares/"
    "ID5145-remote shell execution via SMB admin share.evtx",
    "EVTX_full_APT_attack_steps/"
    "ID4624,4688,5140,5145-Eternal Romance - MS17_010_psexec (GLOBAL).evtx",
    "EVTX_full_APT_attack_steps/"
    "ID4688,4698,4699,5145,4624-ATexec remote trask creation (GLOBAL).evtx",
    "EVTX_full_APT_attack_steps/"
    "ID4688,5140,5145-WMIexec execution via SMB (GLOBAL).evtx",
    "EVTX_full_APT_attack_steps/"
    "ID5140-5145,4688,4697-Encrypted payload deployed with service over SMB (GLOBAL).evtx",
    "EVTX_full_APT_attack_steps/ID5145-4624-DonPAPI full extraction.evtx",
    "EVTX_full_APT_attack_steps/"
    "ID47x,4661-4662,5136,4688,4697-SAM the admin (CVE-2021-42287).evtx",
    "EVTX_full_APT_attack_steps/ID4742,4935,4662,4661,5137-DCshadow attack (GLOBAL) failed.evtx",
    "EVTX_full_APT_attack_steps/ID4674,5142-Mimikatz print spool privileges requested.evtx",
    "EVTX_full_APT_attack_steps/ID4674,6416 New external device connected (PrintNightmare).evtx",
    "EVTX_full_APT_attack_steps/ID4720,4698-Fortinet APT group abuse on Windows.evtx",
]

CLEAR_SOURCE = ("TA0005-Defense Evasion/T1070.001-Clear Windows event logs/"
                "ID104-1102-Event log cleared.evtx")

# Kept out of the sampled population. 1102/104 are what `clearing_rules` looks for,
# so one of them sitting in an untampered log is a false positive created by the
# corpus. They are reachable through load_clear_record() instead.
EXCLUDED_EVENT_IDS = {1102, 104}

KEEP_CHANNEL = "Security"

# The standard text Windows shows for these ids. Only the ids actually present in the
# pooled files are listed; anything else falls back to "Security event <id>", which is
# honest rather than invented.
EVENT_NAMES = {
    1102: "The audit log was cleared",
    4624: "An account was successfully logged on",
    4627: "Group membership information",
    4648: "A logon was attempted using explicit credentials",
    4656: "A handle to an object was requested",
    4658: "The handle to an object was closed",
    4661: "A handle to an object was requested",
    4662: "An operation was performed on an object",
    4672: "Special privileges assigned to new logon",
    4673: "A privileged service was called",
    4674: "An operation was attempted on a privileged object",
    4688: "A new process has been created",
    4690: "An attempt was made to duplicate a handle to an object",
    4697: "A service was installed in the system",
    4698: "A scheduled task was created",
    4699: "A scheduled task was deleted",
    4702: "A scheduled task was updated",
    4720: "A user account was created",
    4724: "An attempt was made to reset an account's password",
    4741: "A computer account was created",
    4742: "A computer account was changed",
    4768: "A Kerberos authentication ticket (TGT) was requested",
    4769: "A Kerberos service ticket was requested",
    4776: "The computer attempted to validate the credentials for an account",
    4781: "The name of an account was changed",
    4929: "An Active Directory replica source naming context was removed",
    4935: "Replication failure begins",
    4936: "Replication failure ends",
    4964: "Special groups have been assigned to a new logon",
    4985: "The state of a transaction has changed",
    5136: "A directory service object was modified",
    5137: "A directory service object was created",
    5140: "A network share object was accessed",
    5142: "A network share object was added",
    5143: "A network share object was modified",
    5145: "A network share object was checked to see whether client can be granted access",
    5447: "A Windows Filtering Platform filter has been changed",
    6416: "A new external device was recognized by the system",
}

# Which EventData fields go into `msg`, per event id. Deliberately short lists: `msg`
# is part of the record's identity, so every field added to it is another way for two
# real records to stop being equal. The fields chosen are the ones an analyst would
# read, and they are the ones that repeat when an operation repeats.
MSG_FIELDS = {
    4624: ["LogonType", "AuthenticationPackageName"],
    4627: ["LogonType"],
    4648: ["TargetUserName", "TargetServerName"],
    4656: ["ObjectType", "ObjectName"],
    4658: ["ObjectServer", "ProcessName"],
    4661: ["ObjectType", "ObjectName"],
    4662: ["ObjectServer", "ObjectType", "OperationType"],
    4672: ["PrivilegeList"],
    4673: ["Service", "PrivilegeList"],
    4674: ["ObjectServer", "ObjectName", "PrivilegeList"],
    4688: ["NewProcessName", "CommandLine", "ParentProcessName"],
    4690: ["SourceProcessId", "TargetProcessId"],
    4697: ["ServiceName", "ServiceFileName"],
    4698: ["TaskName"],
    4699: ["TaskName"],
    4702: ["TaskName"],
    4720: ["TargetUserName", "SamAccountName"],
    4724: ["TargetUserName"],
    4741: ["TargetUserName"],
    4742: ["TargetUserName"],
    4768: ["ServiceName", "Status"],
    4769: ["ServiceName", "Status"],
    4776: ["PackageName", "Status"],
    4781: ["OldTargetUserName", "NewTargetUserName"],
    4929: ["SourceAddr", "NamingContext"],
    4935: ["AuditStatusCode"],
    4936: ["AuditStatusCode", "ReplicationStatusCode"],
    4964: ["TargetUserName"],
    4985: ["ProcessName", "NewState"],
    5136: ["ObjectDN", "AttributeLDAPDisplayName", "AttributeValue"],
    5137: ["ObjectDN", "ObjectClass"],
    5140: ["ShareName", "ShareLocalPath", "AccessMask"],
    5142: ["ShareName", "ShareLocalPath"],
    5143: ["ShareName", "ObjectType"],
    5145: ["ShareName", "RelativeTargetName", "AccessMask"],
    5447: ["FilterName", "LayerName", "ChangeType"],
    6416: ["ClassName", "DeviceDescription"],
}
# Used when an id is not in MSG_FIELDS: the first few of these that are present.
MSG_FALLBACK = ["ObjectName", "ObjectType", "ShareName", "ProcessName", "ServiceName",
                "TargetUserName", "Status"]

VALUE_CHARS = 100          # per-field cap inside msg
MSG_CHARS = 300            # whole-msg cap
NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WS = re.compile(r"\s+")


_EVTX_CACHE = None


def _import_evtx():
    """Return (Evtx class, where it came from).

    Memoised, and not only for speed: the first call puts the vendored directory on
    sys.path, so a second, uncached call would find `Evtx` importable and report it as
    an installed package. The provenance of the parser is part of what this kit reports,
    so it is recorded once, at the moment the decision was actually made.
    """
    global _EVTX_CACHE
    if _EVTX_CACHE is not None:
        return _EVTX_CACHE
    _EVTX_CACHE = _resolve_evtx()
    return _EVTX_CACHE


def _resolve_evtx():
    try:
        from Evtx.Evtx import Evtx
        return Evtx, "installed python-evtx"
    except ImportError:
        pass
    if VENDORED_EVTX.is_dir():
        sys.path.insert(0, str(VENDORED_EVTX))
        try:
            from Evtx.Evtx import Evtx
            return Evtx, f"vendored copy at {VENDORED_EVTX}"
        except ImportError as e:
            raise SystemExit(
                f"python-evtx is vendored at {VENDORED_EVTX} but will not import: {e}")
    raise SystemExit(
        "python-evtx is not available.\n"
        "  pip install --break-system-packages python-evtx\n"
        f"or place a copy of the Evtx package at {VENDORED_EVTX}.\n"
        "No .evtx file can be read without it, and this kit will not fabricate records "
        "in its place.")


def _clean(s):
    if s is None:
        return ""
    return _WS.sub(" ", _CTRL.sub("", s)).strip()


def _short(s):
    s = _clean(s)
    return s if len(s) <= VALUE_CHARS else s[:VALUE_CHARS - 1] + "…"


def _tag(el):
    t = el.tag
    return t.split("}", 1)[1] if "}" in t else t


def _parse_ts(s):
    """python-evtx renders SystemTime as 'YYYY-MM-DD HH:MM:SS.ffffff+00:00'."""
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _data_fields(root):
    """Flatten EventData/Data[@Name] and UserData's leaves into one {name: text} map."""
    out = {}
    for ed in root.iter():
        name = _tag(ed)
        if name == "EventData":
            for d in ed:
                n = d.get("Name")
                if n:
                    out.setdefault(n, _clean(d.text))
        elif name == "UserData":
            for sub in ed:
                for leaf in sub:
                    out.setdefault(_tag(leaf), _clean(leaf.text))
    return out


def _pick_user(d):
    """DOMAIN\\user for whoever the record is about.

    SubjectUserName first - the account that performed the action - falling through to
    TargetUserName when it is absent or '-' (4624 and 4776 carry no subject), then to
    the WFP records' UserName.
    """
    for user_key, dom_key in (("SubjectUserName", "SubjectDomainName"),
                              ("TargetUserName", "TargetDomainName"),
                              ("UserName", None)):
        u = d.get(user_key, "")
        if u and u != "-":
            dom = d.get(dom_key, "") if dom_key else ""
            if dom and dom != "-" and "\\" not in u:
                return f"{dom}\\{u}"
            return u
    return "-"


def _pick_src_host(d):
    """Where the activity came from, as the record reports it."""
    for k in ("IpAddress", "WorkstationName", "Workstation", "SourceAddr",
              "TargetServerName"):
        v = d.get(k, "")
        if v and v != "-":
            if v.startswith("::ffff:"):          # IPv4-mapped IPv6, as 4768/4769 write it
                v = v[len("::ffff:"):]
            return v
    return "-"


def _build_msg(event_id, d):
    name = EVENT_NAMES.get(event_id, f"Security event {event_id}")
    keys = MSG_FIELDS.get(event_id)
    if keys is None:
        keys = [k for k in MSG_FALLBACK if d.get(k)][:3]
    parts = []
    for k in keys:
        v = d.get(k, "")
        if v and v != "-":
            parts.append(f"{k}={_short(v)}")
    msg = name + (" | " + "; ".join(parts) if parts else "")
    return msg if len(msg) <= MSG_CHARS else msg[:MSG_CHARS - 1] + "…"


def normalise(xml_text, source_label, record_num=None):
    """One .evtx record -> the kit's record shape, or None if it is not Security.

    Returns the 8 identity-bearing keys plus `_`-prefixed provenance. `seq` is left as
    None here: the population has no sequence of its own, and build_log() assigns the
    sequence numbers of the log it constructs.
    """
    try:
        root = ET.fromstring(_CTRL.sub("", xml_text))
    except ET.ParseError:
        return None
    sysel = root.find(NS + "System") or root.find("System")
    if sysel is None:
        return None

    def sysget(tag, attr=None):
        el = sysel.find(NS + tag)
        if el is None:
            el = sysel.find(tag)
        if el is None:
            return ""
        return el.get(attr, "") if attr else _clean(el.text)

    channel = sysget("Channel")
    if channel != KEEP_CHANNEL:
        return None
    try:
        event_id = int(sysget("EventID") or -1)
    except ValueError:
        return None
    ts_real = _parse_ts(sysget("TimeCreated", "SystemTime"))
    if ts_real is None:
        return None
    d = _data_fields(root)
    rec = {
        "seq": None,
        "ts": ts_real.isoformat(timespec="milliseconds"),
        "host": sysget("Computer") or "-",
        "channel": channel,
        "event_id": event_id,
        "user": _pick_user(d),
        "src_host": _pick_src_host(d),
        "msg": _build_msg(event_id, d),
        # ---- provenance: excluded from content_hash by the leading underscore
        "_source_file": source_label,
        "_record_id": sysget("EventRecordID"),
        "_record_num": record_num,
        "_ts_real": ts_real.isoformat(timespec="microseconds"),
        "_provider": sysget("Provider", "Name"),
    }
    return rec


def read_file(path, source_label=None, keep_event_ids=None, drop_event_ids=None):
    """Normalised Security records from one .evtx, in file order. (recs, stats)."""
    Evtx, _ = _import_evtx()
    label = source_label or Path(path).name
    recs, seen, other_channel, unparsed, excluded = [], 0, 0, 0, 0
    with Evtx(str(path)) as e:
        for r in e.records():
            seen += 1
            try:
                xml_text = r.xml()
            except Exception:
                unparsed += 1
                continue
            rec = normalise(xml_text, label, record_num=r.record_num())
            if rec is None:
                # Either a different channel or XML this parser could not read; tell
                # the two apart so a silently shrinking population is visible.
                if "<Channel>Security</Channel>" in xml_text:
                    unparsed += 1
                else:
                    other_channel += 1
                continue
            if keep_event_ids is not None and rec["event_id"] not in keep_event_ids:
                excluded += 1
                continue
            if drop_event_ids and rec["event_id"] in drop_event_ids:
                excluded += 1
                continue
            recs.append(rec)
    return recs, {"file": label, "records_in_file": seen, "security_records": len(recs),
                  "other_channel": other_channel, "unparsed": unparsed,
                  "excluded_by_event_id": excluded}


def load_population(sources=None, corpus=None, drop_event_ids=EXCLUDED_EVENT_IDS):
    """Pool the Security records of several real .evtx files into one population.

    Returns (records, per_file_stats). Records keep file order within a file and files
    appear in SOURCES order, so `_ts_real` is monotonic inside each file - which is
    what lets build_log() keep the real same-millisecond groupings intact.
    """
    corpus = Path(corpus or CORPUS)
    if not corpus.is_dir():
        raise SystemExit(
            f"the EVTX corpus is not at {corpus}\n"
            "  cd .. && ./fetch_data.sh evtx\n"
            "clones mdecrevoisier/EVTX-to-MITRE-Attack into _shared_data/.")
    out, stats, missing = [], [], []
    for rel in (sources or SOURCES):
        p = corpus / rel
        if not p.exists():
            missing.append(rel)
            continue
        recs, st = read_file(p, source_label=Path(rel).name,
                             drop_event_ids=drop_event_ids)
        for n, r in enumerate(recs, start=1):
            r["seq"] = len(out) + n
        out.extend(recs)
        stats.append(st)
    if missing:
        raise SystemExit("these source files are named in SOURCES but are not in the "
                         "corpus:\n  " + "\n  ".join(missing))
    if not out:
        raise SystemExit(f"no Security-channel records were read from {corpus}")
    return out, stats


def load_clear_record(corpus=None, rel=CLEAR_SOURCE):
    """The REAL 1102 'audit log was cleared' record, for clear_whole_log.

    `tamper.py` used to hand-build this dict. The corpus has the genuine article -
    Microsoft-Windows-Eventlog, Security channel, EventRecordID 109054, written on
    FS03.offsec.lan - and the clearing-rules baseline is worth more when it is tested
    against the record Windows actually writes.
    """
    corpus = Path(corpus or CORPUS)
    p = corpus / rel
    if not p.exists():
        raise SystemExit(f"the clearing-log sample is not at {p}")
    recs, st = read_file(p, source_label=Path(rel).name,
                         keep_event_ids={1102})
    if not recs:
        raise SystemExit(f"{p} holds no Security-channel 1102 record")
    return recs[0], st


# --------------------------------------------------------------------- reporting
def same_instant_groups(population, precision="milliseconds"):
    """Split the population into maximal runs of records from the same file that share
    a timestamp at `precision`.

    These groups are the real log's own simultaneity: a single operation writes several
    records inside one millisecond, and a WEF/SIEM pipeline that renders a timestamp to
    milliseconds - as this kit's record shape does - shows them as having happened at
    the same instant. They are the reason genuine duplicates exist, and build_log()
    emits a group as a unit so that they survive re-timestamping.
    """
    groups, cur, key = [], [], None
    for r in population:
        dt = _parse_ts(r["_ts_real"].replace("T", " ")) or _parse_ts(r["_ts_real"])
        k = (r["_source_file"], dt.isoformat(timespec=precision) if dt else None)
        if k != key and cur:
            groups.append(cur)
            cur = []
        key = k
        cur.append(r)
    if cur:
        groups.append(cur)
    return groups


def collision_stats(records, hash_fn=None):
    """How many of these records are identical in content to another one."""
    if hash_fn is None:
        sys.path.insert(0, str(HERE))
        from common import content_hash as hash_fn
    c = Counter(hash_fn(r) for r in records)
    surplus = sum(n - 1 for n in c.values() if n > 1)
    return {"records": len(records), "distinct_content_hashes": len(c),
            "colliding_content_groups": sum(1 for n in c.values() if n > 1),
            "surplus_colliding_records": surplus,
            "collision_rate": round(surplus / len(records), 6) if records else None,
            "largest_group": max(c.values()) if c else 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", action="store_true",
                    help="pool the corpus and print what came from where")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument("--sample", type=int, default=0,
                    help="also print this many normalised records")
    a = ap.parse_args()

    _, origin = _import_evtx()
    pop, stats = load_population()
    clear_rec, clear_st = load_clear_record()
    groups = same_instant_groups(pop)
    gsizes = Counter(len(g) for g in groups)
    cs = collision_stats(pop)
    # Collisions WITHIN a same-instant group: the ones that survive re-timestamping,
    # because build_log keeps a group on one synthetic timestamp.
    in_group = 0
    sys.path.insert(0, str(HERE))
    from common import content_hash
    for g in groups:
        c = Counter(content_hash(r) for r in g)
        in_group += sum(n - 1 for n in c.values() if n > 1)

    summary = {
        "corpus": str(CORPUS),
        "parser": origin,
        "files": stats,
        "pooled_security_records": len(pop),
        "event_id_distribution": dict(Counter(r["event_id"] for r in pop).most_common()),
        "distinct_hosts": sorted({r["host"] for r in pop}),
        "distinct_users": len({r["user"] for r in pop}),
        "distinct_src_hosts": len({r["src_host"] for r in pop}),
        "excluded_event_ids": sorted(EXCLUDED_EVENT_IDS),
        "same_instant_groups": len(groups),
        "same_instant_group_sizes": dict(sorted(gsizes.items())),
        "largest_same_instant_group": max(gsizes) if gsizes else 0,
        "raw_collisions": cs,
        "surplus_collisions_within_same_instant_groups": in_group,
        "clear_record": {k: v for k, v in clear_rec.items()},
        "clear_record_file_stats": clear_st,
    }
    if a.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"corpus : {CORPUS}")
    print(f"parser : {origin}\n")
    print("Security-channel records pooled, by file:")
    for st in stats:
        print(f"  {st['security_records']:5d}  {st['file'][:76]}")
        extra = []
        if st["other_channel"]:
            extra.append(f"{st['other_channel']} non-Security")
        if st["excluded_by_event_id"]:
            extra.append(f"{st['excluded_by_event_id']} excluded (1102/104)")
        if st["unparsed"]:
            extra.append(f"{st['unparsed']} UNPARSED")
        if extra:
            print(f"         ({st['records_in_file']} records in file: "
                  + ", ".join(extra) + ")")
    print(f"\n  TOTAL {len(pop)} real Security records pooled from "
          f"{len(stats)} files")
    print(f"\nevent ids: " + ", ".join(
        f"{k}x{v}" for k, v in Counter(r['event_id'] for r in pop).most_common()))
    print(f"hosts    : {', '.join(sorted({r['host'] for r in pop}))}")
    print(f"users    : {len({r['user'] for r in pop})} distinct   "
          f"src_hosts: {len({r['src_host'] for r in pop})} distinct")
    print(f"\nsame-instant groups (same file, same millisecond): {len(groups)}, "
          f"sizes {dict(sorted(gsizes.items()))}")
    print(f"content collisions in the raw pool: {cs['surplus_colliding_records']} "
          f"surplus records in {cs['colliding_content_groups']} groups "
          f"(rate {cs['collision_rate']}, largest group {cs['largest_group']})")
    print(f"  of those, {in_group} are inside a single same-instant group, which is "
          f"what survives re-timestamping")
    print(f"\nreal 1102 for clear_whole_log:")
    print("  " + json.dumps({k: v for k, v in clear_rec.items()
                             if not k.startswith("_")}))
    print("  provenance: " + json.dumps({k: v for k, v in clear_rec.items()
                                         if k.startswith("_")}))
    for r in pop[:a.sample]:
        print("  " + json.dumps(r))


if __name__ == "__main__":
    main()
