"""Grade every logged event as touch, probe or use, and group sources into campaigns.

The grading follows the T1593 design in the paper:

  touch  resolving or fetching a decoy name
  probe  requesting vulnerability paths, sending unsolicited POST requests, or
         guessing a generic account name
  use    acting on a decoy that could only have been harvested from the channel
         where it was planted - the unadvertised upgrade path, the planted API key,
         the planted credential, or a planted decoy service account

Two rules keep the use grade honest, and they are the whole value of the headline
number:

  1. Nothing the server publishes to every visitor can produce a use. The landing
     page and robots.txt advertise the administration page and nothing else, and
     that page grades as a PROBE. The upgrade path is advertised nowhere.
  2. A decoy token arriving in a QUERY VALUE is not a use. Query values are copied
     out of one response into the next request for nothing at all - by link
     checkers, by analytics, by anything replaying a URL - so a visitor who had
     read any page could otherwise mint unlimited uses with /?x=<token>. Such an
     event grades as a probe with its own reason, so it is still counted and still
     visible, but it does not reach the headline number.

Touches are context only, because most come from bots that do nothing malicious
(Kondracki et al. 2022). Probes from campaigns whose paths match ordinary
opportunistic scanning are suppressed. A use raises an alert and is never
suppressed.
"""
import argparse, datetime, ipaddress, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The shared anchor module sits beside the kits. APPENDED, not inserted, so that this
# kit's own scripts/common.py keeps shadowing the repository's common/ package.
sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from common import (EVENT_LOG, GENERIC_ACCOUNT_NAMES, RESULTS, load_plan, log,
                    read_jsonl)
import anchors
import bridge

UTC = datetime.timezone.utc

# A per-day rate needs at least this many hours of exposure behind it. The floor used
# to be six hours, where a single event implied 4.0 uses per day: a figure invented by
# the denominator rather than measured. Three days is the shortest window in which one
# arrival does not dominate the rate it is divided into.
MIN_EXPOSURE_HOURS = 72

# Share of events whose source network is loopback above which nothing is computed.
# The server binds 127.0.0.1 behind a reverse proxy, so it sees the proxy as its peer
# and depends on X-Forwarded-For to learn the real source. If that header is absent or
# stripped, every request is recorded as 127.0.0.0/24: all traffic collapses into ONE
# campaign, unique_clients becomes 1, campaign suppression cannot work because there is
# only one campaign to judge, and every per-day rate becomes a rate per single
# fictional source. That is not a result, so it is not reported as one.
MAX_LOOPBACK_SHARE = 0.20

# Paths that untargeted scanners request everywhere. Seeing these says nothing about
# our decoys, so campaigns made only of these are suppressed.
SCAN_PATTERNS = [
    r"^/wp-(login|admin|content|includes)", r"^/wordpress", r"^/xmlrpc\.php",
    r"^/\.env", r"^/\.git", r"^/\.aws", r"^/\.ssh", r"^/\.vscode", r"^/\.DS_Store",
    r"^/phpmyadmin", r"^/pma", r"^/adminer", r"^/mysql",
    r"^/vendor/", r"^/config\.(json|php|yml)$", r"^/backup", r"^/dump",
    r"^/actuator", r"^/solr", r"^/jenkins", r"^/manager/html", r"^/cgi-bin/",
    r"^/boaform", r"^/HNAP1", r"^/shell", r"^/hudson", r"^/\.well-known/security",
    r"^/owa/", r"^/autodiscover", r"^/ecp/", r"^/telescope", r"^/debug",
    r"^/server-status", r"^/login\.action", r"^/struts", r"^/druid",
]
SCAN_RE = [re.compile(p, re.I) for p in SCAN_PATTERNS]

# User agents that announce themselves as research scanners or secret scanners.
KNOWN_SCANNER_UA = [
    "censys", "shodan", "binaryedge", "internet-measurement", "netsystemsresearch",
    "leakix", "paloaltonetworks", "cyber.casa", "scanworld", "odin", "zgrab",
    "masscan", "nuclei", "expanse", "driftnet", "stretchoid", "gdnplus",
    "github", "gitleaks", "trufflehog", "secret-scanner", "crawler", "spider",
]
# The bare substring "bot" used to be in this list. It is a self-declaration, not a
# name: anything that wanted its probes suppressed only had to put "bot" in its user
# agent. Named scanners stay, because a scanner that names itself Censys or nuclei is
# telling the truth about something it gains nothing by claiming.

def is_scan_path(path):
    return any(r.search(path or "") for r in SCAN_RE)


def net_of(ip):
    """The /24 (v4) or /48 (v6) containing an address.

    decoy_server.py truncates at the moment of recording, so this is normally a
    no-op that simply passes the already-truncated network through. It stays here
    so that logs captured before that change, and the synthetic selftest log, still
    group into campaigns correctly.
    """
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if a.version == 4:
        return str(ipaddress.ip_network(f"{ip}/24", strict=False))
    return str(ipaddress.ip_network(f"{ip}/48", strict=False))


def is_loopback_net(s):
    """Is this recorded source a loopback network or address?"""
    txt = (s or "").strip()
    if not txt:
        return False
    try:
        return ipaddress.ip_network(txt, strict=False).network_address.is_loopback
    except ValueError:
        pass
    try:
        return ipaddress.ip_address(txt).is_loopback
    except ValueError:
        return False


def client_of(ev):
    """The source network to attribute an event to.

    Caddy APPENDS the immediate peer to any X-Forwarded-For the client supplied, so
    the RIGHTMOST element is the only hop our own proxy observed and everything to
    the left of it is attacker-controlled text. Trusting the leftmost element let a
    scanner fabricate addresses across many /24s to defeat campaign suppression, or
    collapse itself into one network to hide its spread. We therefore take the last
    element, and fall back to the peer address the server itself saw.
    """
    for key in ("forwarded_for_net", "forwarded_for"):
        parts = [p.strip() for p in (ev.get(key) or "").split(",") if p.strip()]
        if parts:
            return parts[-1]
    return ev.get("client_net") or ev.get("client_ip", "")


def decoy_accounts(plan):
    """channel -> decoy account name, for every channel that advertises one."""
    a = plan["advertised"]
    users = dict(a.get("decoy_users") or {})
    if not users and a.get("decoy_user"):
        users["web"] = a["decoy_user"]        # plans written before per-channel accounts
    return users


def classify(ev, plan):
    a = plan["advertised"]
    path = ev.get("path", "")
    tokens = {d["token"]: d["channel"] for d in plan["decoys"]}
    users = decoy_accounts(plan)
    submitted = (ev.get("form_username") or "").strip()

    # ---- use: only possible after reading something we planted
    if path == a["vuln_path"]:
        return "use", "web", "requested the advertised upgrade path"
    if ev.get("api_key_header") and ev["api_key_header"] == a["api_key"]:
        return "use", "repo", "presented the planted API key"
    blob = json.dumps(ev.get("query", {})) + " " + path
    for tok, ch in tokens.items():
        if tok in blob:
            return "use", ch, f"carried the {ch} decoy token"
    for ch, name in sorted(users.items()):
        if submitted and submitted == name:
            if ev.get("decoy_credential_used"):
                return "use", ch, f"signed in with the planted {ch} decoy credential"
            return "use", ch, f"tried the planted {ch} decoy service account"
    # Fall back on the flags the server recorded, for logs whose plan has since moved on.
    if ev.get("decoy_credential_used") or ev.get("decoy_username_used"):
        ch = ev.get("decoy_user_channel") or "web"
        return "use", ch, f"server recorded a planted {ch} decoy account attempt"

    # ---- probe
    if submitted and submitted.lower() in GENERIC_ACCOUNT_NAMES:
        return "probe", None, f"guessed the generic account name {submitted!r}"
    if ev.get("method") in ("POST", "PUT", "PATCH", "DELETE"):
        return "probe", None, f"unsolicited {ev['method']}"
    if is_scan_path(path):
        return "probe", None, "requested a path typical of untargeted scanning"
    if path.rstrip("/") == a["admin_path"].rstrip("/"):
        return "probe", None, "requested the advertised administration page"

    # ---- touch
    return "touch", None, "fetched a decoy name"


LINK = "A1"          # stage 1 (reconnaissance, T1593) hands this to stage 2 (T1190)


def a1_subject(channel, plan):
    """The identifier an A1 anchor names, in THIS kit's namespace.

    Link A1's policy requires subject_kind "domain", and only the ct channel plants
    something that is a domain: a hostname published through Certificate Transparency.
    The web channel plants a path, the repo channel an API key, the social channel an
    address. None of those is a domain, so the subject for those channels is the
    channel name rendered as a label under the decoy site's own domain. That is a
    NAMING CONVENTION, declared here and matched in common/bridge.json - it is not a
    claim that such a host exists. It is deterministic and distinct per channel, which
    is what the binding needs, and it keeps the subject_kind the policy demands rather
    than weakening the policy to fit the data.
    """
    for d in plan["decoys"]:
        if d["channel"] == channel and d.get("kind") == "hostname" and d.get("value"):
            return d["value"]
    return f"{channel}.{plan.get('domain') or 'decoy.invalid'}"


def parse_ts(s):
    """Parse a timestamp and force it tz-aware (UTC).

    Event times carry an offset but a published_at written by an older run, or by
    hand, may be naive. Comparing the two raises TypeError, and it would raise it
    only at grading time, after the exposure is over. Coerce defensively instead.
    """
    if isinstance(s, datetime.datetime):
        t = s
    else:
        try:
            t = datetime.datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
        except Exception:
            return None
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def merge_spans(spans):
    """Merge overlapping [start, end) spans.

    Two outage records that overlap would otherwise have their shared hours removed
    from the exposure window twice, which inflates every rate divided into it.
    """
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def load_outages(plan, until):
    """Outage spans from the plan, tz-aware, merged, open ones closed at `until`."""
    spans = []
    for o in plan.get("outages") or []:
        s = parse_ts(o.get("start"))
        if not s:
            continue
        e = parse_ts(o.get("end")) or until
        if e > s:
            spans.append((s, e))
    return merge_spans(spans)


def overlap_hours(spans, lo, hi):
    """Hours of `spans` lying inside [lo, hi]."""
    total = 0.0
    for s, e in spans:
        a, b = max(s, lo), min(e, hi)
        if b > a:
            total += (b - a).total_seconds() / 3600
    return total


def inside_any(spans, t):
    return any(s <= t < e for s, e in spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(EVENT_LOG))
    ap.add_argument("--out", default=str(RESULTS / "graded.json"))
    ap.add_argument("--events-out", default=str(RESULTS / "graded_events.jsonl"),
                    help="every graded row, one per line, so a re-grade never needs a re-run")
    ap.add_argument("--until", help="end of the exposure window; ISO, default now (UTC)")
    ap.add_argument("--tag", default="default",
                    help="names the operating point this grading writes, so two gradings "
                         "of different exposures do not overwrite each other")
    ap.add_argument("--prior", type=float, default=0.01,
                    help="base rate of reconnaissance among the sources we see. Reporting "
                         "only: it turns this kit's operating point into the confidence an "
                         "A1 anchor built from it would carry, so the author can see what "
                         "a decoy use is worth to stage 2.")
    ap.add_argument("--anchor-ttl", type=float, default=86400.0,
                    help="seconds an emitted A1 anchor stays valid after the decoy use "
                         "that produced it. Harvested material is acted on later or not "
                         "at all, so the window is a day by default rather than open.")
    ap.add_argument("--anchors-out", default=None,
                    help="where the A1 anchors go (default results/anchors_<tag>.jsonl)")
    ap.add_argument("--no-anchors", action="store_true",
                    help="grade as before and emit no anchor file at all")
    bridge.add_arguments(ap, link="A1")
    a = ap.parse_args()

    br = bridge.load(a.bridge, no_bridge=a.no_bridge)

    plan = load_plan()
    events = read_jsonl(a.log)
    if not events:
        raise SystemExit(f"no events in {a.log} yet")

    until = parse_ts(a.until) if a.until else datetime.datetime.now(UTC)
    if until is None:
        raise SystemExit(f"--until {a.until!r} is not an ISO timestamp")

    graded, by_campaign = [], defaultdict(list)
    for ev in events:
        grade, channel, why = classify(ev, plan)
        ip = client_of(ev)
        row = dict(ev, grade=grade, channel=channel, reason=why,
                   client=ip, net=net_of(ip))
        graded.append(row)
        by_campaign[row["net"]].append(row)

    # ---- campaigns: source infrastructure plus the paths it asked for (Kondracki et al.)
    campaigns = {}
    for net, rows in by_campaign.items():
        paths = [r["path"] for r in rows]
        scan_share = sum(is_scan_path(p) for p in paths) / len(paths)
        uas = Counter(r.get("user_agent", "") for r in rows)
        top_ua = uas.most_common(1)[0][0] if uas else ""
        known = any(k in top_ua.lower() for k in KNOWN_SCANNER_UA)
        grades = Counter(r["grade"] for r in rows)
        campaigns[net] = dict(
            requests=len(rows),
            # Addresses are truncated to a network when recorded, so this counts distinct
            # recorded values, which for IPv4 is normally one per campaign by construction.
            unique_client_nets=len({r["client"] for r in rows}),
            unique_paths=len(set(paths)), scan_path_share=round(scan_share, 3),
            top_user_agent=top_ua[:160], looks_like_known_scanner=known,
            grades=dict(grades),
            first_seen=min(r["ts"] for r in rows), last_seen=max(r["ts"] for r in rows),
            suppressed=bool(scan_share >= 0.5 or known),
        )

    # suppression applies to probes only; a use always alerts
    alerts, suppressed_probes = [], 0
    for r in graded:
        c = campaigns[r["net"]]
        if r["grade"] == "use":
            alerts.append(r)
        elif r["grade"] == "probe" and c["suppressed"]:
            suppressed_probes += 1

    # ---- the exposure window: publication to `until`, NOT the span of observed traffic.
    # A rate over the observed span is inflated by exactly the dead time between
    # publication and the first event, which is the interval the design is measuring.
    pubs = {}
    for d in plan["decoys"]:
        t = parse_ts(d["published_at"]) if d["published_at"] else None
        if t:
            pubs[d["channel"]] = t
    exposure_start = min(pubs.values()) if pubs else None
    # Downtime is not exposure. Every interval recorded by
    # 01_record_publication.py --outage-start/--outage-end comes out of the
    # denominator: a decoy that cannot be reached cannot be used, and dividing by
    # hours it was unreachable understates every rate by exactly that fraction.
    outages = load_outages(plan, until)
    exposure_h_gross = ((until - exposure_start).total_seconds() / 3600
                        if exposure_start else None)
    outage_h = (overlap_hours(outages, exposure_start, until)
                if exposure_start else 0.0)
    exposure_h = (exposure_h_gross - outage_h
                  if exposure_h_gross is not None else None)
    exposure_by_channel = {
        ch: round((until - t).total_seconds() / 3600
                  - overlap_hours(outages, t, until), 2)
        for ch, t in sorted(pubs.items())}

    # ---- time from publication to first event of each grade, per channel
    timing = {}
    for d in plan["decoys"]:
        pub = pubs.get(d["channel"])
        ch = d["channel"]
        rows = [r for r in graded if r["channel"] == ch] or []
        entry = {"published_at": d["published_at"], "uses": len(rows),
                 "exposure_hours": exposure_by_channel.get(ch)}
        if pub:
            for grade in ("touch", "probe", "use"):
                pool = [r for r in graded if r["grade"] == grade and
                        (r["channel"] == ch or grade != "use")]
                ts = [parse_ts(r["ts"]) for r in pool]
                ts = [t for t in ts if t and t >= pub]
                entry[f"minutes_to_first_{grade}"] = (
                    round((min(ts) - pub).total_seconds() / 60, 1) if ts else None)
        timing[ch] = entry

    # the span of traffic actually seen, kept as its own field for comparison
    tss = [t for t in (parse_ts(r["ts"]) for r in graded) if t]
    observed_span_h = ((max(tss) - min(tss)).total_seconds() / 3600
                       if len(tss) > 1 else None)

    # Rates are over the exposure window, so only events inside it are counted.
    in_window, events_inside_outages = [], []
    for r in graded:
        t = parse_ts(r["ts"]) or until
        if not exposure_start or t < exposure_start:
            continue
        if inside_any(outages, t):
            # An event timestamped inside an outage means the outage record is wrong,
            # not that the event is. Held out of the rate and surfaced, because a
            # denominator that removes hours in which traffic arrived is worse than
            # one that removes none.
            events_inside_outages.append(r)
            continue
        in_window.append(r)
    counts = Counter(r["grade"] for r in graded)
    counts_in_window = Counter(r["grade"] for r in in_window)
    alerts_in_window = [r for r in in_window if r["grade"] == "use"]
    rateable = bool(exposure_h and exposure_h >= MIN_EXPOSURE_HOURS)

    # ---- the operating point this kit hands to stage 2 (link A1).
    # tpr is 1.0 BY CONSTRUCTION and must be read as such: a use is DEFINED as acting on
    # something that could only have been harvested from the channel where it was
    # planted, so the detector cannot miss one. The informative half is the other one.
    # A use attributed to a campaign that looks like untargeted scanning is the honest
    # false positive here - it is a use we would not want stage 2 to act on - and the
    # denominator is every event that was not a use, which is the traffic that had the
    # opportunity to produce one.
    n_uses = counts["use"]
    n_non_use = len(graded) - n_uses
    fp_uses = sum(1 for r in alerts if campaigns[r["net"]]["suppressed"])
    op_path = RESULTS / f"operating_point_{a.tag}.json"
    own_op = anchors.OperatingPoint(
        tpr=1.0,
        fpr=(fp_uses / n_non_use) if n_non_use else 0.0,
        n_pos=n_uses, n_neg=n_non_use,
        # a use is a definition, not a score, so there is no threshold to record
        threshold=None, detector="T1593 decoy use")
    RESULTS.mkdir(parents=True, exist_ok=True)
    own_op.save(op_path)
    # The operating point can be REFUSED as evidence: LR+ at or below 1, or a measured
    # FPR of zero with no negatives recorded. common/anchors.py raises AnchorError for
    # both, and an emitter that let that propagate would lose the grading as well as the
    # anchors. Catch it, say why, and emit nothing.
    op_implied, op_refused = None, None
    try:
        op_implied = anchors.confidence_from_operating_point(own_op, a.prior)
    except anchors.AnchorError as e:
        op_refused = str(e)
        log(f"\nREFUSED: this grading cannot produce an A1 anchor. {op_refused}")

    # ---- link A1: emit one anchor per graded USE, from REAL grading output.
    # The detections are real: a use is a request this kit's own grader classified as
    # acting on planted material. What is assumed is only which corpus site the decoy
    # channel stands for, and that assumption lives in common/bridge.json.
    anchors_path = Path(a.anchors_out) if a.anchors_out else RESULTS / f"anchors_{a.tag}.jsonl"
    a1 = {"link": LINK, "emitted": 0, "skipped": 0, "subjects": {}, "refused": op_refused,
          "anchors_file": None, "ttl_seconds": a.anchor_ttl,
          "unbound_subjects": [], "note": ""}
    if a.no_anchors:
        a1["note"] = "--no-anchors: nothing emitted"
        log("\n--no-anchors: no A1 anchor file written")
    elif op_refused:
        a1["note"] = ("the operating point was refused as evidence, so no anchor was "
                      "emitted: " + op_refused)
    elif not alerts:
        a1["note"] = "no graded use in this log, so there is nothing to anchor"
        log("\nno graded use in this log: link A1 emits nothing")
    else:
        anchors_path.parent.mkdir(parents=True, exist_ok=True)
        anchors_path.write_text("")        # one emission log per tagged grading
        store = anchors.AnchorStore(anchors_path)
        for r in alerts:
            t = parse_ts(r["ts"])
            if t is None:
                a1["skipped"] += 1
                continue
            subj = a1_subject(r["channel"] or "web", plan)
            bound = br.resolve("T1593", "domain", subj, "T1190")
            if bound is None and subj not in a1["unbound_subjects"]:
                a1["unbound_subjects"].append(subj)
            try:
                store.emit(anchors.Anchor.create(
                    LINK, "domain", subj, emitted_at=t.timestamp(),
                    ttl_seconds=a.anchor_ttl,
                    operating_point=own_op, prior=a.prior,
                    evidence={
                        "emitting_stage": "stage 1 reconnaissance (T1593)",
                        "detection": "graded decoy use",
                        "reason": r["reason"], "channel": r["channel"],
                        "path": r.get("path", ""), "client_net": r["net"],
                        "campaign_suppressed": campaigns[r["net"]]["suppressed"],
                        "campaign_looks_like_known_scanner":
                            campaigns[r["net"]]["looks_like_known_scanner"],
                        "subject_naming": ("the ct channel's subject is the published "
                                           "hostname; every other channel's is the channel "
                                           "name under the decoy domain, a declared "
                                           "convention, not a host that exists"),
                        "bridge_binding": (bound.as_dict() if bound else None),
                        "bridge_status": br.status,
                    },
                    note="real grading output: one anchor per graded use"))
                a1["emitted"] += 1
                a1["subjects"][subj] = a1["subjects"].get(subj, 0) + 1
            except anchors.AnchorError as e:
                a1["skipped"] += 1
                a1["refused"] = str(e)
                log(f"  REFUSED one A1 anchor: {e}")
        a1["anchors_file"] = anchors_path.name
        reloaded = anchors.AnchorStore.load(anchors_path)
        a1["loaded_anchor_summary"] = reloaded.summary()
        log(f"\n--- link A1: anchors emitted for stage 2 (T1190) ---")
        log(f"  wrote {anchors_path.name}: {a1['emitted']} anchor(s) from "
            f"{len(alerts)} graded use(s), subject_kind 'domain', confidence "
            f"{op_implied:.4f} at prior {a.prior}, TTL {a.anchor_ttl:g}s from each use")
        for subj, n in sorted(a1["subjects"].items()):
            b = br.resolve("T1593", "domain", subj, "T1190")
            log(f"    {subj:34s} {n:3d} use(s) -> "
                + (f"T1190 {b.to_subject_kind} {b.to_subject!r}" if b
                   else "NOT BOUND in common/bridge.json: T1190 will never match this"))
        if a1["unbound_subjects"]:
            log(f"  WARNING these subjects have no binding and so cannot reach stage 2: "
                f"{a1['unbound_subjects']}")
        log(f"  bridge: {br.status}")
        log(f"  next: python3 ../T1190_aggregation/scripts/04_run_experiment.py "
            f"--anchor-file {anchors_path}")
    # n_matched is 0 by nature: this kit only EMITS A1. a1["emitted"] is the count
    # that matters here, and the bound/unbound split above says how many can match.
    a1.update(br.result_block(LINK, anchor_file=a1["anchors_file"], n_matched=0))

    out = dict(
        anchor_emission=a1,
        generated=datetime.datetime.now(UTC).isoformat(timespec="seconds"),
        operating_point=own_op.as_dict(),
        operating_point_file=op_path.name,
        operating_point_prior_for_report=a.prior,
        operating_point_implied_confidence=(None if op_implied is None
                                            else round(op_implied, 6)),
        operating_point_refused=op_refused,
        operating_point_note=("tpr is 1.0 by construction (a use is defined as a use), so "
                              "it is not a result. fpr is graded uses attributable to a "
                              "suppressed, scanner-looking campaign over every non-use "
                              "event. With no use observed the file is still written, at "
                              "fpr 0 over the real denominator, so the rule-of-three floor "
                              "in common/anchors.py decides what the anchor is worth "
                              "rather than an unbounded likelihood ratio."),
        log=a.log, n_events=len(graded),
        exposure_start=exposure_start.isoformat(timespec="seconds") if exposure_start else None,
        exposure_until=until.isoformat(timespec="seconds"),
        exposure_hours=round(exposure_h, 2) if exposure_h else None,
        exposure_hours_gross=round(exposure_h_gross, 2) if exposure_h_gross else None,
        outage_hours=round(outage_h, 2),
        outages=[{"start": s.isoformat(timespec="seconds"),
                  "end": e.isoformat(timespec="seconds"),
                  "hours": round((e - s).total_seconds() / 3600, 2)}
                 for s, e in outages],
        outages_recorded=plan.get("outages") or [],
        exposure_definition=("publication to --until, MINUS every recorded outage. "
                             "exposure_hours is the net figure and is the denominator "
                             "of every rate below; exposure_hours_gross is before "
                             "subtraction and is reported only so the two can be "
                             "compared."),
        n_events_inside_outages=len(events_inside_outages),
        events_inside_outages_note=("events timestamped inside a recorded outage. Any "
                                    "non-zero count means an outage interval is wrong, "
                                    "since traffic arrived during it; these rows are "
                                    "excluded from the rates until it is corrected."),
        exposure_hours_by_channel=exposure_by_channel,
        observed_span_hours=round(observed_span_h, 2) if observed_span_h else None,
        # kept under its old name so earlier notes still resolve; it is the exposure window
        observation_hours=round(exposure_h, 2) if exposure_h else None,
        counts=dict(counts),
        counts_in_exposure_window=dict(counts_in_window),
        n_events_in_exposure_window=len(in_window),
        unique_clients=len({r["client"] for r in graded}),
        client_granularity=("/24 for IPv4 and /48 for IPv6: addresses are truncated when "
                            "recorded, so a 'client' here is a network, not a host"),
        unique_campaigns=len(campaigns),
        suppressed_campaigns=sum(1 for c in campaigns.values() if c["suppressed"]),
        suppressed_probes=suppressed_probes,
        alerts=[dict(ts=r["ts"], client=r["client"], channel=r["channel"], path=r["path"],
                     reason=r["reason"], user_agent=r.get("user_agent", "")[:160],
                     campaign_looks_like_known_scanner=campaigns[r["net"]]["looks_like_known_scanner"])
                for r in alerts],
        # Rates are per day of EXPOSURE, and only meaningful once that window is a window.
        alerts_per_day=(round(len(alerts_in_window) / (exposure_h / 24), 2) if rateable else None),
        touches_per_day=(round(counts_in_window["touch"] / (exposure_h / 24), 2) if rateable else None),
        rate_note=(None if rateable else
                   ("no decoy has been published, so there is no exposure window and no rate"
                    if not exposure_start else
                    f"net exposure under {MIN_EXPOSURE_HOURS} h, so per-day rates "
                    f"are not reported")),
        rate_basis=("events inside the exposure window, over the hours from the first "
                    "publication to --until LESS recorded downtime; the observed "
                    "traffic span is reported separately and is NOT the denominator"),
        timing_by_channel=timing,
        campaigns=dict(sorted(campaigns.items(), key=lambda kv: -kv[1]["requests"])[:60]),
        campaigns_truncated_to=60,
        campaigns_total=len(campaigns),
        graded_events_file=a.events_out,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    log(f"wrote {a.out}")

    # Every graded row, so unique_campaigns and suppressed_campaigns can be audited
    # without re-running an exposure that cannot be repeated.
    ep = Path(a.events_out)
    ep.parent.mkdir(parents=True, exist_ok=True)
    with ep.open("w") as f:
        for r in graded:
            f.write(json.dumps({
                "ts": r["ts"], "grade": r["grade"], "channel": r["channel"],
                "reason": r["reason"], "client_net": r["net"], "campaign": r["net"],
                "campaign_suppressed": campaigns[r["net"]]["suppressed"],
                "method": r.get("method", ""), "path": r.get("path", ""),
                "user_agent": (r.get("user_agent", "") or "")[:160],
            }) + "\n")
    log(f"wrote {ep} ({len(graded)} graded rows)")

    log(f"\n--- the operating point this kit hands to stage 2 (link A1) ---")
    log(f"  wrote {op_path.name}: tpr {own_op.tpr} over {own_op.n_pos} use(s) "
        f"[1.0 BY CONSTRUCTION, not a measurement], fpr {own_op.fpr:.6g} "
        f"({fp_uses}/{own_op.n_neg} non-use events), effective fpr "
        f"{own_op.effective_fpr():.6g}, LR+ {own_op.lr_plus():.6g}")
    if op_implied is None:
        log(f"  REFUSED as evidence, so no A1 anchor is derivable: {op_refused}")
    else:
        log(f"  at prior {a.prior} an A1 anchor built from it would carry confidence "
            f"{op_implied:.4f} (A1 may only reprioritise: its policy max_drop is 0.25)")

    log(f"\nevents {len(graded)} ({len(in_window)} inside the exposure window) from "
        f"{out['unique_clients']} networks in {out['unique_campaigns']} campaigns")
    log(f"  exposure window : {out['exposure_hours']} h net "
        f"({out['exposure_hours_gross']} h gross less {out['outage_hours']} h downtime; "
        f"traffic observed over {out['observed_span_hours']} h)")
    if events_inside_outages:
        log(f"  WARNING: {len(events_inside_outages)} event(s) are timestamped inside a "
            f"recorded outage, so an outage interval is wrong. First: "
            f"{events_inside_outages[0]['ts']}. Fix it in decoys.json and re-grade.")
    for g in ("touch", "probe", "use"):
        log(f"  {g:6s} {counts[g]} (in window {counts_in_window[g]})")
    log(f"  suppressed probes: {suppressed_probes} "
        f"(from {out['suppressed_campaigns']} campaigns that look untargeted)")
    log(f"  alerts: {len(alerts)}"
        + (f"  ({out['alerts_per_day']}/day of exposure)" if out["alerts_per_day"] is not None
           else f"  ({out['rate_note']})"))
    for al in alerts[:10]:
        log(f"    {al['ts']}  {str(al['channel']):6s} {al['reason']}  <- {al['client']}")
    log("\nnext: python3 scripts/03_bundle_results.py")


if __name__ == "__main__":
    main()
