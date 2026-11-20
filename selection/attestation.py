#!/usr/bin/env python3
"""Group attestation and co-attestation for the five selected techniques.

Reproduces the "Grp." column of Table I and the co-attestation figures in
finding F3 of the paper from the MITRE ATT&CK Enterprise STIX 2.1 bundle.

For each technique it counts the distinct ATT&CK groups (intrusion-set objects)
documented as using the technique or any of its sub-techniques. A campaign that
uses the technique is credited to the group it is attributed to, which is how
the ATT&CK website presents a group. Revoked and deprecated objects are skipped.

Expected co-attestation under independence is n1*n2/N, where N is the number of
groups with at least one mapped technique.

Usage:
    python3 attestation.py enterprise-attack.json [--md attestation.md]

The paper uses ATT&CK Enterprise v19.2, from
https://github.com/mitre-attack/attack-stix-data (enterprise-attack/).
"""
import argparse
import json
from collections import defaultdict

CHAIN = ["T1593", "T1190", "T1021", "T1486", "T1070"]
EXTRA = [("T1055", "excluded by the observability condition"),
         ("T1685", "log clearing, moved out of T1070 in v19")]
LOG_CLEARING = {"T1685.005", "T1685.006"}


def live(o):
    return not o.get("revoked", False) and not o.get("x_mitre_deprecated", False)


def ext_id(o):
    for r in o.get("external_references", []):
        if r.get("source_name") == "mitre-attack" and r.get("external_id"):
            return r["external_id"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--md")
    a = ap.parse_args()
    objs = json.load(open(a.bundle, encoding="utf-8"))["objects"]

    version = "unknown"
    for o in objs:
        if o.get("type") == "x-mitre-collection":
            version = "%s (modified %s)" % (o.get("x_mitre_version", "?"), o.get("modified", "?")[:10])

    tech, groups, camps = {}, {}, set()
    for o in objs:
        if not live(o):
            continue
        if o["type"] == "attack-pattern" and ext_id(o):
            tech[o["id"]] = ext_id(o)
        elif o["type"] == "intrusion-set":
            groups[o["id"]] = o.get("name")
        elif o["type"] == "campaign":
            camps.add(o["id"])

    g_uses = defaultdict(set)        # full technique id -> groups (direct)
    c_uses = defaultdict(set)        # full technique id -> campaigns
    c_attr = defaultdict(set)        # campaign -> groups
    for o in objs:
        if o.get("type") != "relationship" or not live(o):
            continue
        s, r, t = o.get("source_ref"), o.get("relationship_type"), o.get("target_ref")
        if r == "uses" and t in tech:
            if s in groups:
                g_uses[tech[t]].add(s)
            elif s in camps:
                c_uses[tech[t]].add(s)
        elif r == "attributed-to" and s in camps and t in groups:
            c_attr[s].add(t)

    def attested(match):
        out = set()
        for tid in set(g_uses) | set(c_uses):
            if match(tid):
                out |= g_uses[tid]
                for c in c_uses[tid]:
                    out |= c_attr[c]
        return out

    def parent(p):
        return attested(lambda tid: tid.split(".")[0] == p)

    def direct(p):
        out = set()
        for tid, gs in g_uses.items():
            if tid.split(".")[0] == p:
                out |= gs
        return out

    with_any = attested(lambda tid: True)
    n = len(with_any)
    sets = {p: parent(p) for p in CHAIN}

    L = ["ATT&CK Enterprise version: %s" % version,
         "Live groups: %d; groups with at least one mapped technique (N): %d" % (len(groups), n), "",
         "| Technique | Groups (direct) | Groups (+campaigns) | % of N |", "|---|---|---|---|"]
    for p in CHAIN:
        L.append("| %s | %d | %d | %.1f%% |" % (p, len(direct(p)), len(sets[p]), 100.0 * len(sets[p]) / n))
    for p, note in EXTRA:
        g = parent(p)
        L.append("| %s (%s) | %d | %d | %.1f%% |" % (p, note, len(direct(p)), len(g), 100.0 * len(g) / n))
    t1070_wide = sets["T1070"] | attested(lambda tid: tid in LOG_CLEARING)
    L += ["", "T1070 including the v19 log-clearing sub-techniques T1685.005/.006: %d groups" % len(t1070_wide),
          "", "Co-attestation of consecutive stages (expected under independence = n1*n2/N):", "",
          "| Pair | Observed | Expected | Lift |", "|---|---|---|---|"]
    for x, y in zip(CHAIN, CHAIN[1:]):
        obs = len(sets[x] & sets[y])
        exp = len(sets[x]) * len(sets[y]) / n
        L.append("| %s + %s | %d | %.1f | %.2f |" % (x, y, obs, exp, obs / exp))
    allfive = set.intersection(*sets.values())
    L += ["", "Attested for all five: %s" % (", ".join(sorted(groups[g] for g in allfive)) or "none")]
    out = "\n".join(L)
    print(out)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
