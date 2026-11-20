"""The entity binding between two kits that share no namespace.

The five kits run on five unrelated datasets. T1190 scores FILES from GitHub
repositories; T1021 sees HOSTS in a LANL-derived authentication log; T1486 sees
client SESSIONS in a simulated file server; T1070 sees one HOST's event log; T1593
sees source networks hitting a decoy. A web shell detected in a corpus file is not
on a host the LANL log can see, and nothing in either dataset says it is.

So a link between two stages cannot be real end to end here, and pretending
otherwise is the defect this module exists to make visible rather than to hide.
What this module provides is the smaller, honest thing:

  * the DETECTIONS driving each anchor are real detector output from the emitting
    kit, not its ground truth;
  * the ENTITY BINDING from one kit's subject universe to the next's is DECLARED -
    written down in common/bridge.json with a stated basis, copied into every
    result file that depends on it, and switchable off with --no-bridge, which is
    the ablation.

Read `status` out loud: a result produced through this module is a result about a
mechanism, not about an intrusion. Only a dataset that contains the same entity at
two stages - an emulated intrusion - can retire it.

Standard library only, like common/anchors.py: every kit imports it with no install
step.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "bridge.json"

# The one string every consumer must copy into its results. It is checked against
# the file on load, so a bridge.json that has quietly lost it does not load.
REQUIRED_STATUS = ("SYNTHETIC BINDING — stands in for the shared entity namespace "
                   "that only an emulated intrusion provides")

# The kinds a subject may be, kept identical to common/anchors.SUBJECT_KINDS. Imported
# lazily so bridge.py can be read on its own, and asserted in _selftest.
SUBJECT_KINDS = ("host", "session", "account", "domain", "path")

REQUIRED_FIELDS = ("from_kit", "from_subject_kind", "from_subject",
                   "to_kit", "to_subject_kind", "to_subject", "basis")


class BridgeError(ValueError):
    pass


@dataclass(frozen=True)
class Binding:
    """One declared identification of an emitter's subject with a consumer's subject.

    `basis` is not decoration. It is the sentence a reviewer reads to decide whether
    the binding is an assumption they will accept, so it must say plainly WHY the two
    identifiers are being treated as the same thing, and it is reproduced in the
    results of every run that used it.

    `to_valid_from` / `to_ttl_seconds` exist because the clocks do not match either.
    The corpora have no clock at all; the LANL-derived log counts seconds from its own
    start; the file-server runs count seconds from zero every run. When the consumer's
    clock is unrelated to the emitter's, the binding has to DECLARE the window in the
    consumer's own time base, and `to_time_basis` has to say where that declaration
    came from. A binding that leaves them null is saying the emitter's own timestamps
    are usable as they are.
    """
    from_kit: str
    from_subject_kind: str
    from_subject: str
    to_kit: str
    to_subject_kind: str
    to_subject: str
    basis: str
    link: str = ""
    to_valid_from: float | None = None
    to_ttl_seconds: float | None = None
    to_time_basis: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def key(self) -> tuple:
        return (self.from_kit, self.from_subject_kind, self.from_subject, self.to_kit)


class Bridge:
    """The loaded binding table, plus the status string every result must carry."""

    def __init__(self, status: str, bindings: list[Binding], path: Path | None = None,
                 enabled: bool = True, meta: dict | None = None):
        self.status = status
        self.bindings = list(bindings)
        self.path = path
        self.enabled = bool(enabled)
        self.meta = dict(meta or {})
        self._by_key: dict[tuple, Binding] = {}
        for b in self.bindings:
            if b.key() in self._by_key:
                raise BridgeError(
                    f"bridge.json binds {b.from_kit} {b.from_subject_kind} "
                    f"{b.from_subject!r} to {b.to_kit} twice. A subject may have one "
                    f"binding per consuming kit, or a run's results depend on dict order.")
            self._by_key[b.key()] = b

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Bridge":
        p = Path(path) if path is not None else DEFAULT_PATH
        if not p.exists():
            raise BridgeError(
                f"no entity binding at {p}. The kits share no entity namespace, so a "
                f"consumer cannot match an emitter's subject without one. Either write "
                f"it, or run with --no-bridge, which is the ablation.")
        doc = json.loads(p.read_text())
        status = doc.get("status", "")
        if REQUIRED_STATUS not in status:
            raise BridgeError(
                f"{p} does not carry the required status string. Every result derived "
                f"from a binding has to say what the binding is, so the string is not "
                f"optional. Expected to contain:\n  {REQUIRED_STATUS}\ngot:\n  {status!r}")
        out = []
        for i, e in enumerate(doc.get("bindings") or []):
            missing = [f for f in REQUIRED_FIELDS if not e.get(f)]
            if missing:
                raise BridgeError(f"binding {i} in {p} is missing {missing}")
            if e["from_subject_kind"] not in SUBJECT_KINDS:
                raise BridgeError(f"binding {i}: unknown from_subject_kind "
                                  f"{e['from_subject_kind']!r}")
            if e["to_subject_kind"] not in SUBJECT_KINDS:
                raise BridgeError(f"binding {i}: unknown to_subject_kind "
                                  f"{e['to_subject_kind']!r}")
            out.append(Binding(
                from_kit=e["from_kit"], from_subject_kind=e["from_subject_kind"],
                from_subject=e["from_subject"], to_kit=e["to_kit"],
                to_subject_kind=e["to_subject_kind"], to_subject=e["to_subject"],
                basis=e["basis"], link=e.get("link", ""),
                to_valid_from=e.get("to_valid_from"),
                to_ttl_seconds=e.get("to_ttl_seconds"),
                to_time_basis=e.get("to_time_basis", ""), note=e.get("note", "")))
        meta = {k: v for k, v in doc.items() if k not in ("status", "bindings")}
        return cls(status, out, path=p, enabled=True, meta=meta)

    @classmethod
    def disabled(cls) -> "Bridge":
        """The --no-bridge ablation: a bridge that binds nothing.

        Every kit must still run with this, and must produce exactly the numbers it
        produces with no anchor file at all, because without a binding no anchor can
        match anything.
        """
        return cls(REQUIRED_STATUS + " [DISABLED by --no-bridge: no subject is bound, "
                   "so no anchor can match and every kit falls back to unanchored]",
                   [], path=None, enabled=False)

    # ----------------------------------------------------------------- querying
    def resolve(self, from_kit: str, from_subject_kind: str, from_subject: str,
                to_kit: str) -> Binding | None:
        """The declared consumer-side subject for one emitter-side subject, or None."""
        if not self.enabled:
            return None
        return self._by_key.get((from_kit, from_subject_kind, from_subject, to_kit))

    def resolve_any(self, from_kit: str, candidates, to_kit: str) -> Binding | None:
        """First match over an ordered list of (kind, subject) candidates.

        The order is the caller's declared preference and must be stated where it is
        used: T1021 tries the composite host+account subject before the bare host, so a
        binding written for one account does not silently apply to every account on
        that host.
        """
        for kind, subj in candidates:
            b = self.resolve(from_kit, kind, subj, to_kit)
            if b is not None:
                return b
        return None

    def reverse(self, to_kit: str, to_subject_kind: str, to_subject: str,
                from_kit: str | None = None) -> list[Binding]:
        """Every binding that lands on this consumer-side subject.

        Needed by a consumer that iterates over its OWN subjects and has to find which
        emitter subject, if any, is bound to each - T1190 walking corpus files, for
        instance, which cannot enumerate decoy domains.
        """
        if not self.enabled:
            return []
        return [b for b in self.bindings
                if b.to_kit == to_kit and b.to_subject_kind == to_subject_kind
                and b.to_subject == to_subject
                and (from_kit is None or b.from_kit == from_kit)]

    def for_link(self, link: str) -> list[Binding]:
        return [b for b in self.bindings if b.link == link] if self.enabled else []

    # ------------------------------------------------------------------ reporting
    def result_block(self, link: str = "", anchor_file: str | None = None,
                     n_matched: int | None = None) -> dict:
        """What every consuming and emitting kit must write into its results.

        Hard requirement 5 in one call, so no kit can report a bridged number without
        the status string, whether the bridge was on, which anchor file it read and how
        many anchors matched.
        """
        return {
            "bridge_status": self.status,
            "bridge_used": self.enabled,
            "bridge_file": str(self.path) if self.path else None,
            "bridge_link": link or None,
            "bridge_bindings_for_link": [b.as_dict() for b in self.for_link(link)] if link else None,
            "bridge_n_bindings": len(self.bindings),
            "anchor_file": anchor_file,
            "anchors_matched": n_matched,
            "bridge_note": (
                "The detections behind these anchors are real detector output from the "
                "emitting kit. The identification of the emitter's subject with the "
                "consumer's subject is an ASSUMPTION declared in common/bridge.json, not "
                "an observation, because the two datasets share no entity namespace. Run "
                "with --no-bridge for the ablation."),
        }

    def summary(self) -> dict:
        by_link: dict[str, int] = {}
        for b in self.bindings:
            by_link[b.link or "-"] = by_link.get(b.link or "-", 0) + 1
        return {"status": self.status, "enabled": self.enabled,
                "file": str(self.path) if self.path else None,
                "n_bindings": len(self.bindings), "by_link": by_link}


def load(path: str | os.PathLike | None = None, *, no_bridge: bool = False) -> Bridge:
    """One call for every kit's argument handling: `bridge.load(a.bridge, no_bridge=a.no_bridge)`."""
    return Bridge.disabled() if no_bridge else Bridge.load(path)


def add_arguments(ap, *, link: str = "") -> None:
    """The two flags every kit in the chain must expose, worded the same everywhere."""
    ap.add_argument("--bridge", default=None,
                    help="path to the declared entity binding (default common/bridge.json). "
                         "The kits share no entity namespace, so the identification of one "
                         "kit's subject with another's is an assumption, and this file is "
                         "where it is written down.")
    ap.add_argument("--no-bridge", action="store_true",
                    help="THE ABLATION: run with the entity binding OFF. No subject is "
                         "bound, so no anchor can match, and the kit must produce exactly "
                         "the numbers it produces with no anchor file at all.")


# ---------------------------------------------------------------------- selftest
def _selftest() -> None:
    import tempfile
    try:
        import anchors
        assert tuple(anchors.SUBJECT_KINDS) == SUBJECT_KINDS, \
            "bridge.SUBJECT_KINDS has drifted from anchors.SUBJECT_KINDS"
    except ImportError:
        pass

    doc = {
        "status": "x " + REQUIRED_STATUS,
        "bindings": [
            {"link": "A2", "from_kit": "T1190", "from_subject_kind": "path",
             "from_subject": "shell_tennc/php", "to_kit": "T1021",
             "to_subject_kind": "host", "to_subject": "WEB01",
             "basis": "synthetic: the corpus has no host", "to_valid_from": 10,
             "to_ttl_seconds": 60, "to_time_basis": "declared"},
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bridge.json"
        p.write_text(json.dumps(doc))
        b = Bridge.load(p)
        assert b.enabled and len(b.bindings) == 1
        got = b.resolve("T1190", "path", "shell_tennc/php", "T1021")
        assert got is not None and got.to_subject == "WEB01"
        assert b.resolve("T1190", "path", "nope", "T1021") is None
        assert b.resolve("T1190", "host", "shell_tennc/php", "T1021") is None, \
            "the subject KIND is part of the key; a path must not resolve as a host"
        assert [x.to_subject for x in b.reverse("T1021", "host", "WEB01")] == ["WEB01"]
        assert b.for_link("A2") and not b.for_link("A3")
        rb = b.result_block("A2", anchor_file="anchors_links.jsonl", n_matched=3)
        assert REQUIRED_STATUS in rb["bridge_status"] and rb["bridge_used"] is True
        assert rb["anchors_matched"] == 3

        # a bridge missing the status string must not load
        bad = dict(doc, status="all good")
        q = Path(d) / "bad.json"
        q.write_text(json.dumps(bad))
        try:
            Bridge.load(q)
        except BridgeError:
            pass
        else:
            raise AssertionError("a bridge without the status string must be refused")

        # a duplicate binding must not load
        dup = dict(doc, bindings=doc["bindings"] * 2)
        r = Path(d) / "dup.json"
        r.write_text(json.dumps(dup))
        try:
            Bridge.load(r)
        except BridgeError:
            pass
        else:
            raise AssertionError("a duplicated binding must be refused")

    off = Bridge.disabled()
    assert not off.enabled
    assert off.resolve("T1190", "path", "shell_tennc/php", "T1021") is None
    assert off.resolve_any("T1190", [("path", "shell_tennc/php")], "T1021") is None
    assert off.reverse("T1021", "host", "WEB01") == []
    assert off.result_block("A2")["bridge_used"] is False
    assert REQUIRED_STATUS in off.result_block("A2")["bridge_status"]

    # the real file, if it is there
    if DEFAULT_PATH.exists():
        live = Bridge.load(DEFAULT_PATH)
        for lk in ("A1", "A2", "A3", "A4"):
            assert live.for_link(lk), f"common/bridge.json declares no binding for {lk}"
        print(f"bridge.py: common/bridge.json loads, {len(live.bindings)} bindings "
              f"{live.summary()['by_link']}")
    print("bridge.py: all self-tests passed")


if __name__ == "__main__":
    _selftest()
