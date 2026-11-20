"""The anchor: context one stage's detector hands to the next.

Before this module an anchor was a bare string - a hostname in T1021, a session
id in T1486 - tested with `x in anchor_set`. The consuming detector could not
tell which stage had sent it, which technique produced it, when it was emitted,
how long it should be trusted, or how confident the emitter was. Three things
the paper claims were therefore not expressible:

  * a validity window, so an anchor applied to the whole log including traffic
    that predates the intrusion;
  * a confidence, so "merely suspected" could not be said - which is exactly
    what the wrong-origin control is trying to model;
  * an evidence pointer, so "scene recovery reconstructs the evidence chain"
    had nothing to reconstruct from.

An Anchor here is a typed, serialisable record with provenance. Emitting kits
append to a JSONL file; consuming kits open that file and ask a question of the
form "is there an anchor for this host, valid at this moment, at or above this
confidence?". Confidence and window are therefore parameters a sweep can vary,
which is what turns the anchor from a flag into something measurable.

Standard library only, on purpose: every kit must be able to import it with no
install step.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# --------------------------------------------------------------------- stages
# The lifecycle stages of Alshamrani et al.'s attack tree, as the paper uses them.
STAGES = {
    1: ("Reconnaissance",   "T1593"),
    2: ("Establish Foothold", "T1190"),
    3: ("Lateral Movement", "T1021"),
    4: ("Impact",           "T1486"),
    5: ("Cover-up",         "T1070"),
}

# The four links. A link is named by the stage that emits it.
LINKS = {"A1": (1, 2), "A2": (2, 3), "A3": (3, 4), "A4": (4, 5)}

SUBJECT_KINDS = ("host", "session", "account", "domain", "path")


class AnchorError(ValueError):
    pass


# --------------------------------------------------------------------- record
@dataclass(frozen=True)
class Anchor:
    """One piece of context passed from the detector at `stage` to the next.

    `subject` is what the next stage can match on; `subject_kind` says what sort
    of identifier it is, so a consumer never silently compares a host to a
    session (which the two kits previously did without noticing).

    `confidence` is the emitting detector's own belief, in [0, 1]. It is NOT a
    probability of compromise; it is whatever the emitting design defines, and
    each design must state its definition. What matters for the experiments is
    that it is ordered, so a consumer can be swept across thresholds.

    `valid_from` / `valid_until` are unix seconds. A consumer must not apply an
    anchor outside that window.
    """
    anchor_id: str
    link: str                  # "A1".."A4"
    stage: int                 # the stage that EMITTED it, 1..5
    technique: str             # the ATT&CK id of the emitting detector
    subject_kind: str
    subject: str
    emitted_at: float
    valid_from: float
    valid_until: float
    confidence: float
    evidence: dict = field(default_factory=dict)
    note: str = ""

    # ---------------------------------------------------------- construction
    @staticmethod
    def create(link: str, subject_kind: str, subject: str, emitted_at: float,
               ttl_seconds: float, confidence: float | None = None, *,
               operating_point: "OperatingPoint | None" = None,
               prior: float | None = None,
               evidence: dict | None = None,
               note: str = "", valid_from: float | None = None,
               anchor_id: str | None = None) -> "Anchor":
        """Either state `confidence` directly, or supply `operating_point` and
        `prior` and let it be COMPUTED by the shared rule (see below). The second
        form is the one the paper should use: it makes confidence a measured
        quantity rather than a number the author chose, and it records the inputs
        in `evidence` so a reader can recompute it."""
        if operating_point is not None:
            if prior is None:
                raise AnchorError("an operating point needs a prior to become a confidence")
            computed = confidence_from_operating_point(operating_point, prior)
            if confidence is not None and abs(confidence - computed) > 1e-9:
                raise AnchorError(
                    f"confidence {confidence} was given AND an operating point implying "
                    f"{computed:.4f}; supply one or the other, not both")
            confidence = computed
            evidence = dict(evidence or {})
            evidence.setdefault("operating_point", operating_point.as_dict())
            evidence.setdefault("prior", float(prior))
            evidence.setdefault("confidence_rule", "posterior from LR+ and prior")
        if confidence is None:
            raise AnchorError("supply confidence, or an operating point and a prior")
        if link not in LINKS:
            raise AnchorError(f"unknown link {link!r}; expected one of {sorted(LINKS)}")
        if subject_kind not in SUBJECT_KINDS:
            raise AnchorError(f"unknown subject_kind {subject_kind!r}")
        if not subject:
            raise AnchorError("subject must be a non-empty identifier")
        if not 0.0 <= confidence <= 1.0:
            raise AnchorError(f"confidence {confidence} outside [0,1]")
        if ttl_seconds <= 0:
            raise AnchorError("ttl_seconds must be positive; an anchor without a "
                              "window is the defect this module exists to remove")
        stage = LINKS[link][0]
        vf = emitted_at if valid_from is None else valid_from
        return Anchor(
            anchor_id=anchor_id or uuid.uuid4().hex,
            link=link, stage=stage, technique=STAGES[stage][1],
            subject_kind=subject_kind, subject=subject,
            emitted_at=float(emitted_at), valid_from=float(vf),
            valid_until=float(vf + ttl_seconds), confidence=float(confidence),
            evidence=dict(evidence or {}), note=note,
        )

    # ------------------------------------------------------------- behaviour
    def consumer_stage(self) -> int:
        """The stage this anchor is addressed to."""
        return LINKS[self.link][1]

    def covers(self, at: float) -> bool:
        return self.valid_from <= at <= self.valid_until

    def matches(self, subject_kind: str, subject: str) -> bool:
        return self.subject_kind == subject_kind and self.subject == subject

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_dict(d: dict) -> "Anchor":
        missing = {f for f in ("anchor_id", "link", "stage", "technique", "subject_kind",
                               "subject", "emitted_at", "valid_from", "valid_until",
                               "confidence")} - set(d)
        if missing:
            raise AnchorError(f"anchor record missing fields: {sorted(missing)}")
        if d["link"] not in LINKS:
            raise AnchorError(f"anchor record has unknown link {d['link']!r}")
        if d["subject_kind"] not in SUBJECT_KINDS:
            raise AnchorError(f"anchor record has unknown subject_kind {d['subject_kind']!r}")
        if not 0.0 <= float(d["confidence"]) <= 1.0:
            raise AnchorError(f"anchor record confidence {d['confidence']} outside [0,1]")
        return Anchor(
            anchor_id=d["anchor_id"], link=d["link"], stage=int(d["stage"]),
            technique=d["technique"], subject_kind=d["subject_kind"], subject=d["subject"],
            emitted_at=float(d["emitted_at"]), valid_from=float(d["valid_from"]),
            valid_until=float(d["valid_until"]), confidence=float(d["confidence"]),
            evidence=dict(d.get("evidence") or {}), note=d.get("note", ""),
        )


# ---------------------------------------------------------------------- store
class AnchorStore:
    """A set of anchors, optionally backed by a JSONL file.

    Emitting kit:   store = AnchorStore(path); store.emit(anchor)
    Consuming kit:  store = AnchorStore.load(path)
                    a = store.match("host", h, at=t, min_confidence=0.5, link="A2")
    """

    def __init__(self, path: str | os.PathLike | None = None,
                 anchors: Iterable[Anchor] = ()):
        self.path = Path(path) if path is not None else None
        self._anchors: list[Anchor] = list(anchors)

    # ------------------------------------------------------------- emitting
    def emit(self, anchor: Anchor) -> Anchor:
        self._anchors.append(anchor)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(anchor.to_json() + "\n")
        return anchor

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | os.PathLike, *, strict: bool = False) -> "AnchorStore":
        p = Path(path)
        out: list[Anchor] = []
        bad = 0
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Anchor.from_dict(json.loads(line)))
                except Exception:
                    bad += 1
                    if strict:
                        raise
        store = cls(p, out)
        store.malformed_lines = bad          # never silently zero: callers report it
        return store

    # ------------------------------------------------------------ querying
    def __len__(self) -> int:
        return len(self._anchors)

    def __iter__(self) -> Iterator[Anchor]:
        return iter(self._anchors)

    def candidates(self, subject_kind: str, subject: str, *, at: float | None = None,
                   min_confidence: float = 0.0, link: str | None = None,
                   from_stage: int | None = None) -> list[Anchor]:
        """Every anchor matching the filters, strongest confidence first."""
        if subject_kind not in SUBJECT_KINDS:
            raise AnchorError(f"unknown subject_kind {subject_kind!r}")
        out = [a for a in self._anchors
               if a.matches(subject_kind, subject)
               and a.confidence >= min_confidence
               and (link is None or a.link == link)
               and (from_stage is None or a.stage == from_stage)
               and (at is None or a.covers(at))]
        out.sort(key=lambda a: (-a.confidence, a.emitted_at))
        return out

    def match(self, subject_kind: str, subject: str, *, at: float | None = None,
              min_confidence: float = 0.0, link: str | None = None,
              from_stage: int | None = None) -> Anchor | None:
        """The single strongest applicable anchor, or None.

        Returning the record rather than a boolean is the point: the consumer can
        read `.confidence` to scale how much it relaxes, and `.stage` /
        `.technique` to treat an A2 anchor differently from an A3 one.
        """
        c = self.candidates(subject_kind, subject, at=at, min_confidence=min_confidence,
                            link=link, from_stage=from_stage)
        return c[0] if c else None

    def summary(self) -> dict:
        by_link: dict[str, int] = {}
        for a in self._anchors:
            by_link[a.link] = by_link.get(a.link, 0) + 1
        return {
            "n": len(self._anchors),
            "by_link": by_link,
            "subjects": sorted({(a.subject_kind, a.subject) for a in self._anchors})[:50],
            "confidence_min": min((a.confidence for a in self._anchors), default=None),
            "confidence_max": max((a.confidence for a in self._anchors), default=None),
            "malformed_lines": getattr(self, "malformed_lines", 0),
        }


# ===================================================================== confidence
# Confidence had no definition. Each design chose a number, so a 0.8 from the
# foothold detector and a 0.8 from the lateral detector meant nothing in common
# and could not be put on one axis. That is fatal for a framework whose whole
# claim is that stages hand each other context.
#
# The rule below is shared by every design; only its INPUTS are design-specific,
# and every input is something the design's own experiment already measures.
#
#   LR+        = TPR / FPR at the operating point that produced this detection
#   posterior  = (LR+ * prior) / (LR+ * prior + (1 - prior))
#
# So confidence is: "given this detector fired, at the operating point it was
# actually running at, and given a stated base rate of compromise for subjects of
# this kind, how likely is this subject compromised?" It is comparable across
# designs because the rule is the same and the prior is shared and swept.

@dataclass(frozen=True)
class OperatingPoint:
    """What a design measured about itself, at the threshold it actually ran at.

    `tpr` and `fpr` come from that design's own evaluation - the kits now persist
    per-item scores precisely so these can be read off at any threshold. `n_pos`
    and `n_neg` are the denominators, so a reader can see how much the estimate is
    worth, and so a zero FPR can be floored honestly rather than becoming infinite
    evidence.
    """
    tpr: float
    fpr: float
    n_pos: int = 0
    n_neg: int = 0
    threshold: float | None = None
    detector: str = ""

    def __post_init__(self):
        if not 0.0 <= self.tpr <= 1.0:
            raise AnchorError(f"tpr {self.tpr} outside [0,1]")
        if not 0.0 <= self.fpr <= 1.0:
            raise AnchorError(f"fpr {self.fpr} outside [0,1]")

    def effective_fpr(self) -> float:
        """A measured FPR of zero means 'below our resolution', not 'impossible'.

        With n_neg negatives and no false positive, the rule of three puts the 95%
        upper bound at 3/n_neg; we use that rather than 0, so a detector cannot
        claim infinite evidence from a finite test. With no denominator recorded we
        fall back to a conservative floor.
        """
        if self.fpr > 0:
            return self.fpr
        if self.n_neg <= 0:
            raise AnchorError(
                "a measured FPR of zero with NO negatives recorded is not evidence of "
                "anything. Record n_neg, or do not build an anchor from this detector.")
        return 3.0 / self.n_neg

    def is_floored(self) -> bool:
        """True when no false positive was observed, so the FPR used is 3/n_neg.

        3/n_neg is the rule of three's 95% UPPER limit on the FPR. LR+ and the
        confidence computed from it are therefore conservative 95% LOWER bounds: the
        true values are at least this large. The bound moves with how long the
        experiment ran rather than with how good the detector is, so callers must say
        "at least" and must not report it as a point estimate.
        """
        return self.fpr <= 0

    def lr_plus(self) -> float:
        lr = max(self.tpr, 1e-9) / self.effective_fpr()
        if lr <= 1.0:
            raise AnchorError(
                f"this detector has LR+ = {lr:.4g}, at or below 1: firing is no more "
                f"likely when the subject IS compromised than when it is not, so it is "
                f"not evidence and must not become an anchor (tpr={self.tpr}, "
                f"fpr={self.fpr}, n_pos={self.n_pos}, n_neg={self.n_neg}).")
        return lr

    def as_dict(self) -> dict:
        try:
            lr, lr_err = self.lr_plus(), None
        except AnchorError as e:
            lr, lr_err = None, str(e)
        return {"tpr": self.tpr, "fpr": self.fpr, "effective_fpr": self.effective_fpr(),
                "lr_plus": lr, "lr_plus_refused": lr_err,
                "fpr_is_floored": self.is_floored(),
                "confidence_is_lower_bound": self.is_floored(),
                "n_pos": self.n_pos, "n_neg": self.n_neg,
                "threshold": self.threshold, "detector": self.detector}

    # ------------------------------------------- the hand-off between kits
    # Each kit writes ONE of these at the end of its experiment. The next kit
    # reads it to build its anchors, so a stage's confidence is derived from the
    # previous stage's MEASURED performance rather than from a flag. This file is
    # the entire coupling between the kits: no imports, no API, one small JSON.
    STANDARD_NAME = "operating_point.json"

    def save(self, path: str | os.PathLike) -> "OperatingPoint":
        q = Path(path)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))
        return self

    @staticmethod
    def load(path: str | os.PathLike) -> "OperatingPoint":
        q = Path(path)
        if not q.exists():
            raise AnchorError(
                f"no operating point at {q}. The emitting kit must run first and write "
                f"one; without it this kit would have to invent a confidence, which is "
                f"the defect this module exists to remove.")
        d = json.loads(q.read_text())
        missing = {"tpr", "fpr"} - set(d)
        if missing:
            raise AnchorError(f"operating point at {q} is missing {sorted(missing)}")
        return OperatingPoint(tpr=float(d["tpr"]), fpr=float(d["fpr"]),
                              n_pos=int(d.get("n_pos", 0)), n_neg=int(d.get("n_neg", 0)),
                              threshold=d.get("threshold"), detector=d.get("detector", ""))


def confidence_from_operating_point(op: OperatingPoint, prior: float,
                                    *, cap: float = 0.99) -> float:
    """The shared confidence rule. Capped, because no single detector should be
    allowed to assert near-certainty on its own."""
    if not 0.0 < prior < 1.0:
        raise AnchorError(f"prior {prior} must be strictly between 0 and 1")
    lr = op.lr_plus()
    post = (lr * prior) / (lr * prior + (1.0 - prior))
    return float(min(post, cap))


# Base rates to sweep. A paper reporting one prior is reporting one opinion; the
# honest form is a curve. These are defaults, not truths - each design states the
# population its prior refers to.
PRIOR_GRID = (0.0001, 0.001, 0.01, 0.05, 0.10)


# ====================================================================== decay
# A hard cutoff at valid_until says an anchor is fully trusted one second before
# it expires and worthless one second after. Decay says what is actually meant:
# evidence ages. The consumer gets a confidence that is a function of WHEN it asks.

def decayed_confidence(anchor: "Anchor", at: float, *,
                       half_life: float | None = None, floor: float = 0.0) -> float:
    """Confidence at time `at`. Outside the window it is zero.

    Inside, it halves every `half_life` seconds since emission. The default
    half-life is a quarter of the anchor's own window, so an anchor is worth about
    an eighth of its emitted confidence by the time it expires.
    """
    if not anchor.covers(at):
        return 0.0
    span = max(anchor.valid_until - anchor.valid_from, 1e-9)
    hl = half_life if half_life is not None else span / 4.0
    age = max(at - anchor.emitted_at, 0.0)
    return max(anchor.confidence * (0.5 ** (age / hl)), floor)


# ============================================================ consumer policy
# stage and technique were carried but never consulted, so a reviewer could fairly
# say they were decoration. This table is what makes them decision-relevant: the
# consumer looks up what it is ALLOWED to do with an anchor of a given link, and
# different links license different relaxations. It is data, not code, so the
# paper can print it and an ablation can switch a row off.
#
#   relax     - the named step in the consuming design that the anchor may skip
#   max_drop  - the largest fraction of that design's threshold the anchor may
#               remove, at confidence 1.0; the actual drop is max_drop * confidence
#   requires  - the subject_kind the consumer must match on

CONSUMER_POLICY = {
    "A1": {"consumer_stage": 2, "requires": "domain",
           "relax": "raise priority of requests from a harvesting source",
           "max_drop": 0.25,
           "why": "a decoy use proves harvesting, not exploitation; it reprioritises only"},
    "A2": {"consumer_stage": 3, "requires": "host",
           "relax": "route paths from this origin to the rules path instead of the budget",
           "max_drop": 1.00,
           "why": "Hopper's budget is the thing the origin is meant to bypass"},
    "A3": {"consumer_stage": 4, "requires": "session",
           "relax": "accept the behavioural flag without the content confirmation",
           "max_drop": 1.00,
           "why": "stage 3 has already established suspicion for this session"},
    # ---- the A4 subject-kind defect, and which way it was fixed.
    # This row required subject_kind "host" while stage 4's own keys are session ids of
    # the form "WKS-199\\svc_backup", so no T1486 anchor could ever satisfy it and the
    # link was unimplementable as written. Two fixes were available: weaken this row to
    # "session", or have the EMITTER project its session key onto the host namespace.
    #
    # The emitter projects. T1070 has no notion of a session at all: its unit of analysis
    # is one host's event log, compared record for record against the collector's copy of
    # THAT HOST's log. A policy demanding a session would name an identifier the consumer
    # cannot match on, which is the kind-confusion this module exists to remove, and it
    # would be weakening the contract to fit the first emitter that failed it. The host
    # part of a "HOST\\account" session key is a real host identifier in T1486's own
    # dataset, so the projection throws information away rather than inventing any:
    # T1486_sessions/scripts/01_run_experiment.py splits on the last backslash, emits
    # subject_kind "host", and records the full session key in the anchor's evidence so
    # nothing is lost. The identification of that workstation with the host whose log
    # T1070 reads is a separate assumption, declared in common/bridge.json.
    "A4": {"consumer_stage": 5, "requires": "host",
           "relax": "lower the threshold for treating a missing record as removal",
           "max_drop": 0.50,
           "why": "impact gives a time window, not proof the log was touched",
           "emitter_projection": ("T1486 keys on HOST\\account sessions; it splits the "
                                  "host out and emits subject_kind 'host', because stage "
                                  "5's unit of analysis is a host's log and it has no "
                                  "session to match on. The full session key travels in "
                                  "the anchor's evidence.")},
}


def policy_for(anchor: "Anchor") -> dict:
    return CONSUMER_POLICY[anchor.link]


def relaxation(anchor: "Anchor", at: float, *, half_life: float | None = None) -> float:
    """How much of the consuming design's threshold this anchor removes, now.

    Zero means the consumer behaves exactly as if unanchored - which is the
    fallback every design in the paper promises.
    """
    c = decayed_confidence(anchor, at, half_life=half_life)
    return policy_for(anchor)["max_drop"] * c


# ------------------------------------------------------------- sweep helper
def confidence_grid(lo: float = 0.0, hi: float = 1.0, steps: int = 11) -> list[float]:
    """The thresholds a consuming experiment should sweep.

    The cost of anchoring is a curve over confidence, not a single number. Every
    kit that consumes anchors should report its results at each of these.
    """
    if steps < 2:
        raise AnchorError("steps must be at least 2")
    return [round(lo + (hi - lo) * i / (steps - 1), 6) for i in range(steps)]


# -------------------------------------------------------------------- selftest
def _selftest() -> None:
    import tempfile

    t0 = 1_700_000_000.0
    a = Anchor.create("A2", "host", "WEB01", emitted_at=t0, ttl_seconds=3600,
                      confidence=0.8, evidence={"rule": "webshell-write", "score": 0.93},
                      note="foothold detector")
    assert a.stage == 2 and a.technique == "T1190" and a.consumer_stage() == 3
    assert a.covers(t0) and a.covers(t0 + 3600) and not a.covers(t0 + 3601)
    assert not a.covers(t0 - 1), "an anchor must not apply before it was emitted"

    # round trip
    b = Anchor.from_dict(json.loads(a.to_json()))
    assert b == a

    # validation
    for bad in (
        lambda: Anchor.create("A9", "host", "X", t0, 60, 0.5),
        lambda: Anchor.create("A2", "potato", "X", t0, 60, 0.5),
        lambda: Anchor.create("A2", "host", "", t0, 60, 0.5),
        lambda: Anchor.create("A2", "host", "X", t0, 60, 1.5),
        lambda: Anchor.create("A2", "host", "X", t0, 0, 0.5),
    ):
        try:
            bad()
        except AnchorError:
            pass
        else:
            raise AssertionError("validation did not fire")

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "anchors.jsonl"
        s = AnchorStore(p)
        s.emit(a)
        s.emit(Anchor.create("A2", "host", "WEB01", emitted_at=t0 + 10, ttl_seconds=60,
                             confidence=0.4))
        s.emit(Anchor.create("A3", "session", "WKS-199\\svc", emitted_at=t0,
                             ttl_seconds=600, confidence=0.9))
        r = AnchorStore.load(p)
        assert len(r) == 3 and r.malformed_lines == 0

        # kind is honoured: a host query never returns the session anchor
        assert r.match("host", "WKS-199\\svc") is None
        assert r.match("session", "WKS-199\\svc").link == "A3"

        # strongest first
        m = r.match("host", "WEB01", at=t0 + 20)
        assert m.confidence == 0.8

        # confidence filter
        assert r.match("host", "WEB01", at=t0 + 20, min_confidence=0.9) is None

        # window is enforced
        assert r.match("host", "WEB01", at=t0 + 7200) is None
        # the weaker, shorter anchor is the only one alive in its own window
        assert len(r.candidates("host", "WEB01", at=t0 + 20)) == 2

        # link filter
        assert r.match("host", "WEB01", at=t0, link="A3") is None

        # malformed lines are counted, not hidden
        with p.open("a") as f:
            f.write("{not json\n")
        assert AnchorStore.load(p).malformed_lines == 1

    g = confidence_grid(steps=11)
    assert g[0] == 0.0 and g[-1] == 1.0 and len(g) == 11

    # ---------------------------------------------------- confidence from a measurement
    op = OperatingPoint(tpr=0.90, fpr=0.01, n_pos=100, n_neg=1000,
                        threshold=0.73, detector="T1190 max-rule")
    assert abs(op.lr_plus() - 90.0) < 1e-6
    c01 = confidence_from_operating_point(op, 0.01)
    c10 = confidence_from_operating_point(op, 0.10)
    assert 0 < c01 < c10 < 1, "a higher prior must give a higher posterior"
    assert abs(c01 - (90 * .01) / (90 * .01 + .99)) < 1e-9

    # a measured FPR of zero must not become infinite evidence
    z = OperatingPoint(tpr=1.0, fpr=0.0, n_pos=50, n_neg=600)
    assert abs(z.effective_fpr() - 3 / 600) < 1e-12
    assert confidence_from_operating_point(z, 0.01) <= 0.99

    # computing beats asserting, and the inputs are recorded
    a2 = Anchor.create("A2", "host", "WEB01", emitted_at=t0, ttl_seconds=3600,
                       operating_point=op, prior=0.01)
    assert abs(a2.confidence - c01) < 1e-12
    assert a2.evidence["operating_point"]["lr_plus"] == op.lr_plus()
    assert a2.evidence["prior"] == 0.01
    try:
        Anchor.create("A2", "host", "X", t0, 60, 0.5, operating_point=op, prior=0.01)
    except AnchorError:
        pass
    else:
        raise AssertionError("stating confidence AND an operating point must be refused")

    # ---------------------------------------------------------------------- decay
    d0 = decayed_confidence(a2, t0)
    dm = decayed_confidence(a2, t0 + 900)          # one default half-life
    de = decayed_confidence(a2, t0 + 3600)
    assert abs(d0 - a2.confidence) < 1e-12
    assert abs(dm - a2.confidence / 2) < 1e-9, "default half-life is a quarter of the window"
    assert de < dm and de > 0
    assert decayed_confidence(a2, t0 + 3601) == 0.0, "outside the window it is worth nothing"

    # --------------------------------------------------------------------- policy
    pol = policy_for(a2)
    assert pol["consumer_stage"] == 3 and pol["requires"] == "host"
    assert relaxation(a2, t0) > relaxation(a2, t0 + 900) > 0
    a1 = Anchor.create("A1", "domain", "decoy.example", emitted_at=t0,
                       ttl_seconds=86400, confidence=1.0)
    assert relaxation(a1, t0) == 0.25, "A1 may only reprioritise, never bypass"
    assert set(CONSUMER_POLICY) == set(LINKS), "every link needs a stated policy"
    for lk, pdef in CONSUMER_POLICY.items():
        assert pdef["consumer_stage"] == LINKS[lk][1]
        assert pdef["requires"] in SUBJECT_KINDS
        assert 0.0 < pdef["max_drop"] <= 1.0

    print("anchors.py: all self-tests passed")


if __name__ == "__main__":
    _selftest()
