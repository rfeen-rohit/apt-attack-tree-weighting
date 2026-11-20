"""Index the raw corpora, de-duplicate, and split by SOURCE GROUP (not by file).

Two things make this a leakage-controlled split, the protocol Kwon 2026 shows
matters:

  1. Files are grouped by provenance and whole groups go to one split. A group is
     a repository, or - for an aggregate collection such as tennc/webshell - one
     of its sub-collections, named in SPLIT_BY_SUBDIR below.
  2. Anything in calibration or test that is a near-duplicate of a training file
     is removed, because public web-shell collections repackage each other.
"""
import argparse, hashlib, json, random, sys
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, ROOT, CODE_EXT, relpath, bottom_k_sketch, lang_of, log, normalise_for_dedup,
                    read_text, read_jsonl, save_json, sha256_bytes, sketch_jaccard,
                    token_set, write_jsonl)

# Sources whose sub-directories are separate collections and must be split separately.
SPLIT_BY_SUBDIR = {"shell_tennc"}

# Explicit assignment of groups. Groups not listed are assigned deterministically
# (60/20/20 by a hash of the group name) and reported, so nothing is dropped silently.
DEFAULT_SPLIT = {
    "train": ["benign_wordpress",
              "shell_tennc/php", "shell_tennc/jsp", "shell_tennc/asp", "shell_tennc/aspx"],
    "calib": ["benign_joomla",
              "shell_tennc/web-malware-collection-13-06-2012", "shell_tennc/138shell",
              "shell_tennc/fuzzdb-webshell"],
    "test":  ["benign_drupal", "shell_johntroony",
              "shell_tennc/net-friend", "shell_tennc/caidao-shell", "shell_tennc/www-7jyewu-cn",
              "shell_tennc/xakep-shells", "shell_tennc/antSword-shells",
              "shell_tennc/Behinder", "shell_tennc/Godzilla"],
}
AUTO_WEIGHTS = [("train", 0.6), ("calib", 0.2), ("test", 0.2)]

# Labels are read from data/sources.json, never inferred from the directory name.
LABEL_CODE = {"benign": 0, "malicious": 1}


def load_labels():
    """source_id -> 0/1, from the manifest 01_fetch_corpus.py writes.

    Guessing the label from the directory name would silently call an unknown
    source benign, which is the one mistake this experiment cannot absorb.
    """
    p = DATA / "sources.json"
    if not p.exists():
        sys.exit(f"{p} is missing - run 01_fetch_corpus.py first")
    labels, odd = {}, []
    for m in json.loads(p.read_text()):
        lab = m.get("label")
        if lab in LABEL_CODE:
            labels[m["source_id"]] = LABEL_CODE[lab]
        elif lab == "reference":
            continue  # read-only reference repositories are not part of the corpus
        else:
            odd.append(f"{m.get('source_id')!r} (label {lab!r})")
    if odd:
        sys.exit("data/sources.json has entries whose label is neither benign nor "
                 "malicious: " + ", ".join(odd))
    return labels


def group_of(path: Path, src_dir: Path, source_id: str) -> str:
    if source_id not in SPLIT_BY_SUBDIR:
        return source_id
    rel = path.relative_to(src_dir).parts
    return f"{source_id}/{rel[0]}" if len(rel) > 1 else f"{source_id}/_root"


def auto_split(group: str) -> str:
    h = int(hashlib.blake2b(group.encode(), digest_size=8).hexdigest(), 16) / 2 ** 64
    acc = 0.0
    for name, w in AUTO_WEIGHTS:
        acc += w
        if h < acc:
            return name
    return AUTO_WEIGHTS[-1][0]


def index_files(raw: Path, labels, min_bytes, max_bytes):
    dirs = [p for p in sorted(raw.iterdir()) if p.is_dir() and not p.name.startswith("ref_")]
    unknown = [p.name for p in dirs if p.name not in labels]
    if unknown:
        sys.exit("these directories in data/raw have no entry in data/sources.json, "
                 "so their label is unknown: " + ", ".join(unknown) +
                 "\n  add them to data/sources.json with a label of benign or malicious "
                 "(or remove the directory); labels are never guessed from the name")
    rows = []
    for src_dir in dirs:
        source_id = src_dir.name
        label = labels[source_id]
        n = 0
        for p in src_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in CODE_EXT:
                continue
            try:
                b = p.read_bytes()
            except OSError:
                continue
            if not (min_bytes <= len(b) <= max_bytes):
                continue
            rows.append(dict(path=relpath(p), source_id=source_id,
                             group=group_of(p, src_dir, source_id), label=label,
                             lang=lang_of(p), size=len(b), sha256=sha256_bytes(b)))
            n += 1
        log(f"  {source_id:22s} {n:6d} candidate files")
    return rows


def sketch_of(row, cache):
    key = row["sha256"]
    if key not in cache:
        norm = normalise_for_dedup(read_text(row["path"]))
        cache[key] = (bottom_k_sketch(token_set(norm)), int(len(norm) ** 0.5 / 4))
    return cache[key]


def dedupe_within(rows, cache, jac, sample_cap, seed):
    seen, out, dropped_exact = set(), [], 0
    for r in rows:
        if r["sha256"] in seen:
            dropped_exact += 1
            continue
        seen.add(r["sha256"])
        out.append(r)
    log(f"  dropped {dropped_exact} exact duplicates")

    rng = random.Random(seed)
    by_group = defaultdict(list)
    for r in out:
        by_group[(r["group"], r["label"])].append(r)

    keep, dropped_near = [], 0
    for grp, items in by_group.items():
        if len(items) > sample_cap:
            rng.shuffle(items)
            log(f"  {grp[0]}: capped at {sample_cap} of {len(items)} files")
            items = items[:sample_cap]
        buckets = defaultdict(list)
        for r in items:
            sk, b = sketch_of(r, cache)
            if any(sketch_jaccard(sk, o) >= jac
                   for bb in (b - 1, b, b + 1) for o in buckets.get(bb, ())):
                dropped_near += 1
                continue
            buckets[b].append(sk)
            keep.append(r)
    log(f"  dropped {dropped_near} near-duplicates inside a group (Jaccard >= {jac})")
    return keep


def dedupe_against_train(rows, cache, jac):
    """Remove calibration/test files that repeat a training file."""
    train_buckets = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            sk, b = sketch_of(r, cache)
            train_buckets[b].append(sk)
    keep, dropped = [], Counter()
    for r in rows:
        if r["split"] == "train":
            keep.append(r)
            continue
        sk, b = sketch_of(r, cache)
        if any(sketch_jaccard(sk, o) >= jac
               for bb in (b - 1, b, b + 1) for o in train_buckets.get(bb, ())):
            dropped[r["split"]] += 1
            continue
        keep.append(r)
    log(f"  dropped {sum(dropped.values())} files that repeat a training file "
        f"({dict(dropped)})")
    return keep, dict(dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bytes", type=int, default=200)
    ap.add_argument("--max-bytes", type=int, default=300_000)
    ap.add_argument("--jaccard", type=float, default=0.85)
    ap.add_argument("--sample-cap", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-json", help="JSON file overriding the group assignment")
    a = ap.parse_args()

    raw = DATA / "raw"
    if not raw.exists():
        sys.exit("data/raw is missing - run 01_fetch_corpus.py first")

    split_plan = DEFAULT_SPLIT
    if a.split_json:
        split_plan = json.loads(Path(a.split_json).read_text())
        log(f"  using split from {a.split_json}")

    labels = load_labels()
    log("indexing")
    rows = index_files(raw, labels, a.min_bytes, a.max_bytes)
    if not rows:
        sys.exit("no candidate files found")
    log(f"  {len(rows)} files before de-duplication")

    cache = {}
    log("de-duplicating within groups")
    rows = dedupe_within(rows, cache, a.jaccard, a.sample_cap, a.seed)

    split_of = {g: s for s, gs in split_plan.items() for g in gs}
    groups = sorted({r["group"] for r in rows})
    auto = {}
    for g in groups:
        if g not in split_of:
            split_of[g] = auto_split(g)
            auto[g] = split_of[g]
    if auto:
        log(f"  {len(auto)} group(s) not named in the split plan, assigned by hash:")
        for g, s in sorted(auto.items()):
            log(f"      {g} -> {s}")
    rows = [dict(r, split=split_of[r["group"]]) for r in rows]

    log("removing calibration/test files that repeat a training file")
    rows, cross_dropped = dedupe_against_train(rows, cache, a.jaccard)

    write_jsonl(DATA / "index.jsonl", rows)
    counts = Counter((r["split"], r["label"]) for r in rows)
    summary = {}
    for split in ("train", "calib", "test"):
        summary[split] = dict(
            benign=counts[(split, 0)], malicious=counts[(split, 1)],
            groups=sorted({r["group"] for r in rows if r["split"] == split}))
    summary["config"] = vars(a)
    summary["path_style"] = "relative to the kit root"
    summary["auto_assigned_groups"] = auto
    summary["dropped_repeating_train"] = cross_dropped
    save_json(DATA / "split_summary.json", summary)

    log("")
    for split in ("train", "calib", "test"):
        d = summary[split]
        log(f"  {split:6s} benign={d['benign']:6d} malicious={d['malicious']:6d} "
            f"({len(d['groups'])} groups)")
        if d["benign"] == 0 or d["malicious"] == 0:
            log(f"  WARNING {split} has an empty class; the experiment needs both.")
    log("\nnext: python3 scripts/03_make_dilution_set.py")


if __name__ == "__main__":
    main()
