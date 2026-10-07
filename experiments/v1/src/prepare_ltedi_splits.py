#!/usr/bin/env python3
"""Prepare private, group-disjoint LT-EDI splits without manual review or GPU.

Requires audit_ltedi_images.py and download_ltedi_images.py in this directory.
Original dev is reserved as final test, not used for training/model selection.
User spot-checks are recorded honestly; politics-free status is NOT certified.
No source file, original label, image or prior output is changed/deleted.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from audit_ltedi_images import decode_record
from download_ltedi_images import COMMIT, DEFAULT_REPO, file_blob_sha1, image_path, read_specs


DEFAULT_OUTPUT = "/root/autodl-tmp/social_risk_jev/data/processed/ltedi_grouped_v1"
LABEL_MAP = {"Not-Misogyny": 0, "Misogyny": 1}


def normalize_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


class Components:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_groups(rows, near_distance=4):
    """Candidate similarities are conservative split guards, not proven duplicates."""
    graph = Components(row["id"] for row in rows)
    by_pixels, by_text = defaultdict(list), defaultdict(list)
    for row in rows:
        by_pixels[row["pixel_sha256"]].append(row["id"])
        by_text[normalize_text(row["text"])].append(row["id"])
    for groups in (by_pixels, by_text):
        for ids in groups.values():
            for key in ids[1:]:
                graph.union(ids[0], key)

    near_pairs = cross_pairs = 0
    for a, b in combinations(rows, 2):
        if a["pixel_sha256"] == b["pixel_sha256"]:
            continue
        aw, ah = a["size"]
        bw, bh = b["size"]
        if abs((aw / ah) / (bw / bh) - 1) > .02:
            continue
        if (a["dhash"] ^ b["dhash"]).bit_count() <= near_distance:
            graph.union(a["id"], b["id"])
            near_pairs += 1
            cross_pairs += a["original_split"] != b["original_split"]

    members = defaultdict(list)
    for row in rows:
        members[graph.find(row["id"])].append(row["id"])
    group_ids = {
        root: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:20]
        for root, ids in members.items()
    }
    assigned = [{**row, "group_id": group_ids[graph.find(row["id"])]} for row in rows]
    summary = {
        "exact_image_groups": sum(len(ids) > 1 for ids in by_pixels.values()),
        "identical_normalized_text_groups": sum(len(ids) > 1 for ids in by_text.values()),
        "near_image_candidate_pairs": near_pairs, "cross_original_split_near_pairs": cross_pairs,
        "component_count": len(members), "largest_component_rows": max(map(len, members.values()), default=0),
        "near_distance": near_distance, "aspect_ratio_relative_tolerance": .02,
        "similarity_is_not_a_duplicate_or_harm_label": True,
    }
    return assigned, summary


def select_validation(pool, fraction, seed):
    """Deterministic grouped random search to approximate class-stratified proportions."""
    groups = defaultdict(list)
    for row in pool:
        groups[row["group_id"]].append(row)
    keys = sorted(groups)
    totals = Counter(row["label"] for row in pool)
    if set(totals) != {0, 1} or min(totals.values()) < 2:
        raise RuntimeError("Insufficient labels remaining after leakage guards; inspect preparation report")
    vectors = {key: Counter(row["label"] for row in groups[key]) for key in keys}
    rng = random.Random(seed)
    best = None
    for _ in range(2048):
        chosen = {key for key in keys if rng.random() < fraction}
        counts = Counter()
        for key in chosen:
            counts.update(vectors[key])
        if any(not 0 < counts[label] < totals[label] for label in (0, 1)):
            continue
        score = sum(((counts[label] - fraction * totals[label]) / max(1, fraction * totals[label])) ** 2
                    for label in (0, 1))
        if best is None or score < best[0]:
            best = (score, chosen)
    if best is None:
        raise RuntimeError("Cannot make two-class group-disjoint train/validation split; do not fall back to row shuffling")
    val_groups = best[1]
    return ([row for row in pool if row["group_id"] not in val_groups],
            [row for row in pool if row["group_id"] in val_groups])


def make_splits(rows, val_fraction=.15, seed=2060, near_distance=4):
    rows, grouping = assign_groups(rows, near_distance)
    by_input = defaultdict(list)
    for row in rows:
        by_input[(row["pixel_sha256"], normalize_text(row["text"]))].append(row)
    # Prefer official dev when the entire image+text input is identical. Never
    # silently deduplicate different transcriptions or change conflicting labels.
    retained, exclusions = [], []
    for group in by_input.values():
        if len({row["label"] for row in group}) > 1:
            raise RuntimeError(f"IDENTICAL_INPUT_LABEL_CONFLICT_STOP: {[row['id'] for row in group]}")
        ordered = sorted(group, key=lambda row: (row["original_split"] != "dev", row["id"]))
        retained.append(ordered[0])
        for row in ordered[1:]:
            exclusions.append({"id": row["id"], "reason": "exact_image_and_normalized_text_duplicate",
                               "representative": ordered[0]["id"]})
    retained.sort(key=lambda row: row["id"])
    test = [row for row in retained if row["original_split"] == "dev"]
    test_groups = {row["group_id"] for row in test}
    pool = []
    for row in retained:
        if row["original_split"] != "train":
            continue
        if row["group_id"] in test_groups:
            exclusions.append({"id": row["id"], "reason": "conservative_group_overlap_with_heldout_dev"})
        else:
            pool.append(row)
    train, val = select_validation(pool, val_fraction, seed)
    splits = {"train": train, "val": val, "test": test}
    for a, b in combinations(splits, 2):
        if {row["group_id"] for row in splits[a]} & {row["group_id"] for row in splits[b]}:
            raise RuntimeError(f"GROUP_LEAKAGE_STOP: {a}/{b}")
    if len({row["id"] for split in splits.values() for row in split}) != sum(map(len, splits.values())):
        raise RuntimeError("Repeated IDs across splits")
    return splits, grouping, sorted(exclusions, key=lambda row: row["id"])


def collect_source_rows(repo):
    specs = read_specs(repo)  # Validate pinned source Git tree and unchanged CSVs.
    annotations = {}
    annotation_hashes = {}
    for split in ("train", "dev"):
        csv_path = repo / f"Datasets/{split}/{split}.csv"
        annotation_hashes[split] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                annotations[f"{split}:{row['image_name'].strip()}"] = row
    rows, quality_exclusions = [], []
    for index, spec in enumerate(specs, 1):
        key = f"{spec.split}:{Path(spec.relative).name}"
        original = annotations[key]
        path = image_path(repo, spec)
        if file_blob_sha1(path) != spec.sha1:
            raise RuntimeError(f"SOURCE_BYTES_CHANGED_STOP: {key}; preparation does not repair source files")
        try:
            record = decode_record(path, key, spec.split, original["labels"].strip())
        except Exception as exc:
            quality_exclusions.append({"id": key, "reason": "unsupported_or_unreadable_image",
                                       "detail": f"{type(exc).__name__}: {exc}"[:240]})
        else:
            rows.append({"id": key, "original_split": spec.split, "image": str(path.resolve()),
                         "text": original["transcriptions"], "label": LABEL_MAP[record.label],
                         "pixel_sha256": record.pixel_sha256, "file_sha256": record.file_sha256,
                         "dhash": record.dhash, "size": record.size})
        if index % 200 == 0 or index == len(specs):
            print(f"PREPARING={index}/{len(specs)} READABLE={len(rows)}", flush=True)
    return rows, quality_exclusions, annotation_hashes


def prepare(repo, output, val_fraction=.15, seed=2060, user_spot_check=False, dry_run=False):
    repo, output = repo.resolve(), output.resolve()
    if not user_spot_check:
        raise RuntimeError("Use --accept-user-spot-check to record the requested waiver of full political review")
    if output.exists():
        raise RuntimeError(f"OUTPUT_EXISTS_STOP: {output}; existing outputs are never overwritten")
    if output.is_relative_to(repo) or repo.is_relative_to(output):
        raise RuntimeError("Output must be separate from source repository")
    rows, quality_exclusions, annotation_hashes = collect_source_rows(repo)
    splits, grouping, exclusions = make_splits(rows, val_fraction, seed)
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": COMMIT, "source_repo": str(repo), "source_annotation_sha256": annotation_hashes,
        "seed": seed, "requested_validation_fraction": val_fraction, "label_map": LABEL_MAP,
        "political_screening": {"method": "user_reported_spot_check", "sample_count": "not_recorded",
                                "full_dataset_verified": False, "manual_full_review_waived_by_user": True},
        "scope": "Chinese misogyny meme classification; not a general fraud or all-risk benchmark",
        "grouping": grouping, "splits": {
            name: {"rows": len(items), "groups": len({row["group_id"] for row in items}),
                   "label_counts": dict(sorted(Counter(row["label"] for row in items).items()))}
            for name, items in splits.items()},
        "quality_exclusions": quality_exclusions, "deduplication_and_leakage_guard_exclusions": exclusions,
        "source_readable_rows": len(rows), "group_overlap_between_splits": 0,
        "source_files_modified_or_deleted": False, "raw_data_uploaded": False,
        "test_policy": "Original dev retained (exact input duplicates collapsed); final evaluation only, no tuning.",
        "model_input_policy": "Only image and text are inputs. label is supervision/evaluation, never part of inference prompts.",
        "risk_band_policy": "Source labels are binary. Low/medium/high operational bands are not annotated severity labels.",
        "notes": [
            "dHash/identical-text connected components are conservative split guards, not confirmed semantic duplicates.",
            "Rows overlapping the test component are excluded only from new train/validation manifests, not deleted.",
            "Group disjointness applies to the specified heuristics, not all conceivable semantic near-duplicates.",
            "User spot-checks do not establish that every image/text is free of political content.",
        ],
    }
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
        for name, items in splits.items():
            with (output / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
                for row in items:
                    item = {key: row[key] for key in ("id", "image", "text", "label", "group_id")}
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        with (output / "preparation_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    for name, counts in report["splits"].items():
        print(f"{name.upper()} N={counts['rows']} GROUPS={counts['groups']} LABELS={counts['label_counts']}", flush=True)
    print(f"QUALITY_EXCLUDED={len(quality_exclusions)} DEDUP_OR_GUARD_EXCLUDED={len(exclusions)} GROUP_OVERLAP=0", flush=True)
    print("POLITICAL_STATUS=USER_SPOT_CHECK_ONLY; NOT_FULLY_VERIFIED\nSOURCE_FILES_UNCHANGED; NO_GPU_OR_CLOUD_UPLOAD", flush=True)
    print("DRY_RUN; NO_OUTPUT_WRITTEN" if dry_run else f"DATA_PREPARED={output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(DEFAULT_REPO))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--val-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument("--accept-user-spot-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not .05 <= args.val_fraction <= .4:
        parser.error("--val-fraction must be between 0.05 and 0.4")
    prepare(args.repo, args.output, args.val_fraction, args.seed, args.accept_user_spot_check, args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PREPARATION_STOPPED {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(2)
