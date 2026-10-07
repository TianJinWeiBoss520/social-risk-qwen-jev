#!/usr/bin/env python3
"""Read-only image quality/duplicate audit. No data is changed or uploaded.

dHash pairs are review candidates, never automatic deduplication decisions.
This script does NOT check political content or establish dataset eligibility.
"""

import argparse
import csv
import hashlib
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_REPO = "/root/autodl-tmp/social_risk_jev/data/raw/ltedi_cn_misogyny"
EXPECTED = {"train": 1190, "dev": 170}
ALLOWED_LABELS = {"Misogyny", "Not-Misogyny"}


@dataclass(frozen=True)
class ImageRecord:
    key: str
    split: str
    label: str
    size: tuple
    file_sha256: str
    pixel_sha256: str
    dhash: int


def difference_hash(image):
    samples = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    return sum(
        int(samples[y * 9 + x] > samples[y * 9 + x + 1]) << (y * 8 + x)
        for y in range(8) for x in range(8)
    )


def decode_record(path, key, split, label):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Multi-frame image requires separate review")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            size = image.size
            prefix = f"RGB:{size[0]}x{size[1]}\0".encode()
            pixel_digest = hashlib.sha256(prefix + image.tobytes()).hexdigest()
            dhash = difference_hash(image)
    with path.open("rb") as stream:
        file_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return ImageRecord(key, split, label, size, file_digest, pixel_digest, dhash)


def duplicate_summary(records, attribute, examples):
    groups = defaultdict(list)
    for record in records:
        groups[getattr(record, attribute)].append(record)
    repeated = [group for group in groups.values() if len(group) > 1]
    cross_split = [group for group in repeated if len({r.split for r in group}) > 1]
    conflicts = [group for group in repeated if len({r.label for r in group}) > 1]
    return {
        "groups": len(repeated),
        "rows_in_groups": sum(len(group) for group in repeated),
        "cross_split_groups": len(cross_split),
        "mixed_label_groups_for_review": len(conflicts),
        "examples": [[r.key for r in group] for group in repeated[:examples]],
        "cross_split_examples": [[r.key for r in group] for group in cross_split[:examples]],
        "mixed_label_examples": [[r.key for r in group] for group in conflicts[:examples]],
    }


def audit(repo, distance=4, examples=3, progress=False):
    repo = repo.resolve()
    report = {"read_only": True, "splits": {}, "unreadable_or_unsupported_count": 0}
    records, failures = [], []
    for split, expected in EXPECTED.items():
        csv_path = repo / f"Datasets/{split}/{split}.csv"
        if csv_path.is_symlink() or not csv_path.resolve().is_relative_to(repo):
            raise RuntimeError(f"Unsafe annotation path: {csv_path}")
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"image_name", "labels", "transcriptions"}.issubset(reader.fieldnames or []):
                raise RuntimeError(f"Unexpected CSV fields: {split}")
            rows = list(reader)
        if len(rows) != expected:
            raise RuntimeError(f"Unexpected annotation count: {split}={len(rows)}")
        labels, names, readable, sizes = Counter(), set(), 0, []
        for index, row in enumerate(rows, 1):
            name, label = row["image_name"].strip(), row["labels"].strip()
            if not name or Path(name).name != name or "\\" in name or name in names:
                raise RuntimeError(f"Unsafe or repeated name: {split}:{name}")
            if label not in ALLOWED_LABELS or not row["transcriptions"].strip():
                raise RuntimeError(f"Invalid original annotation: {split}:{name}")
            names.add(name)
            labels[label] += 1
            key = f"{split}:{name}"
            path = repo / f"Datasets/{split}/{split} images/{name}"
            if path.is_symlink() or not path.resolve().is_relative_to(repo):
                raise RuntimeError(f"Unsafe image path: {key}")
            try:
                record = decode_record(path, key, split, label)
                records.append(record)
                readable += 1
                sizes.append(record.size)
            except Exception as exc:
                failures.append({"id": key, "error": type(exc).__name__, "detail": str(exc)[:200]})
            if progress and (index % 200 == 0 or index == expected):
                print(f"AUDITING {split.upper()}={index}/{expected} READABLE={readable}", flush=True)
        report["splits"][split] = {
            "annotations": len(rows), "readable": readable, "labels": dict(labels),
            "smallest_image_by_area": min(sizes, key=lambda s: s[0] * s[1]) if sizes else None,
            "largest_image_by_area": max(sizes, key=lambda s: s[0] * s[1]) if sizes else None,
        }
    report["unreadable_or_unsupported_count"] = len(failures)
    report["unreadable_or_unsupported_examples"] = failures[:10]
    report["exact_file_duplicates"] = duplicate_summary(records, "file_sha256", examples)
    report["exact_pixel_duplicates"] = duplicate_summary(records, "pixel_sha256", examples)
    near_count = cross_count = 0
    near_examples, cross_examples = [], []
    for a, b in combinations(records, 2):
        if a.pixel_sha256 == b.pixel_sha256:
            continue  # Report exact duplicates separately.
        aspect_a, aspect_b = a.size[0] / a.size[1], b.size[0] / b.size[1]
        if abs(aspect_a / aspect_b - 1) > 0.02:
            continue
        difference = (a.dhash ^ b.dhash).bit_count()
        if difference > distance:
            continue
        pair = {"ids": [a.key, b.key], "dhash_distance": difference}
        near_count += 1
        if len(near_examples) < examples:
            near_examples.append(pair)
        if a.split != b.split:
            cross_count += 1
            if len(cross_examples) < examples:
                cross_examples.append(pair)
    report["near_duplicate_candidates"] = {
        "method": "64-bit dHash; aspect-ratio difference <=2%",
        "distance_threshold": distance, "pairs": near_count, "cross_split_pairs": cross_count,
        "examples": near_examples, "cross_split_examples": cross_examples,
        "automatic_removal": False,
        "caution": "Similar backgrounds with different meme text can be false positives; manual review required.",
    }
    report["readability_check_passed"] = not failures
    report["political_content_screening"] = "NOT_DONE; text and image review required"
    report["data_split_changes"] = "NONE"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(DEFAULT_REPO))
    parser.add_argument("--near-distance", type=int, choices=range(0, 9), default=4)
    parser.add_argument("--examples", type=int, choices=range(0, 11), default=3)
    args = parser.parse_args()
    print("READ_ONLY_AUDIT; no files changed or uploaded; no GPU required", flush=True)
    report = audit(args.repo, args.near_distance, args.examples, progress=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("IMAGE_READABILITY_CHECK_OK" if report["readability_check_passed"] else "IMAGE_READABILITY_CHECK_FAILED", flush=True)
    print("POLITICAL_SCREENING_NOT_DONE; do not start training yet", flush=True)
    return 0 if report["readability_check_passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AUDIT_STOPPED {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(2)
