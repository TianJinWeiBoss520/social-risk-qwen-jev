#!/usr/bin/env python3
"""Privately export existing LT-EDI manifests and cached LoRA+Jev errors.

Standard library only: no model, key lookup, API, download or training. The
saved request bodies, not newly inferred explanations, are exported. This is
a data-export utility, not another experiment or permission to redistribute.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from pathlib import Path

VERSION = "ltedi_private_data_badcases_v1"
COMMIT = "0179687f197a0b2babddcb4eb32547a07cba0933"
MODEL = "jev-1.13.0"
THRESHOLD = 0.40
QUESTION_SHA256 = "4edc38166e1f7f1111e3d9723060de3b98069d9779bc07a20f9d1a4b3fac2769"
COUNTS = {"train": (978, 278), "val": (172, 49), "test": (170, 47)}
MATRICES = {"val": [[120, 3], [17, 32]], "test": [[111, 12], [15, 32]]}
FEATURES = {"gender_targeted", "attack_present", "stance", "image_role", "needs_review", "evidence"}
REDACTIONS = (
    ("url", re.compile(r"https?://[^\s]+|www\.[^\s]+", re.I)),
    ("email", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("cn_id", re.compile(r"(?<![A-Za-z0-9])\d{17}[0-9Xx](?![A-Za-z0-9])")),
    ("mobile", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")),
    ("handle", re.compile(r"@[\w.\-]+")),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Non-finite JSON constant")


def decode(value):
    return json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)


class Sources:
    """Bind every consumed file to a byte digest; verify again before success."""

    def __init__(self):
        self.hashes = {}

    def add(self, path):
        path = path.resolve(strict=True)
        require(path.is_file(), "Source is not a regular file")
        digest = file_digest(path)
        require(path not in self.hashes or self.hashes[path] == digest, "Source changed during planning")
        self.hashes[path] = digest
        return path

    def json(self, path):
        path = self.add(path)
        value = decode(path.read_text(encoding="utf-8-sig"))
        self.verify_one(path)
        return value

    def jsonl(self, path):
        path = self.add(path)
        rows = [decode(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        self.verify_one(path)
        return rows

    def verify_one(self, path):
        require(file_digest(path) == self.hashes[path], "Source changed; incomplete export must be preserved")

    def verify(self):
        for path in self.hashes:
            self.verify_one(path)

    def check_binding(self, mapping, paths):
        require(isinstance(mapping, dict), "Missing recorded source-hash bindings")
        for path in paths:
            path = path.resolve(strict=True)
            require(mapping.get(str(path)) == self.hashes[path], "Recorded source binding differs or is missing")


def mask_text(text):
    for name, pattern in REDACTIONS:
        text = pattern.sub(f"[{name.upper()}_REDACTED]", text)
    return text


def check_decision(record):
    require(isinstance(record, dict) and record.get("status") == "ok", "Incomplete Qwen evidence")
    value = record.get("decision")
    require(isinstance(value, dict) and set(value) == FEATURES | {"label"}, "Invalid seven-field evidence")
    require(type(value["label"]) is int and value["label"] in (0, 1), "Invalid Qwen label")
    for field in ("gender_targeted", "attack_present", "needs_review"):
        require(type(value[field]) is bool, "Invalid evidence flag")
    require(value["stance"] in {"support", "oppose", "report", "unclear"}, "Invalid stance")
    require(value["image_role"] in {"supports", "changes_context", "neutral"}, "Evidence is not image-text")
    require(isinstance(value["evidence"], str) and 0 < len(value["evidence"].strip()) <= 120, "Invalid evidence text")
    return value


def checked_request(row, decision, body):
    require(isinstance(body, dict) and set(body) == {"model", "state", "questions"}, "Unexpected request fields")
    require(body["model"] == MODEL, "Wrong Jev model")
    require(isinstance(body["questions"], dict) and set(body["questions"]) == {"policy_violation"}, "Wrong policy")
    require(object_digest(body["questions"]["policy_violation"]) == QUESTION_SHA256, "Policy changed")
    features = {k: decision[k] for k in FEATURES}
    features["evidence"] = mask_text(features["evidence"])
    expected = {"transcription": mask_text(row["text"]), "perception_hypotheses": features,
                "evidence_status": "model_generated_hypotheses_not_verified_annotations"}
    require(body["state"] == expected, "Saved body differs from the row's filtered evidence")
    require(len(canonical(body).encode("utf-8")) <= 12000, "Oversized saved request")


def checked_result(record):
    require(isinstance(record, dict) and record.get("status") == "ok", "Unresolved Jev output")
    require(record.get("model") == MODEL, "Wrong returned model")
    score = record.get("violation_probability")
    require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1, "Invalid risk score")
    for field in ("input_tokens", "output_tokens"):
        require(type(record.get(field)) is int and record[field] >= 0, "Invalid token count")
    seconds = record.get("api_roundtrip_seconds")
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, "Invalid latency")
    # Do not copy arbitrary HTTP headers, credentials, raw errors or account metadata.
    return {k: record[k] for k in ("status", "model", "violation_probability", "input_tokens",
                                  "output_tokens", "api_roundtrip_seconds")}


def load_splits(root, sources):
    directory = root / "data/processed/ltedi_grouped_v1"
    report = sources.json(directory / "preparation_report.json")
    require(report.get("source_commit") == COMMIT and report.get("label_map") ==
            {"Not-Misogyny": 0, "Misogyny": 1}, "Wrong dataset provenance or labels")
    splits, all_ids, all_groups = {}, set(), set()
    for split, (count, positives) in COUNTS.items():
        rows = sources.jsonl(directory / f"{split}.jsonl")
        require(len(rows) == count, "Split count differs from the prepared experiment")
        names, groups = set(), set()
        for row in rows:
            require(isinstance(row, dict) and set(row) == {"id", "image", "text", "label", "group_id"}, "Invalid manifest schema")
            require(type(row["label"]) is int and row["label"] in (0, 1), "Invalid original label")
            require(isinstance(row["text"], str) and row["text"].strip(), "Missing original transcript")
            require(isinstance(row["group_id"], str) and row["group_id"], "Missing group ID")
            require(isinstance(row["id"], str) and re.fullmatch(r"(?:train|dev):[0-9]+\.jpg", row["id"]), "Unsafe source ID")
            require(row["id"] not in all_ids, "Repeated IDs across manifests")
            image = Path(row["image"])
            require(image.is_absolute(), "Image must use its existing absolute source path")
            image = sources.add(image)
            origin, name = row["id"].split(":")
            expected = root / f"data/raw/ltedi_cn_misogyny/Datasets/{origin}/{origin} images/{name}"
            require(image == expected.resolve(strict=True) and image.is_relative_to(root / "data/raw"), "Unexpected source image location")
            require(name not in names, "Ambiguous image name within a split")
            require((split == "test") == (origin == "dev"), "Wrong original-source split")
            names.add(name)
            all_ids.add(row["id"])
            groups.add(row["group_id"])
        require(not groups & all_groups, "Group overlap between splits")
        require(sum(r["label"] for r in rows) == positives, "Split label support differs")
        require(report.get("splits", {}).get(split, {}).get("rows") == count, "Preparation report differs")
        all_groups.update(groups)
        splits[split] = rows
    return splits


def load_errors(root, splits, sources):
    cases, matrices = [], {}
    for split in ("val", "test"):
        if split == "val":
            directory = root / "outputs/ltedi_evidence_val172_v1/lora_evidence"
            qwen_path = directory / "predictions.json"
            requests_path = directory / "jev_requests_LOCAL_ONLY.jsonl"
            controls_path = directory / "request_controls_DO_NOT_UPLOAD.jsonl"
            cloud_path = root / "outputs/jev_lora_remaining152_cloud_v1/predictions_LOCAL_ONLY.json"
            completion = sources.json(cloud_path.parent / "report.json")
            require(completion.get("complete") is True and completion.get("valid_results") == 172, "Validation cloud run incomplete")
            report, protocol, ready = None, None, None
        else:
            directory = root / "outputs/ltedi_frozen_test170_v1"
            qwen_path = directory / "lora_evidence.json"
            requests_path = directory / "requests_LOCAL_ONLY.jsonl"
            controls_path = directory / "request_controls_DO_NOT_UPLOAD.jsonl"
            cloud_path = directory / "cloud/predictions_LOCAL_ONLY.json"
            completion = sources.json(directory / "cloud/completion.json")
            report = sources.json(directory / "final_test_report.json")
            require(completion.get("complete") is True and completion.get("valid_n") == 170 and
                    report.get("complete") is True and report.get("test_n") == 170, "Frozen test run incomplete")
            protocol = sources.json(directory / "frozen_protocol.json")
            require(protocol.get("jev_threshold") == THRESHOLD and protocol.get("jev_model") == MODEL and
                    protocol.get("jev_policy_sha256") == QUESTION_SHA256, "Frozen protocol differs")
            ready = sources.json(directory / "evidence_complete.json")
        qwen = sources.json(qwen_path)
        cloud = sources.json(cloud_path)
        bodies = sources.jsonl(requests_path)
        controls = sources.jsonl(controls_path)
        by_id = {r["id"]: r for r in splits[split]}
        require(set(qwen["items"]) == set(cloud["items"]) == set(by_id), "Cache does not cover the exact split")
        require(len(bodies) == len(controls) == len(by_id), "Request mapping incomplete")
        if split == "val":
            metadata = qwen.get("metadata", {})
            require(metadata.get("stage") == "LORA_EVIDENCE" and metadata.get("adapter_active") is True and
                    metadata.get("modality") == "image-text", "Not the LoRA image-text evidence stage")
            sources.check_binding(cloud.get("metadata", {}).get("source_sha256"),
                                  [qwen_path, requests_path, controls_path, root / "data/processed/ltedi_grouped_v1/val.jsonl"])
        else:
            digest = sources.hashes[(directory / "frozen_protocol.json").resolve()]
            require(qwen.get("stage") == "LORA_EVIDENCE" and
                    all(v.get("protocol_sha256") == digest for v in (qwen, cloud, report, ready)), "Test protocol binding differs")
            require(ready.get("n") == 170 and ready.get("requests_sha256") == object_digest(bodies), "Test request export changed")
            sources.check_binding(ready.get("source_sha256"), [qwen_path, requests_path, controls_path])
            prepared = sources.json(directory / "preparation_complete.json")
            sources.check_binding(prepared.get("prepared_source_sha256"), [root / "data/processed/ltedi_grouped_v1/test.jsonl"])
            sources.check_binding(report.get("report_source_sha256"), [cloud_path])
        seen, matrix = set(), [[0, 0], [0, 0]]
        for index, (body, control) in enumerate(zip(bodies, controls), 1):
            key = control.get("local_id") if split == "val" else control.get("local_id_DO_NOT_UPLOAD")
            require(key in by_id and key not in seen, "Missing or duplicate local request mapping")
            row, raw_qwen = by_id[key], qwen["items"][key]
            decision = check_decision(raw_qwen)
            expected_control = ({"request_index": index, "local_id": key,
                "gold_label_LOCAL_ONLY": row["label"], "qwen_predicted_label_LOCAL_ONLY": decision["label"],
                "prepared_request_sha256": object_digest(body)} if split == "val" else
                {"index": index, "local_id_DO_NOT_UPLOAD": key, "request_sha256": object_digest(body)})
            require(control == expected_control, "Request index, label or body digest differs")
            checked_request(row, decision, body)
            result = checked_result(cloud["items"][key])
            prediction = int(result["violation_probability"] >= THRESHOLD)
            matrix[row["label"]][prediction] += 1
            if prediction != row["label"]:
                cases.append({"split": split, "row": row, "decision": decision, "body": body,
                              "output": result, "prediction": prediction,
                              "error_type": "false_positive" if prediction else "false_negative"})
            seen.add(key)
        require(seen == set(by_id) and matrix == MATRICES[split], "Cached errors differ from the existing published experiment")
        if report:
            require(report["fixed_comparators"]["JEV_T040"]["confusion_matrix_true_rows_predicted_columns_0_1"] == matrix, "Reported test matrix differs")
        matrices[split] = matrix
    return cases, matrices


def build_plan(project_root, output_root, archive=False):
    root, out = project_root.resolve(strict=True), output_root.resolve()
    require(not out.exists() and not output_root.is_symlink(), "OUTPUT_EXISTS_STOP: never overwrite or delete an existing export")
    require(not root.is_relative_to(out), "Output cannot be the project or an ancestor")
    protected = [root / "data", root / "src", root / "outputs/ltedi_evidence_val172_v1",
                 root / "outputs/jev_lora_remaining152_cloud_v1", root / "outputs/ltedi_frozen_test170_v1"]
    require(not any(out.is_relative_to(p) or p.is_relative_to(out) for p in protected), "Output overlaps protected source trees")
    archive_path = out.with_name(out.name + "_PRIVATE.zip") if archive else None
    require(not archive_path or (not archive_path.exists() and not archive_path.is_symlink()), "ARCHIVE_EXISTS_STOP")
    sources = Sources()
    splits = load_splits(root, sources)
    cases, matrices = load_errors(root, splits, sources)
    sources.verify()
    return {"root": root, "out": out, "archive": archive_path, "sources": sources,
            "splits": splits, "cases": cases, "matrices": matrices}


def write_text(path, text, encoding="utf-8"):
    with path.open("x", encoding=encoding, newline="") as stream:
        stream.write(text)
    path.chmod(0o600)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def copy_image(path, destination, sources):
    require(not destination.exists(), "Destination unexpectedly exists")
    sources.verify_one(path)
    with path.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target)
    destination.chmod(0o600)
    require(file_digest(destination) == sources.hashes[path], "Copied image digest differs")


def export(plan):
    out, sources = plan["out"], plan["sources"]
    sources.verify()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    out.chmod(0o700)
    # On failure preserve this new partial directory; never remove user data.
    data, badcase = out / "data", out / "badcase"
    data.mkdir()
    badcase.mkdir()
    metadata, labels = [], []
    for split, rows in plan["splits"].items():
        directory = data / split
        directory.mkdir()
        for row in rows:
            source = Path(row["image"]).resolve(strict=True)
            name = source.name
            copy_image(source, directory / name, sources)
            write_text(directory / (source.stem + ".txt"), row["text"])
            relative = f"{split}/{name}"
            labels.append((relative, row["label"]))
            metadata.append({**row, "image": relative, "source_image_PRIVATE": str(source),
                             "split": split, "image_sha256": sources.hashes[source]})
    with (data / "labels.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", "label"])
        writer.writerows(labels)
    (data / "labels.csv").chmod(0o600)
    write_text(data / "manifest_PRIVATE.jsonl", "".join(canonical(row) + "\n" for row in metadata))
    with (data / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.reader(stream))
    require(csv_rows == [["image_name", "label"], *[[name, str(label)] for name, label in labels]], "Two-column CSV roundtrip differs")
    index = []
    for case in plan["cases"]:
        row, split, kind = case["row"], case["split"], case["error_type"]
        source = Path(row["image"]).resolve(strict=True)
        directory = badcase / split / kind / source.stem
        directory.mkdir(parents=True, exist_ok=False)
        copy_image(source, directory / source.name, sources)
        write_text(directory / "text.txt", row["text"])
        write_json(directory / "qwen_perception.json", case["decision"])
        write_json(directory / "jev_input.json", case["body"])
        write_json(directory / "jev_output.json", case["output"])
        result = {"local_id": row["id"], "split": split, "group_id": row["group_id"],
                  "image_name": source.name, "gold_label": row["label"], "predicted_label": case["prediction"],
                  "qwen_local_label": case["decision"]["label"], "threshold": THRESHOLD,
                  "violation_probability": case["output"]["violation_probability"], "error_type": kind,
                  "jev_input_sha256": object_digest(case["body"]), "eligible_for_training": False,
                  "cause_status": "NOT_ANNOTATED; evidence is a model hypothesis, not ground truth",
                  "saved_response_scope": "Parsed model/probability/usage/latency; not a full HTTP response"}
        write_json(directory / "result.json", result)
        index.append({**result, "case_directory": directory.relative_to(out).as_posix()})
    write_json(badcase / "index_PRIVATE.json", {"threshold": THRESHOLD, "cases": index, "eligible_for_training": False})
    write_text(out / "README_PRIVATE.md", "# Private LT-EDI export\n\n"
               "Do not upload this directory or its ZIP to GitHub. Dataset redistribution rights are unconfirmed.\n"
               "data/labels.csv has exactly image_name,label; image names include project split prefixes.\n"
               "Original image bytes/transcripts/labels and split membership are preserved.\n"
               "badcase contains only LoRA image-text evidence + Jev mistakes at fixed t=0.40, not TEXT_OR.\n"
               "Validation and test cases are diagnostic only, not preference/SFT/DPO training data.\n"
               "No new API requests, inference or training occurred. Source files are not modified.\n")
    sources.verify()
    summary = {"version": VERSION, "complete": True, "private_only": True, "api_calls": 0,
               "gpu_used": False, "source_files_unchanged": True, "source_commit": COMMIT,
               "counts": {split: len(rows) for split, rows in plan["splits"].items()},
               "csv_columns": ["image_name", "label"], "csv_rows": len(labels),
               "badcase_counts": {split: {kind: sum(c["split"] == split and c["error_type"] == kind for c in plan["cases"])
                                         for kind in ("false_positive", "false_negative")} for split in ("val", "test")},
               "threshold": THRESHOLD, "jev_model": MODEL, "matrices": plan["matrices"],
               "source_sha256_PRIVATE": {str(path): digest for path, digest in sources.hashes.items()}}
    if plan["archive"]:
        with plan["archive"].open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                require(not path.is_symlink(), "Unexpected symlink in new export")
                if path.is_file():
                    archive.write(path, path.relative_to(out).as_posix())
            archive.writestr("completion_PRIVATE.json", json.dumps(summary, ensure_ascii=False,
                              indent=2, allow_nan=False) + "\n")
        plan["archive"].chmod(0o600)
    sources.verify()
    # A failed/interrupted ZIP must not leave a successful completion on disk.
    write_json(out / "completion_PRIVATE.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/social_risk_jev"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--archive", action="store_true", help="Create a new PRIVATE ZIP, never upload it")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    out = args.output_root or args.project_root / "outputs/private_dataset_badcases_v1"
    plan = build_plan(args.project_root, out, args.archive)
    print("PLAN TRAIN=978 VAL=172 TEST=170 CSV_ROWS=1320 BADCASE_VAL=20 BADCASE_TEST=27 THRESHOLD=0.40")
    print("API_CALLS=0 KEY_READ=False GPU=False NETWORK=False PRIVATE_ONLY=True")
    if args.plan_only:
        print("PLAN_ONLY; no output written, sources changed or archive created")
        return 0
    summary = export(plan)
    print(f"EXPORT_COMPLETE CSV_ROWS={summary['csv_rows']} BADCASES={len(plan['cases'])} SOURCE_FILES_UNCHANGED=True")
    print(f"PRIVATE_OUTPUT={plan['out']}")
    if plan["archive"]:
        print(f"PRIVATE_ZIP={plan['archive']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # Our exact ValueError messages are static guards; OS/JSON/key errors can
        # contain private paths/content, so report only their types.
        print(f"EXPORT_STOPPED ERROR_TYPE={type(exc).__name__}; preserve any partial output; do not upload private files")
        if type(exc) is ValueError:
            print(f"GUARD={exc}")
        raise SystemExit(2)
