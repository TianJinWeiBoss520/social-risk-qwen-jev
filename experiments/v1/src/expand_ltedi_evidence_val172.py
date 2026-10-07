#!/usr/bin/env python3
"""Expand matched local evidence to all 172 validation rows; zero API calls.

Reuse the frozen 20-row base/LoRA evidence and completed Jev results. Infer
only the remaining 152 rows in each matched Qwen condition. Never access a
key, train/test manifests, optimizer or cloud endpoint. Local inference may
be resumed with identical provenance; the completed cloud ledger is read-only.
"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import evaluate_ltedi_qwen_base as probe
import prepare_ltedi_jev_pilot as prep
import probe_ltedi_lora_evidence20 as evidence
import run_ltedi_jev_cloud20 as cloud


VERSION = "ltedi_matched_evidence_val172_reuse20_v1"
TOTAL = 172
REUSED = 20
REMAINING = 152
STAGES = ("BASE_EVIDENCE", "LORA_EVIDENCE")


def settings(root, out):
    return SimpleNamespace(
        data_dir=root / "data/processed/ltedi_grouped_v1",
        evidence_source=root / "outputs/qwen_base_val_image_text_v1/predictions.json",
        cpu_baseline_dir=root / "outputs/cpu_text_baselines_v1",
        prepared_dir=root / "outputs/jev_base_evidence_val20_prepare_v1",
        adapter_path=root / "outputs/ltedi_language_lora_epoch1_v1/adapter/final",
        model_path=root.parent / "models/Qwen3-VL-32B-Instruct-bnb-4bit-ms",
        # Only a read-only reconstruction target, never created. The original
        # helper rejects existing output paths, including the completed pilot.
        output_dir=out / ".pilot_provenance_check_only",
    )


def check_record(record):
    if record.get("status") not in {"ok", "invalid_json"}:
        raise ValueError("Unexpected local evidence status")
    seconds = record.get("latency_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid local latency measurement")
    if record["status"] == "ok":
        probe.parse_decision(prep.canonical(record["decision"]), "image-text")
    elif "decision" in record:
        raise ValueError("Unresolved evidence must not contain an invented decision")


def check_completed_cloud(directory, pilot, pilot_rows, lora, cpu):
    """Verify old results and one-use ledger without running its API client."""
    plan = prep.read_json(directory / "plan.json")
    report = prep.read_json(directory / "report.json")
    saved = prep.read_json(directory / "predictions_LOCAL_ONLY.json")
    if (plan.get("version") != cloud.VERSION or plan.get("selected_n") != REUSED
            or plan.get("requested_model") != prep.MODEL
            or plan.get("requests_sha256") != cloud.EXPECTED_REQUESTS
            or plan.get("max_post_attempts") != REUSED
            or plan.get("user_authorized_max_posts") != REUSED
            or plan.get("automatic_retries") != 0 or plan.get("resume_allowed") is not False
            or plan.get("test_read") is not False or plan.get("gpu_used") is not False
            or saved.get("metadata") != plan):
        raise ValueError("Completed cloud pilot metadata does not match the original authorization")
    if (report.get("complete") is not True or report.get("api_post_attempts_started") != REUSED
            or report.get("valid_results") != REUSED or report.get("test_read") is not False):
        raise ValueError("All 20 original cloud calls must be accounted for before expansion")
    files = [pilot / "comparison_report.json", pilot / "plan.json",
             pilot / "lora_evidence/predictions.json", pilot / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl",
             pilot / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl",
             directory.parents[1] / "data/processed/ltedi_grouped_v1/val.jsonl",
             Path(probe.__file__), Path(prep.__file__), Path(evidence.__file__), Path(cloud.__file__)]
    hashes = {str(path.resolve()): probe.digest(path) for path in files}
    if plan.get("source_sha256") != hashes:
        raise ValueError("The cloud pilot's allowlisted sources changed")
    items = saved["items"]
    if set(items) != {row["id"] for row in pilot_rows}:
        raise ValueError("Completed cloud results do not cover the exact frozen 20 IDs")
    controls = evidence.read_jsonl(pilot / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl")
    ledger_dir = directory / "started_attempts"
    expected_names = {f"{index:02d}.json" for index in range(1, REUSED + 1)}
    if {path.name for path in ledger_dir.iterdir()} != expected_names:
        raise ValueError("Original cloud authorization ledger is incomplete or has extra entries")
    ledger_files = []
    for index, (row, control) in enumerate(zip(pilot_rows, controls), 1):
        path = ledger_dir / f"{index:02d}.json"
        ledger = prep.read_json(path)
        if (ledger.get("ordinal") != index or ledger.get("local_id_DO_NOT_UPLOAD") != row["id"]
                or ledger.get("request_sha256") != control["prepared_request_sha256"]
                or ledger.get("automatic_replay_forbidden") is not True):
            raise ValueError("Original started-attempt ledger does not bind ID to request")
        record = items[row["id"]]
        if record.get("status") != "ok":
            raise ValueError("Original API result is unresolved; do not replay it")
        cloud.parse_response(prep.canonical({
            "model": record["model"],
            "answers": {"policy_violation": {"type": "noul", "noul": record["violation_probability"]}},
            "usage": {key: record[key] for key in ("input_tokens", "output_tokens")},
        }))
        if (type(record.get("api_roundtrip_seconds")) not in (int, float)
                or not math.isfinite(record["api_roundtrip_seconds"]) or record["api_roundtrip_seconds"] < 0):
            raise ValueError("Invalid cached API latency")
        ledger_files.append(path)
    decisions = {row["id"]: lora[row["id"]]["decision"] for row in pilot_rows}
    if cloud.make_report(plan, pilot_rows, items, decisions, cpu) != report:
        raise ValueError("Completed cloud report does not reproduce from its saved results")
    return [directory / "plan.json", directory / "report.json",
            directory / "predictions_LOCAL_ONLY.json", *ledger_files]


def stage_metadata(plan, stage):
    return {**plan, "stage": stage, "adapter_active": stage == "LORA_EVIDENCE"}


def load_stage(directory, stage, plan, seeds):
    path = directory / "predictions.json"
    if not path.exists():
        if directory.exists():
            raise ValueError("Incomplete stage directory has no checkpoint; preserve it and report")
        return dict(seeds)
    checkpoint = prep.read_json(path)
    if checkpoint.get("metadata") != stage_metadata(plan, stage):
        raise ValueError("Checkpoint provenance differs; no resume under changed settings")
    items = checkpoint["items"]
    if not set(seeds) <= set(items) <= set(plan["validation_ids_LOCAL_ONLY"]):
        raise ValueError("Unexpected or missing cached validation IDs")
    if any(items[key] != value for key, value in seeds.items()):
        raise ValueError("The reused 20 evidence records must remain byte-content equivalent")
    for value in items.values():
        check_record(value)
    return items


def prepare_plan(args):
    root = args.project_root.resolve()
    out = root / "outputs/ltedi_evidence_val172_v1"
    if out.exists() and not args.resume:
        raise ValueError("OUTPUT_EXISTS_STOP: only --resume may reuse this LOCAL inference output")
    if args.resume and not out.is_dir():
        raise ValueError("RESUME_OUTPUT_MISSING_STOP")
    inputs = settings(root, out)
    pilot = root / "outputs/ltedi_lora_evidence20_v1"
    cloud_dir = root / "outputs/jev_lora_val20_cloud_v1"
    for directory in (pilot, cloud_dir):
        if out.is_relative_to(directory) or directory.is_relative_to(out):
            raise ValueError("New output must be separate from the frozen pilot sources")
    pilot_plan, pilot_rows, cpu_name, cpu = evidence.prepare_plan(inputs)
    # JSON object keys (including integer Counter labels) round-trip as
    # strings. Compare persisted metadata in its canonical JSON form.
    pilot_plan = json.loads(prep.canonical(pilot_plan))
    if prep.read_json(pilot / "plan.json") != pilot_plan:
        raise ValueError("Reconstructed matched-pilot provenance differs from the frozen plan")
    rows = probe.load_validation(inputs.data_dir)  # Neither train nor test is read.
    if len(rows) != TOTAL:
        raise ValueError(f"Expected the frozen {TOTAL}-row validation, got {len(rows)}")
    pilot_ids = [row["id"] for row in pilot_rows]
    remaining = [row for row in rows if row["id"] not in set(pilot_ids)]
    if len(pilot_ids) != REUSED or len(remaining) != REMAINING:
        raise ValueError("The complement must contain exactly 152 never-called validation IDs")
    pilot_report = prep.read_json(pilot / "comparison_report.json")
    if (pilot_report.get("version") != evidence.VERSION or pilot_report.get("sources_unchanged") is not True
            or pilot_report.get("api_calls") != 0 or pilot_report.get("test_read") is not False):
        raise ValueError("Unexpected local matched-pilot report")
    seeds, extra_files = {}, [pilot / "plan.json", pilot / "comparison_report.json", Path(cloud.__file__), Path(__file__)]
    for stage, name in zip(STAGES, ("base", "lora")):
        directory = pilot / f"{name}_evidence"
        saved = prep.read_json(directory / "predictions.json")
        if (saved.get("metadata") != stage_metadata(pilot_plan, stage)
                or set(saved["items"]) != set(pilot_ids)):
            raise ValueError("Frozen matched evidence metadata or selection changed")
        requests, controls = [], []
        for index, row in enumerate(pilot_rows, 1):
            record = saved["items"][row["id"]]
            check_record(record)
            if record["status"] != "ok":
                raise ValueError("The frozen evidence pilot must be complete")
            request, _, _ = prep.make_request(row, record["decision"])
            requests.append(request)
            controls.append({"request_index": index, "local_id": row["id"],
                             "gold_label_LOCAL_ONLY": row["label"],
                             "qwen_predicted_label_LOCAL_ONLY": record["decision"]["label"],
                             "prepared_request_sha256": prep.object_digest(request)})
        if (evidence.read_jsonl(directory / "jev_requests_LOCAL_ONLY.jsonl") != requests
                or evidence.read_jsonl(directory / "request_controls_DO_NOT_UPLOAD.jsonl") != controls
                or pilot_report[f"{name}_request_preparation"].get("requests_sha256") != prep.object_digest(requests)):
            raise ValueError("Frozen local evidence requests/controls differ from reconstruction")
        if name == "lora" and prep.object_digest(requests) != cloud.EXPECTED_REQUESTS:
            raise ValueError("The original 20 authorized LoRA payloads changed")
        seeds[stage] = saved["items"]
        extra_files.extend([directory / "predictions.json", directory / "jev_requests_LOCAL_ONLY.jsonl",
                            directory / "request_controls_DO_NOT_UPLOAD.jsonl"])
    extra_files += check_completed_cloud(cloud_dir, pilot, pilot_rows, seeds["LORA_EVIDENCE"], cpu)
    hashes = {**pilot_plan["source_sha256"],
              **{str(path.resolve()): probe.digest(path) for path in extra_files}}
    plan = {
        "version": VERSION, "validation_n": TOTAL, "reused_evidence_n_each_stage": REUSED,
        "new_evidence_n_each_stage": REMAINING, "new_sample_inferences_planned": 2 * REMAINING,
        "stage_order": list(STAGES), "validation_ids_LOCAL_ONLY": [row["id"] for row in rows],
        "pilot_ids_LOCAL_ONLY": pilot_ids, "remaining_ids_LOCAL_ONLY": [row["id"] for row in remaining],
        "validation_groups": len({row["group_id"] for row in rows}),
        "labels_LOCAL_ONLY": dict(Counter(row["label"] for row in rows)),
        "model_path": str(inputs.model_path), "adapter_path": str(inputs.adapter_path),
        "modality": "image-text", "prompt_version": probe.PROMPT_VERSION,
        "prompt_sha256": pilot_plan["prompt_sha256"], "max_new_tokens": evidence.NEW_TOKENS,
        "max_pixels": evidence.PIXELS, "do_sample": False, "max_format_attempts_per_sample": 2,
        "precision_control": pilot_plan["precision_control"],
        "source_validation_image_bytes_sha256": pilot_plan["frozen_preparation_identity"]["qwen_cache_metadata"]["validation_image_bytes_sha256"],
        "source_sha256": hashes, "cpu_name": cpu_name,
        "existing_cloud_results_n": REUSED, "existing_cloud_results_read_only": True,
        "future_cloud_remaining_max_posts": REMAINING,
        "api_calls": 0, "api_key_read": False, "cloud_upload": False,
        "test_read": False, "train_read": False, "optimizer_steps": 0,
        "local_inference_resume_allowed": True, "cloud_resume_or_replay_allowed": False,
    }
    plan = json.loads(prep.canonical(plan))
    if args.resume and prep.read_json(out / "plan.json") != plan:
        raise ValueError("LOCAL_RESUME_PROVENANCE_CHANGED_STOP")
    cache = {}
    for stage, name in zip(STAGES, ("base", "lora")):
        cache[stage] = load_stage(out / f"{name}_evidence", stage, plan, seeds[stage])
    return inputs, out, plan, rows, remaining, seeds, cache, cpu


def verify_sources(plan, rows):
    for filename, expected in plan["source_sha256"].items():
        if probe.digest(Path(filename)) != expected:
            raise RuntimeError("A source file changed; stop before preparing any cloud request")
    image_hashes = [(row["id"], probe.digest(Path(row["image"]))) for row in rows]
    checksum = hashlib.sha256(json.dumps(image_hashes, sort_keys=True).encode()).hexdigest()
    if checksum != plan["source_validation_image_bytes_sha256"]:
        raise RuntimeError("Validation image bytes changed during inference")


def evaluate_remaining(processor, model, rows, directory, stage, plan, items):
    if not directory.exists():
        directory.mkdir(exist_ok=False)
    pending = [row for row in rows if row["id"] not in items]
    try:
        for index, row in enumerate(pending, 1):
            record = probe.infer_one(processor, model, {"image": row["image"], "text": row["text"]},
                                     "image-text", evidence.NEW_TOKENS)
            check_record(record)
            items[row["id"]] = record
            print(f"{stage} NEW=[{index}/{len(pending)}] TOTAL_DONE={len(items)}/{TOTAL} "
                  f"STATUS={record['status']} SEC={record['latency_seconds']:.2f}", flush=True)
            if index % 5 == 0:
                probe.atomic_json(directory / "predictions.json", {"metadata": stage_metadata(plan, stage), "items": items})
    finally:
        probe.atomic_json(directory / "predictions.json", {"metadata": stage_metadata(plan, stage), "items": items})
    return items


def finish(out, plan, rows, remaining, cache, cpu):
    base, lora = [cache[stage] for stage in STAGES]
    verify_sources(plan, rows)
    paired = [row for row in rows if base.get(row["id"], {}).get("status") == lora.get(row["id"], {}).get("status") == "ok"]
    gained = sum(base[r["id"]]["decision"]["label"] != r["label"] and lora[r["id"]]["decision"]["label"] == r["label"] for r in paired)
    lost = sum(base[r["id"]]["decision"]["label"] == r["label"] and lora[r["id"]]["decision"]["label"] != r["label"] for r in paired)
    exports = {name: evidence.export_local_requests(rows, cache[stage], out / f"{name}_evidence")
               for stage, name in zip(STAGES, ("base", "lora"))}
    if exports["lora"]["ready"]:
        directory = out / "remaining152_for_jev"
        directory.mkdir(exist_ok=True)
        remaining_export = evidence.export_local_requests(remaining, lora, directory)
        controls = evidence.read_jsonl(directory / "request_controls_DO_NOT_UPLOAD.jsonl")
        if (len(controls) != REMAINING
                or [row["local_id"] for row in controls] != plan["remaining_ids_LOCAL_ONLY"]
                or set(row["local_id"] for row in controls) & set(plan["pilot_ids_LOCAL_ONLY"])):
            raise RuntimeError("Remaining request export contains reused or unexpected IDs")
    else:
        remaining_export = {"ready": False, "reason": "Incomplete full-validation LoRA evidence; no subset uploaded or exported"}
    report = {
        "version": VERSION, "scope": "full_validation_not_final_test", "test_read": False,
        "api_calls": 0, "cloud_upload": False, "optimizer_steps": 0,
        "base": evidence.stage_report(rows, base), "lora": evidence.stage_report(rows, lora),
        "text_only_same_all_rows": {"name": plan["cpu_name"], "metrics": probe.classification_metrics(
            [r["label"] for r in rows], [cpu[r["id"]] for r in rows])},
        "paired_both_schema_valid": {"n": len(paired), "gained": gained, "lost": lost, "net_correct": gained - lost},
        "local_request_preparation": exports,
        "remaining152_request_preparation": remaining_export,
        "existing_jev_results_reused_read_only": REUSED,
        "jev_full_validation_results_available": False,
        "sources_unchanged": True,
        "cautions": [
            "No new Jev calls were made. A separate zero-call budget preflight must precede the authorized 152 calls.",
            "The 20 original payloads/results/attempt ledgers are unchanged; never call them again in this expansion.",
            "Base and LoRA share the same prompt, precision preparation and generation settings.",
            "Seven-field evidence is model-generated and has no supervised explanation ground truth.",
            "Do not tune on the final test set. Validation is not an unbiased final performance estimate.",
            "Qwen timings include reused earlier runs and exclude model loading and Jev; not live end-to-end latency.",
            "Basic masking is not guaranteed anonymization. Prepared files are local only.",
            "Dataset political status remains USER_SPOT_CHECK_ONLY, not fully verified.",
        ],
    }
    probe.atomic_json(out / "comparison_report.json", report)
    ready = all(export["ready"] for export in exports.values()) and remaining_export["ready"]
    print(f"LOCAL_EVIDENCE_COMPLETE={ready} BASE_VALID={report['base']['schema_valid_n']}/{TOTAL} "
          f"LORA_VALID={report['lora']['schema_valid_n']}/{TOTAL} NEW_API_CALLS=0", flush=True)
    print("PAIRED=" + prep.canonical(report["paired_both_schema_valid"]), flush=True)
    print(f"REPORT={out / 'comparison_report.json'}; next step is cloud budget preflight, not training", flush=True)
    return 0 if ready else 2


def run(args):
    inputs, out, plan, rows, remaining, seeds, cache, cpu = prepare_plan(args)
    pending = {stage: sum(row["id"] not in cache[stage] for row in remaining) for stage in STAGES}
    print(f"VAL={TOTAL} REUSE_EVIDENCE={REUSED} REUSE_COMPLETED_JEV={REUSED} REMAINING={REMAINING}", flush=True)
    print("PENDING=" + prep.canonical(pending), flush=True)
    print("API_CALLS=0 KEY_READ=False CLOUD_UPLOAD=False TEST_READ=False TRAINING=False", flush=True)
    if args.plan_only:
        print("PLAN_ONLY: no output written, model loaded, GPU package imported, image decoded or network accessed", flush=True)
        return 0
    lock = out / ".local_worker.lock"
    if lock.exists():
        raise ValueError("LOCAL_WORKER_LOCK_EXISTS_STOP: do not run concurrent writers or remove the lock blindly")
    # A missing GPU fails before creating output. A completed local resume
    # can reproduce reports without loading a GPU model at all.
    processor, model = evidence.load_matched_model(inputs) if sum(pending.values()) else (None, None)
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
        probe.atomic_json(out / "plan.json", plan)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(prep.canonical({"pid": os.getpid(), "scope": "local_inference_only"}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for stage, name in zip(STAGES, ("base", "lora")):
            directory = out / f"{name}_evidence"
            if stage == "BASE_EVIDENCE" and pending[stage]:
                with model.disable_adapter():
                    cache[stage] = evaluate_remaining(processor, model, remaining, directory, stage, plan, cache[stage])
            else:
                cache[stage] = evaluate_remaining(processor, model, remaining, directory, stage, plan, cache[stage])
        return finish(out, plan, rows, remaining, cache, cpu)
    finally:
        lock.unlink()  # Only this process's explicitly named, exclusive lock.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=probe.DEFAULT_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume LOCAL evidence only, never cloud calls")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
