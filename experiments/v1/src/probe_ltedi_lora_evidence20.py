#!/usr/bin/env python3
"""Local matched base/LoRA seven-field evidence probe on the frozen 20-row pilot.

No API client, key access, optimizer, training, or final-test access. Plan mode
uses only the standard library. Request files are LOCAL ONLY, never uploaded.
Requires the unchanged evaluate_ltedi_qwen_base and prepare_ltedi_jev_pilot
helpers in the same directory. No existing source/output file is overwritten.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import evaluate_ltedi_qwen_base as probe
import prepare_ltedi_jev_pilot as prep


ROOT = probe.DEFAULT_ROOT
VERSION = "ltedi_matched_seven_field_evidence20_v1"
EXPECTED_WEIGHTS = "8c97666529ed29de285430323acf34a740e484eb75f025db15eb4a50c598f3a2"
PIXELS = 768 * 32 * 32
NEW_TOKENS = 384


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line, object_pairs_hook=probe.unique_object)
                for line in stream if line.strip()]


def prepare_plan(args):
    out = args.output_dir.resolve()
    protected = (args.data_dir, args.evidence_source.parent, args.cpu_baseline_dir,
                 args.prepared_dir, args.adapter_path, args.model_path)
    for directory in protected:
        directory = directory.resolve()
        if out.is_relative_to(directory) or directory.is_relative_to(out):
            raise ValueError("Output must be separate from every source directory")
    if out.exists():
        raise ValueError("OUTPUT_EXISTS_STOP: do not overwrite or restart this probe")

    prepared = prep.read_json(args.prepared_dir / "preparation_report.json")
    if (prepared.get("stage") != "OFFLINE_PREPARATION_ONLY"
            or prepared.get("test_read") is not False
            or prepared.get("cloud_upload") is not False
            or prepared.get("api_key_read") is not False
            or type(prepared.get("api_calls")) is not int or prepared["api_calls"] != 0):
        raise ValueError("Expected the completed local-only preparation report")
    identity = prepared["identity"]
    if identity.get("selected_n") != 20:
        raise ValueError("This probe requires the existing frozen 20-row selection")
    # Recompute provenance, source-image hashes and selection without writing.
    # This reads val.jsonl only; train/test manifests are never opened or hashed.
    inputs = SimpleNamespace(data_dir=args.data_dir, evidence_source=args.evidence_source,
                             cpu_baseline_dir=args.cpu_baseline_dir, output_dir=out,
                             limit=20, seed=identity["seed"])
    requests, controls, reconstructed = prep.prepare(inputs)
    if identity != reconstructed["identity"]:
        raise ValueError("Preparation provenance changed; do not redraw the pilot")
    if read_jsonl(args.prepared_dir / "prepared_requests.jsonl") != requests:
        raise ValueError("Prepared request file differs from its frozen provenance")
    if read_jsonl(args.prepared_dir / "local_controls_DO_NOT_UPLOAD.jsonl") != controls:
        raise ValueError("Local controls differ from their frozen provenance")
    all_rows = probe.load_validation(args.data_dir)
    by_id = {row["id"]: row for row in all_rows}
    rows = [by_id[key] for key in identity["selected_ids_LOCAL_ONLY"]]
    cpu_name, cpu = probe.load_cpu_predictions(args.cpu_baseline_dir, args.data_dir, all_rows)

    adapter = args.adapter_path.resolve()
    model_path = args.model_path.resolve()
    weights = adapter / "adapter_model.safetensors"
    if probe.digest(weights) != EXPECTED_WEIGHTS:
        raise ValueError("ADAPTER_HASH_MISMATCH: expected the verified new-project epoch1 final")
    config = prep.read_json(adapter / "adapter_config.json")
    metadata = prep.read_json(adapter / "training_metadata.json")
    if (config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM"
            or config.get("r") != 8 or config.get("lora_alpha") != 16
            or config.get("bias") != "none" or config.get("modules_to_save")):
        raise ValueError("Unexpected adapter configuration")
    if (metadata.get("task_version") != "ltedi_binary_json_label_v1"
            or metadata.get("trainable_scope") != "language_attention_lora_only"
            or metadata.get("visual_encoder_frozen") is not True
            or metadata.get("complete") is not True
            or metadata.get("completed_steps") != 245
            or metadata.get("used_train_records") != 978
            or metadata.get("max_pixels") != PIXELS
            or Path(metadata["base_model"]).resolve() != model_path):
        raise ValueError("Not the completed, frozen-vision, one-epoch project adapter")
    if ((model_path / "adapter_config.json").exists()
            or metadata["source_sha256"]["base_config"] != probe.digest(model_path / "config.json")
            or metadata["source_sha256"]["val"] != identity["source_validation_sha256"]):
        raise ValueError("Base config or validation manifest differs from LoRA training")
    cache_meta = identity["qwen_cache_metadata"]
    if (Path(cache_meta["model_path"]).resolve() != model_path
            or cache_meta["model_config_sha256"] != probe.digest(model_path / "config.json")
            or cache_meta["max_pixels"] != PIXELS
            or cache_meta["max_new_tokens"] != NEW_TOKENS
            or cache_meta["do_sample"] is not False):
        raise ValueError("Source cache has different base/generation settings")

    source_files = [args.data_dir / "val.jsonl", args.evidence_source,
                    args.prepared_dir / "preparation_report.json",
                    args.prepared_dir / "prepared_requests.jsonl",
                    args.prepared_dir / "local_controls_DO_NOT_UPLOAD.jsonl",
                    args.cpu_baseline_dir / "validation_report.json",
                    args.cpu_baseline_dir / f"{cpu_name}_validation_predictions.jsonl",
                    model_path / "config.json", weights, adapter / "adapter_config.json",
                    adapter / "training_metadata.json", Path(probe.__file__),
                    Path(prep.__file__), Path(__file__)]
    hashes = {str(path.resolve()): probe.digest(path) for path in source_files}
    plan = {
        "version": VERSION, "stage_order": ["BASE_EVIDENCE", "LORA_EVIDENCE"],
        "selected_ids_LOCAL_ONLY": [row["id"] for row in rows], "selected_n": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "selected_labels_LOCAL_ONLY": dict(Counter(row["label"] for row in rows)),
        "sample_inferences_planned": 40, "max_format_attempts_per_sample": 2,
        "modality": "image-text", "prompt_version": probe.PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(probe.SYSTEM_PROMPT.encode()).hexdigest(),
        "max_pixels": PIXELS, "max_new_tokens": NEW_TOKENS, "do_sample": False,
        "model_path": str(model_path), "adapter_path": str(adapter),
        "source_sha256": hashes, "frozen_preparation_identity": identity,
        "precision_control": "Same PEFT-prepared model; disable adapter for base, enable saved adapter for LoRA",
        "test_read": False, "train_read": False, "optimizer_steps": 0,
        "api_calls": 0, "api_key_read": False, "cloud_upload": False,
    }
    return plan, rows, cpu_name, cpu


def verify_loaded_adapter(model, saved, torch):
    live = {name.replace(".default.", "."): value
            for name, value in model.named_parameters() if "lora_" in name}
    if len(live) != 512 or set(live) != set(saved):
        raise RuntimeError("Loaded adapter tensor keys/count do not match the 512 saved tensors")
    for name, value in live.items():
        if ("language_model" not in name.split(".") or "visual" in name.split(".")
                or not bool(torch.isfinite(saved[name]).all())
                or not torch.equal(value.detach().cpu(), saved[name])):
            raise RuntimeError("Loaded weights do not exactly match the saved language adapter")
    if any(value.requires_grad for value in model.parameters()):
        raise RuntimeError("Inference must not have any trainable parameters")


def load_matched_model(args):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("GPU_REQUIRED: do not run actual inference in no-card mode")
    print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)
    processor, base = probe.load_model(args.model_path, PIXELS)
    visuals = [module for name, module in base.named_modules() if name.split(".")[-1] == "visual"]
    if len(visuals) != 1:
        raise RuntimeError("Expected one visual encoder")
    # Reproduce the one-epoch evaluation's precision preparation, without
    # training, new adapters, optimizers or gradient-checkpointing hooks.
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    visuals[0].to(dtype=torch.bfloat16)
    base.config.use_cache = True
    if hasattr(base.config, "text_config"):
        base.config.text_config.use_cache = True
    base.gradient_checkpointing_disable()
    model = PeftModel.from_pretrained(base, str(args.adapter_path), adapter_name="default",
                                     is_trainable=False, local_files_only=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    saved = load_file(str(args.adapter_path / "adapter_model.safetensors"), device="cpu")
    verify_loaded_adapter(model, saved, torch)
    layers = model.get_layer_status()
    if (len(layers) != 256 or list(model.active_adapters) != ["default"]
            or any(layer.enabled is not True or layer.active_adapters != ["default"]
                   or layer.merged_adapters for layer in layers)):
        raise RuntimeError("Adapter activation audit failed")
    del saved
    print("LOADED_WEIGHT_MATCH=512/512 ALL_PARAMETERS_FROZEN=True VISION_FROZEN=True", flush=True)
    return processor, model


def evaluate_stage(processor, model, rows, directory, stage, plan):
    directory.mkdir(exist_ok=False)
    items = {}
    metadata = {**plan, "stage": stage,
                "adapter_active": stage == "LORA_EVIDENCE"}
    # Only image and transcription are given to the inference helper. Labels,
    # group IDs and local IDs cannot be interpolated into model messages.
    try:
        for index, row in enumerate(rows, 1):
            record = probe.infer_one(processor, model,
                                     {"image": row["image"], "text": row["text"]},
                                     "image-text", NEW_TOKENS)
            if record.get("status") == "ok":
                probe.parse_decision(prep.canonical(record["decision"]), "image-text")
            if not math.isfinite(record["latency_seconds"]) or record["latency_seconds"] < 0:
                raise ValueError("Invalid latency measurement")
            items[row["id"]] = record
            print(f"{stage} [{index}/{len(rows)}] {row['id']} STATUS={record['status']} "
                  f"FIRST_PASS={record.get('first_pass_schema_valid', False)} "
                  f"SEC={record['latency_seconds']:.2f}", flush=True)
            if index % 5 == 0:
                probe.atomic_json(directory / "predictions.json", {"metadata": metadata, "items": items})
    finally:
        probe.atomic_json(directory / "predictions.json", {"metadata": metadata, "items": items})
    return items


def stage_report(rows, items):
    valid = [row for row in rows if items.get(row["id"], {}).get("status") == "ok"]
    correct = sum(items[row["id"]]["decision"]["label"] == row["label"] for row in valid)
    latency = sorted(item["latency_seconds"] for item in items.values())
    return {
        "n": len(rows), "schema_valid_n": len(valid),
        "first_pass_schema_valid_n": sum(item.get("first_pass_schema_valid") is True for item in items.values()),
        "schema_valid_rate": len(valid) / len(rows), "correct_n": correct,
        "pipeline_accuracy_errors_counted_wrong": correct / len(rows),
        "classification_on_valid_outputs_only": probe.classification_metrics(
            [row["label"] for row in valid], [items[row["id"]]["decision"]["label"] for row in valid]
        ) if valid else None,
        "unresolved_ids_LOCAL_ONLY": [row["id"] for row in rows if row not in valid],
        "latency_seconds": {"n": len(latency), "mean": sum(latency) / len(latency) if latency else None,
                            "p50": latency[len(latency) // 2] if latency else None,
                            "p95": latency[max(0, math.ceil(len(latency) * .95) - 1)] if latency else None,
                            "scope": "Qwen image+text processing/generation/parsing/retries only; excludes loading, saves and Jev"},
        "failure_policy": "Invalid output remains unresolved and requires review; no invented benign/harmful label",
    }


def export_local_requests(rows, items, directory):
    if any(items.get(row["id"], {}).get("status") != "ok" for row in rows):
        return {"ready": False, "reason": "Incomplete seven-field evidence; no success-only subset exported"}
    requests, controls = [], []
    redactions = Counter()
    for index, row in enumerate(rows, 1):
        request, masked, _ = prep.make_request(row, items[row["id"]]["decision"])
        requests.append(request)
        redactions.update(masked)
        controls.append({"request_index": index, "local_id": row["id"],
                         "gold_label_LOCAL_ONLY": row["label"],
                         "qwen_predicted_label_LOCAL_ONLY": items[row["id"]]["decision"]["label"],
                         "prepared_request_sha256": prep.object_digest(request)})
    prep.atomic_jsonl(directory / "jev_requests_LOCAL_ONLY.jsonl", requests)
    prep.atomic_jsonl(directory / "request_controls_DO_NOT_UPLOAD.jsonl", controls)
    return {"ready": True, "n": len(requests), "requests_sha256": prep.object_digest(requests),
            "basic_redactions": dict(redactions), "manual_payload_review_done": False,
            "api_calls": 0, "cloud_upload": False, "full_deidentification_guaranteed": False}


def run(args):
    plan, rows, cpu_name, cpu = prepare_plan(args)
    print(f"SELECTED={len(rows)} GROUPS={plan['groups']} MODALITY=image-text BASE=20 LORA=20 TEST_READ=False", flush=True)
    print("SELECTED_LABELS_LOCAL_ONLY=" + prep.canonical(plan["selected_labels_LOCAL_ONLY"]), flush=True)
    print("API_CALLS=0 CLOUD_UPLOAD=False TRAINING=False SAME_PROMPT_AND_PRECISION=True", flush=True)
    if args.plan_only:
        print("PLAN_ONLY: no model loaded, GPU dependency imported, image decoded, key read or output written", flush=True)
        return 0
    processor, model = load_matched_model(args)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    probe.atomic_json(out / "plan.json", plan)
    with model.disable_adapter():
        base = evaluate_stage(processor, model, rows, out / "base_evidence", "BASE_EVIDENCE", plan)
    lora = evaluate_stage(processor, model, rows, out / "lora_evidence", "LORA_EVIDENCE", plan)
    for filename, expected in plan["source_sha256"].items():
        if probe.digest(Path(filename)) != expected:
            raise RuntimeError("Source file changed during inference; stop before request export")
    paired_rows = [row for row in rows if base[row["id"]]["status"] == lora[row["id"]]["status"] == "ok"]
    gained = sum(base[row["id"]]["decision"]["label"] != row["label"]
                 and lora[row["id"]]["decision"]["label"] == row["label"] for row in paired_rows)
    lost = sum(base[row["id"]]["decision"]["label"] == row["label"]
               and lora[row["id"]]["decision"]["label"] != row["label"] for row in paired_rows)
    report = {
        "version": VERSION, "test_read": False, "api_calls": 0, "cloud_upload": False,
        "base": stage_report(rows, base), "lora": stage_report(rows, lora),
        "text_only_same_rows": {"name": cpu_name, "metrics": probe.classification_metrics(
            [row["label"] for row in rows], [cpu[row["id"]] for row in rows])},
        "paired_both_schema_valid": {"n": len(paired_rows), "gained": gained, "lost": lost, "net_correct": gained - lost},
        "base_request_preparation": export_local_requests(rows, base, out / "base_evidence"),
        "lora_request_preparation": export_local_requests(rows, lora, out / "lora_evidence"),
        "sources_unchanged": True,
        "cautions": ["Same fixed 20 examples, not a quality estimate or final test; only 3 positives in the real pilot.",
                     "LoRA was supervised on labels, not seven-field evidence: valid JSON does not prove factual explanations.",
                     "No Jev calls, probabilities, risk calibration, severity labels or whole-system latency were measured.",
                     "Request files are local preparation only; any later cloud transfer needs separate user authorization.",
                     "The old cached base is not the matched precision control; use this run's disabled-adapter base.",
                     "Pattern masking does not remove every name/identifier; final API payloads still need inspection."],
    }
    probe.atomic_json(out / "comparison_report.json", report)
    for stage in ("base", "lora"):
        value = report[stage]
        print(f"{stage.upper()} SCHEMA_VALID={value['schema_valid_n']}/20 FIRST_PASS={value['first_pass_schema_valid_n']}/20", flush=True)
    print("PAIRED=" + prep.canonical(report["paired_both_schema_valid"]), flush=True)
    ready = report["base_request_preparation"]["ready"] and report["lora_request_preparation"]["ready"]
    print(f"LOCAL_EVIDENCE_PROBE_COMPLETE FULL_REQUESTS_PREPARED={ready} API_CALLS=0 REPORT={out / 'comparison_report.json'}", flush=True)
    return 0 if ready else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ltedi_grouped_v1")
    parser.add_argument("--evidence-source", type=Path, default=ROOT / "outputs/qwen_base_val_image_text_v1/predictions.json")
    parser.add_argument("--cpu-baseline-dir", type=Path, default=ROOT / "outputs/cpu_text_baselines_v1")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "outputs/jev_base_evidence_val20_prepare_v1")
    parser.add_argument("--adapter-path", type=Path, default=ROOT / "outputs/ltedi_language_lora_epoch1_v1/adapter/final")
    parser.add_argument("--model-path", type=Path, default=probe.DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ltedi_lora_evidence20_v1")
    parser.add_argument("--plan-only", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
