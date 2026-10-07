#!/usr/bin/env python3
"""Frozen, one-use LT-EDI final test. Four explicit stages; no training.

Plan is stdlib-only and never opens test/train manifests, a key, or a service.
Prepare freezes rules before opening test labels, saves label-free inputs and
existing CPU-model predictions without reporting test quality. Evidence uses
the matched base/LoRA loader. Cloud sends only masked text/six evidence fields,
with a durable ledger, no retry/redirect/resume, at most 170 new POSTs. Only
the final report evaluates test labels. Do not tune anything on this test.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import evaluate_ltedi_qwen_base as ev
import prepare_ltedi_jev_pilot as prep
import probe_ltedi_lora_evidence20 as evidence
import expand_ltedi_evidence_val172 as expand
import run_ltedi_jev_cloud20 as cloud
import run_ltedi_jev_remaining152 as previous
import train_ltedi_cpu_baselines as cpu_baseline

VERSION = "ltedi_frozen_final_test170_v1"
TEST_N = 170
THRESHOLD = .40
NEW_POST_LIMIT = 170
USER_BUDGET_USD = 5.0
SOFTWARE_NEW_FORECAST_USD = .50
PRICE = .042
RESERVE_TOKENS = 65536
MODEL_NAMES = ("tfidf_logreg_c1", "tfidf_logreg_c4", "tfidf_xgboost_d2", "tfidf_xgboost_d4")
STAGES = ("BASE_EVIDENCE", "LORA_EVIDENCE")


def paths(root):
    root = root.resolve()
    return SimpleNamespace(root=root, data=root / "data/processed/ltedi_grouped_v1",
        cpu=root / "outputs/cpu_text_baselines_v1", validation=root / "outputs/jev_lora_remaining152_cloud_v1",
        local_validation=root / "outputs/ltedi_evidence_val172_v1",
        out=root / "outputs/ltedi_frozen_test170_v1",
        model_path=root.parent / "models/Qwen3-VL-32B-Instruct-bnb-4bit-ms",
        adapter_path=root / "outputs/ltedi_language_lora_epoch1_v1/adapter/final")


def cost(tokens):
    return tokens * PRICE / 1_000_000


def write_new_json(path, value):
    # Freeze durably before any test read. Never replace a frozen rule.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(prep.canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def checked_probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid finite probability")
    return float(value)


def plan(root):
    p = paths(root)
    if p.out.exists():
        raise ValueError("OUTPUT_EXISTS_STOP: final test preparation cannot be overwritten")
    old = prep.read_json(p.validation / "predictions_LOCAL_ONLY.json")
    report = prep.read_json(p.validation / "report.json")
    previous.verify_hashes(old["metadata"]["source_sha256"])
    if (report.get("complete") is not True or report.get("version") != previous.VERSION
            or report.get("validation_n") != 172 or report.get("valid_results") != 172
            or report.get("test_read") is not False or report.get("new_post_attempts_started") != 152):
        raise ValueError("Expected the completed, unchanged 172-row validation run")
    rows = ev.load_validation(p.data)
    jev = old["items"]
    lora = prep.read_json(p.local_validation / "lora_evidence/predictions.json")["items"]
    if len(rows) != 172 or set(jev) != set(lora) or set(jev) != {r["id"] for r in rows}:
        raise ValueError("Validation source coverage changed")
    if any(jev[r["id"]].get("status") != "ok" or lora[r["id"]].get("status") != "ok" for r in rows):
        raise ValueError("Unresolved validation is not a frozen successful experiment")
    selected, cpu = ev.load_cpu_predictions(p.cpu, p.data, rows)
    cpu_report = prep.read_json(p.cpu / "validation_report.json")
    if selected != "tfidf_logreg_c4" or set(cpu_report.get("models", {})) != set(MODEL_NAMES):
        raise ValueError("Expected the four existing, already-trained CPU controls")
    thresholds = {name: checked_probability(cpu_report["models"][name]["validation_selected_threshold"]["threshold"])
                  for name in MODEL_NAMES}
    truth = [r["label"] for r in rows]
    scores = [checked_probability(jev[r["id"]]["violation_probability"]) for r in rows]
    if ev.classification_metrics(truth, [int(s >= .5) for s in scores]) != report["jev_classification_valid_only"]:
        raise ValueError("Validation metrics do not reproduce")
    prepared = prep.read_json(p.data / "preparation_report.json")
    if prepared.get("group_overlap_between_splits") != 0 or prepared["splits"]["test"]["rows"] != TEST_N:
        raise ValueError("Expected the original prepared, group-disjoint 170-row test")
    if not (p.data / "test.jsonl").is_file():
        raise ValueError("TEST_MANIFEST_MISSING")
    files = [p.validation / "plan.json", p.validation / "report.json",
        p.validation / "predictions_LOCAL_ONLY.json", p.data / "preparation_report.json",
        p.cpu / "validation_report.json", p.local_validation / "lora_evidence/predictions.json",
        *(p.cpu / f"{name}.joblib" for name in MODEL_NAMES), Path(cpu_baseline.__file__), Path(__file__)]
    if ev.digest(p.adapter_path / "adapter_model.safetensors") != evidence.EXPECTED_WEIGHTS:
        raise ValueError("Frozen language adapter weights changed")
    hashes = {**old["metadata"]["source_sha256"], **{str(f.resolve()): ev.digest(f) for f in files}}
    known_previous = report["billing_estimates"]["combined_known_usage_cost_usd"]
    if type(known_previous) not in (int, float) or not math.isfinite(known_previous) or known_previous < 0:
        raise ValueError("Invalid previous expense accounting")
    reserved = cost(NEW_POST_LIMIT * RESERVE_TOKENS)
    if reserved > SOFTWARE_NEW_FORECAST_USD or known_previous + reserved > USER_BUDGET_USD:
        raise ValueError("BUDGET_FORECAST_STOP: no new calls authorized")
    frozen = {"version": VERSION, "test_n": TEST_N,
        "primary_rule": "selected_text_label == 1 OR jev_probability >= 0.40",
        "jev_threshold": THRESHOLD, "jev_control_threshold": .5,
        "selected_text_model": selected, "cpu_thresholds_frozen_from_validation": thresholds,
        "jev_model": prep.MODEL, "jev_policy_sha256": prep.object_digest(prep.QUESTION),
        "qwen_prompt_sha256": hashlib.sha256(ev.SYSTEM_PROMPT.encode()).hexdigest(),
        "qwen_modality": "image-text", "qwen_max_new_tokens": evidence.NEW_TOKENS,
        "qwen_max_pixels": evidence.PIXELS, "qwen_do_sample": False,
        "model_path": str(p.model_path), "adapter_path": str(p.adapter_path),
        "expected_adapter_sha256": evidence.EXPECTED_WEIGHTS,
        "matched_qwen_stages": list(STAGES), "max_local_sample_inferences": 2 * TEST_N,
        "risk_band_edges": [.2, .8], "risk_bands_calibrated": False,
        "validation_selected_primary_metrics": ev.classification_metrics(truth,
            [int(cpu[r["id"]] or s >= THRESHOLD) for r, s in zip(rows, scores)]),
        "max_new_post_attempts": NEW_POST_LIMIT, "automatic_retries": 0,
        "cloud_resume_allowed": False, "redirects_allowed": False,
        "local_evidence_resume_allowed": True,
        "user_authorized_budget_usd": USER_BUDGET_USD,
        "software_new_forecast_limit_usd": SOFTWARE_NEW_FORECAST_USD,
        "input_price_usd_per_million_tokens": PRICE, "price_checked_date": "2026-10-07",
        "price_source": "https://docs.typesafe.ai/models", "output_tokens_free_at_snapshot": True,
        "reserve_input_tokens_per_post": RESERVE_TOKENS,
        "reserved_new_cost_usd": reserved, "previous_known_cost_estimate_usd": known_previous,
        "reserved_project_known_cost_usd": known_previous + reserved,
        "reference_new_cost_usd": report["billing_estimates"]["new_known_usage_cost_usd"] / 152 * TEST_N,
        "provider_enforced_limit": False, "account_bill_verified": False,
        "billing_caution": "Token forecasts are not an account-level billing cap; no subscription/top-up/tax/AutoDL/other-job costs are included.",
        "privacy": {"only_send": ["masked transcription", "six generated evidence fields", "fixed policy", "pinned model"],
            "never_send": ["image", "gold label", "Qwen final label", "sample ID", "group ID", "path", "CPU prediction"],
            "full_anonymization_guaranteed": False},
        "test_label_use": "Only the final reporting stage computes quality; no tuning or training",
        "training": False, "test_read_at_plan": False, "api_calls_at_plan": 0,
        "source_sha256": hashes}
    return p, json.loads(prep.canonical(frozen)), rows


def cpu_predictions(p, frozen, inputs):
    import joblib
    import numpy as np
    predictions, durations = {}, {}
    for name in MODEL_NAMES:
        artifact = p.cpu / f"{name}.joblib"
        if ev.digest(artifact) != frozen["source_sha256"][str(artifact.resolve())]:
            raise ValueError("Frozen CPU artifact changed")
        bundle = joblib.load(artifact)  # Own, previously trained local artifacts only.
        if (bundle.get("input_modality") != "text_only_ablation"
                or bundle.get("source_split_sha256") != prep.read_json(p.cpu / "validation_report.json")["source_split_sha256"]
                or bundle.get("threshold") != frozen["cpu_thresholds_frozen_from_validation"][name]):
            raise ValueError("CPU bundle provenance/threshold differs from its frozen validation report")
        items, seconds = {}, []
        for row in inputs:
            started = time.perf_counter()
            score = checked_probability(float(cpu_baseline.positive_scores(bundle["classifier"],
                bundle["vectorizer"].transform([row["text"]]))[0]))
            seconds.append(time.perf_counter() - started)
            items[row["id"]] = {"score": score,
                "predicted_label": int(score >= frozen["cpu_thresholds_frozen_from_validation"][name])}
        predictions[name] = items
        durations[name] = {"n": len(seconds), "mean_seconds": float(np.mean(seconds)),
            "p50_seconds": float(np.percentile(seconds, 50)), "p95_seconds": float(np.percentile(seconds, 95)),
            "scope": "batch1 TF-IDF transform + classifier; excludes artifact load, file/network/JSON"}
    return predictions, durations


def prepare_run(p, frozen, validation_rows):
    # Atomic directory ownership and exclusive frozen record block duplicate tests.
    p.out.mkdir(parents=True, mode=0o700, exist_ok=False)
    frozen = {**frozen, "frozen_utc_before_test_read": datetime.now(timezone.utc).isoformat()}
    write_new_json(p.out / "frozen_protocol.json", frozen)
    previous.verify_hashes(frozen["source_sha256"])
    test_path = p.data / "test.jsonl"
    rows = evidence.read_jsonl(test_path)
    if len(rows) != TEST_N or len({r["id"] for r in rows}) != TEST_N:
        raise ValueError("Expected 170 unique prepared test rows")
    val_ids, val_groups = {r["id"] for r in validation_rows}, {r["group_id"] for r in validation_rows}
    inputs, controls = [], []
    for row in rows:
        if (set(row) != {"id", "text", "image", "label", "group_id"}
                or not isinstance(row["id"], str) or not row["id"].startswith("dev:")
                or row["id"] in val_ids or row["group_id"] in val_groups
                or type(row["label"]) is not int or row["label"] not in (0, 1)
                or not isinstance(row["text"], str) or not row["text"].strip()
                or not isinstance(row["group_id"], str) or not row["group_id"]
                or not Path(row["image"]).is_absolute() or not Path(row["image"]).is_file()):
            raise ValueError("Invalid test schema/split/leakage guard")
        inputs.append({k: row[k] for k in ("id", "text", "image", "group_id")})
        controls.append({"id": row["id"], "gold_label_LOCAL_ONLY": row["label"]})
    # Gold labels are isolated; downstream GPU/API stages load only inputs.
    prep.atomic_jsonl(p.out / "inputs_LOCAL_ONLY.jsonl", inputs)
    prep.atomic_jsonl(p.out / "test_controls_DO_NOT_UPLOAD.jsonl", controls)
    del rows, controls
    predictions, latency = cpu_predictions(p, frozen, inputs)
    ev.atomic_json(p.out / "cpu_predictions_LOCAL_ONLY.json", {"models": predictions, "latency": latency})
    files = [test_path, p.out / "frozen_protocol.json", p.out / "inputs_LOCAL_ONLY.jsonl",
             p.out / "test_controls_DO_NOT_UPLOAD.jsonl", p.out / "cpu_predictions_LOCAL_ONLY.json"]
    completion = {"version": VERSION, "test_n": TEST_N, "test_read": True,
        "test_quality_computed": False, "api_calls": 0, "gpu_used": False,
        "prepared_source_sha256": {str(f.resolve()): ev.digest(f) for f in files},
        "test_image_bytes_sha256": prep.object_digest([(r["id"], ev.digest(Path(r["image"]))) for r in inputs])}
    previous.verify_hashes(frozen["source_sha256"])
    ev.atomic_json(p.out / "preparation_complete.json", completion)
    print(f"PREPARED_FROZEN_TEST N=170 CPU_MODELS=4 TEST_QUALITY_NOT_COMPUTED=True API_CALLS=0 GPU=False OUTPUT={p.out}", flush=True)
    return 0


def load_prepared(root):
    p = paths(root)
    frozen = prep.read_json(p.out / "frozen_protocol.json")
    completed = prep.read_json(p.out / "preparation_complete.json")
    if (frozen.get("version") != VERSION or frozen.get("jev_threshold") != THRESHOLD
            or frozen.get("jev_model") != prep.MODEL or frozen.get("jev_policy_sha256") != prep.object_digest(prep.QUESTION)
            or frozen.get("max_new_post_attempts") != NEW_POST_LIMIT
            or frozen.get("user_authorized_budget_usd") != USER_BUDGET_USD
            or completed.get("version") != VERSION or completed.get("test_n") != TEST_N):
        raise ValueError("Frozen protocol changed or preparation incomplete")
    previous.verify_hashes(frozen["source_sha256"])
    previous.verify_hashes(completed["prepared_source_sha256"])
    inputs = evidence.read_jsonl(p.out / "inputs_LOCAL_ONLY.jsonl")
    if len(inputs) != TEST_N or len({r["id"] for r in inputs}) != TEST_N or any("label" in r for r in inputs):
        raise ValueError("Label-free test inputs are invalid")
    if prep.object_digest([(r["id"], ev.digest(Path(r["image"]))) for r in inputs]) != completed["test_image_bytes_sha256"]:
        raise ValueError("Test image bytes changed")
    cpu = prep.read_json(p.out / "cpu_predictions_LOCAL_ONLY.json")
    return p, frozen, inputs, cpu


def evidence_run(root, resume=False):
    p, frozen, inputs, cpu = load_prepared(root)
    if (p.out / "cloud").exists() or (p.out / "final_test_report.json").exists():
        raise ValueError("CLOUD_ALREADY_STARTED: no regeneration after the final experiment")
    checkpoints = {stage: p.out / f"{stage.lower()}.json" for stage in STAGES}
    if any(path.exists() for path in checkpoints.values()) and not resume:
        raise ValueError("LOCAL_OUTPUT_EXISTS: use --resume-local only for unchanged GPU evidence")
    protocol_digest = ev.digest(p.out / "frozen_protocol.json")
    cached = {}
    for stage, path in checkpoints.items():
        data = prep.read_json(path) if path.exists() else {"protocol_sha256": protocol_digest, "stage": stage, "items": {}}
        if data.get("protocol_sha256") != protocol_digest or data.get("stage") != stage:
            raise ValueError("Local evidence provenance changed")
        if not set(data["items"]) <= {r["id"] for r in inputs}:
            raise ValueError("Unexpected evidence IDs")
        for item in data["items"].values():
            expand.check_record(item)
        cached[stage] = data
    lock = p.out / ".evidence_worker.lock"
    write_new_json(lock, {"pid": os.getpid(), "protocol_sha256": protocol_digest})
    try:
        pending = sum(len(inputs) - len(cached[stage]["items"]) for stage in STAGES)
        processor, model = evidence.load_matched_model(p) if pending else (None, None)
        for stage in STAGES:
            state = cached[stage]
            context = model.disable_adapter() if model is not None and stage == "BASE_EVIDENCE" else contextlib.nullcontext()
            with context:
                for index, row in enumerate(inputs, 1):
                    if row["id"] in state["items"]:
                        continue  # Invalid results are recorded, not silently retried.
                    record = ev.infer_one(processor, model, {"image": row["image"], "text": row["text"]},
                                          "image-text", evidence.NEW_TOKENS)
                    expand.check_record(record)
                    state["items"][row["id"]] = record
                    print(f"{stage} [{index}/170] STATUS={record['status']} SEC={record['latency_seconds']:.2f} NO_TEST_METRICS", flush=True)
                    if len(state["items"]) % 5 == 0:
                        ev.atomic_json(checkpoints[stage], state)
            ev.atomic_json(checkpoints[stage], state)
        if any(len(data["items"]) != TEST_N or any(v["status"] != "ok" for v in data["items"].values()) for data in cached.values()):
            print("EVIDENCE_UNRESOLVED_STOP: no API upload or test metrics; preserve outputs", flush=True)
            return 2
        requests, controls = [], []
        for index, row in enumerate(inputs, 1):
            request, _, size = prep.make_request(row, cached["LORA_EVIDENCE"]["items"][row["id"]]["decision"])
            if size > 12000:
                raise ValueError("Prepared request exceeds its frozen privacy/size bound")
            requests.append(request)
            controls.append({"index": index, "local_id_DO_NOT_UPLOAD": row["id"], "request_sha256": prep.object_digest(request)})
        prep.atomic_jsonl(p.out / "requests_LOCAL_ONLY.jsonl", requests)
        prep.atomic_jsonl(p.out / "request_controls_DO_NOT_UPLOAD.jsonl", controls)
        files = [*checkpoints.values(), p.out / "requests_LOCAL_ONLY.jsonl", p.out / "request_controls_DO_NOT_UPLOAD.jsonl"]
        load_prepared(root)
        ev.atomic_json(p.out / "evidence_complete.json", {"protocol_sha256": protocol_digest, "n": TEST_N,
            "requests_sha256": prep.object_digest(requests), "source_sha256": {str(f.resolve()): ev.digest(f) for f in files}})
        print("TEST_EVIDENCE_COMPLETE BASE=170/170 LORA=170/170 API_CALLS=0 TEST_QUALITY_NOT_COMPUTED=True", flush=True)
        return 0
    finally:
        for stage, data in cached.items():
            if data["items"]:
                ev.atomic_json(checkpoints[stage], data)
        lock.unlink()  # Only this worker's exclusively-created lock is removed.


def load_cloud_inputs(root):
    p, frozen, inputs, cpu = load_prepared(root)
    if (p.out / ".evidence_worker.lock").exists():
        raise ValueError("GPU_WORKER_ACTIVE: do not upload while evidence is being written")
    ready = prep.read_json(p.out / "evidence_complete.json")
    if ready.get("n") != TEST_N or ready.get("protocol_sha256") != ev.digest(p.out / "frozen_protocol.json"):
        raise ValueError("Evidence not bound to the frozen protocol")
    previous.verify_hashes(ready["source_sha256"])
    requests = evidence.read_jsonl(p.out / "requests_LOCAL_ONLY.jsonl")
    controls = evidence.read_jsonl(p.out / "request_controls_DO_NOT_UPLOAD.jsonl")
    lora = prep.read_json(p.out / "lora_evidence.json")["items"]
    if len(requests) != TEST_N or len(controls) != TEST_N or prep.object_digest(requests) != ready["requests_sha256"]:
        raise ValueError("Prepared request count/digest changed")
    for index, (row, request, control) in enumerate(zip(inputs, requests, controls), 1):
        rebuilt, _, size = prep.make_request(row, lora[row["id"]]["decision"])
        if request != rebuilt or size > 12000 or control != {
                "index": index, "local_id_DO_NOT_UPLOAD": row["id"], "request_sha256": prep.object_digest(request)}:
            raise ValueError("Payload/privacy/local mapping changed")
    old_requests = evidence.read_jsonl(p.local_validation / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl")
    old_controls = evidence.read_jsonl(p.local_validation / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl")
    old_results = prep.read_json(p.validation / "predictions_LOCAL_ONLY.json")["items"]
    payload_cache = {}
    for request, control in zip(old_requests, old_controls):
        digest = prep.object_digest(request)
        if digest != control["prepared_request_sha256"]:
            raise ValueError("Read-only validation cache changed")
        payload_cache.setdefault(digest, (control["local_id"], old_results[control["local_id"]]))
    unique = len({prep.object_digest(body) for body in requests} - set(payload_cache))
    return p, frozen, inputs, cpu, requests, controls, payload_cache, unique, ready


def cloud_run(root, authorized=False, plan_only=False):
    p, frozen, inputs, cpu, requests, controls, payload_cache, unique, ready = load_cloud_inputs(root)
    output = p.out / "cloud"
    if output.exists():
        raise ValueError("CLOUD_OUTPUT_EXISTS_STOP: no replay or cloud resume")
    print(f"TEST_CLOUD_PLAN N=170 UNIQUE_NEW_POSTS={unique} MAX_NEW_POSTS=170 RESERVED_USD={cost(unique * RESERVE_TOKENS):.8f} SOFTWARE_NEW_LIMIT_USD=0.50 USER_BUDGET_USD=5.00", flush=True)
    if plan_only:
        print("PLAN_ONLY API_CALLS=0 KEY_READ=False OUTPUT_WRITTEN=False TEST_QUALITY_NOT_COMPUTED=True", flush=True)
        return 0
    if not authorized:
        raise ValueError("TEST_CLOUD_AUTHORIZATION_FLAG_REQUIRED")
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key or "\r" in key or "\n" in key or len(key) > 4096:
        raise ValueError("KEY_MISSING_OR_INVALID: enter privately; never paste into chat")
    opener = urllib.request.build_opener(cloud.NoRedirect())
    output.mkdir(mode=0o700, exist_ok=False)
    (output / "started_attempts").mkdir(mode=0o700)
    protocol_hash = ev.digest(p.out / "frozen_protocol.json")
    write_new_json(output / "authorization.json", {"version": VERSION, "protocol_sha256": protocol_hash,
        "authorized_max_new_posts": NEW_POST_LIMIT, "user_budget_usd": USER_BUDGET_USD,
        "software_new_forecast_limit_usd": SOFTWARE_NEW_FORECAST_USD, "automatic_retries": 0,
        "requests_sha256": prep.object_digest(requests), "started_utc": datetime.now(timezone.utc).isoformat()})
    items, started, cache_hits = {}, 0, 0
    try:
        for index, (row, request) in enumerate(zip(inputs, requests), 1):
            previous.verify_hashes(frozen["source_sha256"])
            previous.verify_hashes(ready["source_sha256"])
            digest = prep.object_digest(request)
            if digest in payload_cache:
                from_id, record = payload_cache[digest]
                items[row["id"]] = {**copy.deepcopy(record), "result_origin": "exact_payload_cache",
                    "cached_from_id_LOCAL_ONLY": from_id}
                cache_hits += 1
                ev.atomic_json(output / "predictions_LOCAL_ONLY.json", {"protocol_sha256": protocol_hash, "items": items})
                print(f"TEST_JEV [{index}/170] EXACT_PAYLOAD_CACHE NO_NEW_POST", flush=True)
                continue
            reserved = cost((started + 1) * RESERVE_TOKENS)
            if (started >= NEW_POST_LIMIT or reserved > SOFTWARE_NEW_FORECAST_USD
                    or frozen["previous_known_cost_estimate_usd"] + reserved > USER_BUDGET_USD):
                print("AUTHORIZATION_OR_BUDGET_FORECAST_STOP", flush=True)
                break
            cloud.write_started(output / "started_attempts" / f"{started + 1:03d}.json", {
                "ordinal": started + 1, "row_ordinal": index, "local_id_DO_NOT_UPLOAD": row["id"],
                "request_sha256": digest, "automatic_replay_forbidden": True})
            started += 1
            items[row["id"]] = {"status": "uncertain_started_attempt_requires_review"}
            start = time.perf_counter()
            try:
                result = cloud.post_once(opener, prep.canonical(request).encode("utf-8"), key)
                elapsed = time.perf_counter() - start
                items[row["id"]] = {"status": "ok", **result, "api_roundtrip_seconds": elapsed, "result_origin": "new_api"}
                payload_cache[digest] = (row["id"], items[row["id"]])
                print(f"TEST_JEV [{index}/170] POST={started} SEC={elapsed:.3f} TOKENS={result['input_tokens']} NO_TEST_METRICS", flush=True)
            except Exception as exc:
                safe = f"HTTP_{exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
                items[row["id"]] = {"status": "failed_or_uncertain_no_retry", "error_type": safe, "possibly_billed": True}
                print(f"TEST_CLOUD_STOP ERROR={safe} NO_RETRY; preserve the ledger", flush=True)
                break
            finally:
                ev.atomic_json(output / "predictions_LOCAL_ONLY.json", {"protocol_sha256": protocol_hash, "items": items})
    finally:
        valid = sum(record.get("status") == "ok" for record in items.values())
        new_successes = [r for r in items.values() if r.get("status") == "ok" and r.get("result_origin") == "new_api"]
        summary = {"version": VERSION, "complete": valid == TEST_N, "valid_n": valid,
            "new_post_attempts": started, "new_successes": len(new_successes), "exact_payload_cache_hits": cache_hits,
            "new_known_usage_cost_usd": cost(sum(r["input_tokens"] for r in new_successes)),
            "uncertain_attempts": started - len(new_successes), "automatic_retries": 0,
            "test_quality_computed": False, "provider_enforced_limit": False}
        ev.atomic_json(output / "completion.json", summary)
    print(f"TEST_CLOUD_COMPLETE={summary['complete']} VALID={valid}/170 NEW_POSTS={started} NEW_KNOWN_COST_USD={summary['new_known_usage_cost_usd']:.8f}", flush=True)
    return 0 if summary["complete"] else 2


def report_run(root):
    p, frozen, inputs, cpu, requests, controls, payload_cache, unique, ready = load_cloud_inputs(root)
    summary = prep.read_json(p.out / "cloud/completion.json")
    saved = prep.read_json(p.out / "cloud/predictions_LOCAL_ONLY.json")
    if not summary.get("complete") or summary.get("valid_n") != TEST_N or saved.get("protocol_sha256") != ev.digest(p.out / "frozen_protocol.json"):
        raise ValueError("COMPLETE_TEST_REQUIRED: no quality evaluation on partial cloud results")
    output = p.out / "final_test_report.json"
    report_files = [p.out / "cloud/authorization.json", p.out / "cloud/completion.json", p.out / "cloud/predictions_LOCAL_ONLY.json",
                    *sorted((p.out / "cloud/started_attempts").glob("*.json"))]
    if output.exists():
        result = prep.read_json(output)
        previous.verify_hashes(result["report_source_sha256"])
        print("EXISTING_FROZEN_TEST_REPORT=" + str(output), flush=True)
        return 0
    test_controls = evidence.read_jsonl(p.out / "test_controls_DO_NOT_UPLOAD.jsonl")
    if [r["id"] for r in test_controls] != [r["id"] for r in inputs]:
        raise ValueError("Test controls mapping changed")
    truth = [r["gold_label_LOCAL_ONLY"] for r in test_controls]
    items = saved["items"]
    if set(items) != {r["id"] for r in inputs} or any(r.get("status") != "ok" for r in items.values()):
        raise ValueError("Incomplete or invalid test results")
    scores = [checked_probability(items[r["id"]]["violation_probability"]) for r in inputs]
    predictions = {name: [cpu["models"][name][r["id"]]["predicted_label"] for r in inputs] for name in MODEL_NAMES}
    selected = predictions[frozen["selected_text_model"]]
    local = {stage: prep.read_json(p.out / f"{stage.lower()}.json")["items"] for stage in STAGES}
    for stage, name in zip(STAGES, ("QWEN_BASE_IMAGE_TEXT", "QWEN_LORA_IMAGE_TEXT")):
        predictions[name] = [local[stage][r["id"]]["decision"]["label"] for r in inputs]
        predictions["TEXT_OR_" + name] = [int(a or b) for a, b in zip(selected, predictions[name])]
    predictions["JEV_T040"] = [int(s >= THRESHOLD) for s in scores]
    predictions["JEV_T050_FIXED_CONTROL"] = [int(s >= .5) for s in scores]
    predictions["PRIMARY_TEXT_OR_JEV_T040"] = [int(a or b) for a, b in zip(selected, predictions["JEV_T040"])]
    metrics = {name: ev.classification_metrics(truth, labels) for name, labels in predictions.items()}
    primary = predictions["PRIMARY_TEXT_OR_JEV_T040"]
    paired = {"gained": sum(a != y and b == y for a, b, y in zip(selected, primary, truth)),
              "lost": sum(a == y and b != y for a, b, y in zip(selected, primary, truth))}
    grouped = {}
    for index, row in enumerate(inputs):
        grouped.setdefault(row["group_id"], []).append(index)
    # Reproducible group bootstrap respects the preparation's dependent pairs.
    import random
    rng = random.Random(2063)
    groups = list(grouped.values())
    differences = []
    for _ in range(1000):
        indices = [i for group in rng.choices(groups, k=len(groups)) for i in group]
        y = [truth[i] for i in indices]
        a = ev.classification_metrics(y, [selected[i] for i in indices])["macro_f1"]
        b = ev.classification_metrics(y, [primary[i] for i in indices])["macro_f1"]
        differences.append(b - a)
    differences.sort()
    api_times = [items[r["id"]]["api_roundtrip_seconds"] for r in inputs]
    qwen_times = [local["LORA_EVIDENCE"][r["id"]]["latency_seconds"] for r in inputs]
    negative_indices = [i for i, label in enumerate(selected) if label == 0]
    routes = ["HIGH_PRIORITY_REVIEW" if score >= .8 else "HUMAN_REVIEW"
              if label or score >= .2 or local["LORA_EVIDENCE"][row["id"]]["decision"]["needs_review"] else "LOW_QUEUE"
              for row, label, score in zip(inputs, primary, scores)]
    bands = {}
    for name, lower, upper in (("LOW", 0, .2), ("MEDIUM", .2, .8), ("HIGH", .8, 1.01)):
        indices = [i for i, score in enumerate(scores) if lower <= score < upper]
        bands[name] = {"n": len(indices), "harmful_n": sum(truth[i] for i in indices), "not_calibrated": True}
    result = {"version": VERSION, "scope": "one_frozen_final_test_not_validation_selection", "complete": True,
        "test_n": TEST_N, "test_groups": len(grouped), "training": False, "test_threshold_tuning": False,
        "protocol_sha256": ev.digest(p.out / "frozen_protocol.json"), "primary_rule": frozen["primary_rule"],
        "primary": metrics["PRIMARY_TEXT_OR_JEV_T040"], "fixed_comparators": metrics,
        "paired_primary_vs_selected_text": paired,
        "group_bootstrap_primary_minus_text_macro_f1_95pct": {"low": differences[24], "high": differences[974], "resamples": 1000, "seed": 2063},
        "jev_probability_metrics": {**cloud.probability_metrics(truth, scores), "caution": "Single final test, not proof of operational probability calibration"},
        "billing": {**summary, "previous_known_cost_estimate_usd": frozen["previous_known_cost_estimate_usd"],
            "project_known_cost_estimate_usd": frozen["previous_known_cost_estimate_usd"] + summary["new_known_usage_cost_usd"],
            "user_budget_usd": USER_BUDGET_USD, "provider_enforced_limit": False, "account_bill_verified": False},
        "cpu_latency": cpu["latency"],
        "latency": {"qwen_lora_evidence_mean_seconds": sum(qwen_times) / TEST_N,
            "api_or_cached_roundtrip_mean_seconds": sum(api_times) / TEST_N,
            "reconstructed_dense_qwen_plus_api_mean_seconds": sum(qwen_times + api_times) / TEST_N,
            "caution": "Dense batch1 stage timings; excludes model load and orchestration/I/O. Cached API times are historical. Not a live cascade service benchmark."},
        "cascade_simulation_not_live_benchmark": {"text_positive_gpu_skips": TEST_N - len(negative_indices),
            "text_negative_expensive_branches": len(negative_indices),
            "branch_timing_reconstruction_mean_seconds": cpu["latency"][frozen["selected_text_model"]]["mean_seconds"]
                + sum(qwen_times[i] + api_times[i] for i in negative_indices) / TEST_N,
            "skipped_jev_probabilities_would_be_unavailable_not_fabricated": True},
        "risk_bands_not_calibrated": bands, "routing_counts": dict(Counter(routes)),
        "report_source_sha256": {str(f.resolve()): ev.digest(f) for f in report_files},
        "cautions": ["Primary rule chosen on validation and frozen before test; do not retune after reading this report.",
            "CPU controls are text-only, not same-input traditional multimodal baselines.",
            "Jev sees generated text evidence, not images. Evidence can be wrong.",
            "Binary labels do not supervise severity; risk bands are not calibrated operational risks.",
            "Small benchmark and heuristic grouping; pretrained benchmark contamination cannot be excluded.",
            "No automated content deletion/block/allow or training was performed."]}
    write_new_json(output, result)
    for name, m in metrics.items():
        print(f"FINAL_TEST {name} N=170 ACC={m['accuracy']:.4f} MACRO_F1={m['macro_f1']:.4f} RECALL={m['harmful_recall']:.4f}", flush=True)
    print(f"FROZEN_FINAL_TEST_COMPLETE REPORT={output} NO_MORE_TEST_TUNING", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ev.DEFAULT_ROOT)
    parser.add_argument("--stage", choices=("prepare", "evidence", "cloud", "report"), default="prepare")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume-local", action="store_true")
    parser.add_argument("--authorize-test-cloud", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        p, frozen, val = plan(args.project_root)
        print(f"FROZEN_TEST_PLAN TEST_N=170 PRIMARY=TEXT_OR_JEV_GE_0.40 CPU_MODELS=4 MATCHED_QWEN_INFERENCES=340 JEV_MODEL={prep.MODEL}", flush=True)
        print(f"MAX_NEW_POSTS=170 REFERENCE_NEW_USD={frozen['reference_new_cost_usd']:.8f} RESERVED_NEW_USD={frozen['reserved_new_cost_usd']:.8f} SOFTWARE_NEW_LIMIT_USD=0.50 USER_BUDGET_USD=5.00 NOT_A_PROVIDER_CAP", flush=True)
        if args.plan_only:
            print("PLAN_ONLY TEST_READ=False TRAIN_READ=False API_CALLS=0 KEY_READ=False GPU_USED=False OUTPUT_WRITTEN=False", flush=True)
            return 0
        return prepare_run(p, frozen, val)
    if args.stage == "evidence":
        if args.plan_only:
            p, frozen, rows, cpu = load_prepared(args.project_root)
            print("EVIDENCE_PLAN TEST_N=170 INFERENCES_MAX=340 API_CALLS=0 KEY_READ=False GPU_USED=False OUTPUT_WRITTEN=False", flush=True)
            return 0
        return evidence_run(args.project_root, args.resume_local)
    if args.stage == "cloud":
        return cloud_run(args.project_root, args.authorize_test_cloud, args.plan_only)
    if args.plan_only:
        raise ValueError("Report has no plan mode; it computes final quality only after all 170 results complete")
    return report_run(args.project_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FROZEN_TEST_STOPPED ERROR_TYPE={type(exc).__name__}; preserve outputs and report the stage; no automatic retry", flush=True)
        raise SystemExit(2)
