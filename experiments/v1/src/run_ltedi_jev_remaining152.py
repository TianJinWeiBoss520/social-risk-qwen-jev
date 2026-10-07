#!/usr/bin/env python3
"""One-use, authorized validation expansion: reuse 20; at most 152 new POSTs.

No GPU, training, test access, automatic retries, redirects or cloud resume.
Plan mode does not read a key, call a service or write any output. Only the
prepared transcription/evidence/policy goes to TypeSafe. Output presence
blocks reruns even after an uncertain timeout; preserve all attempt ledgers.
"""

import argparse
import copy
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

import evaluate_ltedi_qwen_base as probe
import prepare_ltedi_jev_pilot as prep
import probe_ltedi_lora_evidence20 as evidence
import expand_ltedi_evidence_val172 as expand
import run_ltedi_jev_cloud20 as cloud


ROOT = probe.DEFAULT_ROOT
VERSION = "ltedi_authorized_jev_remaining152_once_v1"
MAX_NEW_POSTS = 152
PRICE_USD_PER_MTOK = 0.042
RESERVED_TOKENS_PER_POST = 65536
SOFTWARE_FORECAST_LIMIT_USD = 0.50


def cost(tokens):
    return tokens * PRICE_USD_PER_MTOK / 1_000_000


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        if probe.digest(Path(filename)) != expected:
            raise ValueError("SOURCE_CHANGED_STOP: preserve the run; do not upload changed data")


def checked_request_files(directory, rows, records, export):
    requests = evidence.read_jsonl(directory / "jev_requests_LOCAL_ONLY.jsonl")
    controls = evidence.read_jsonl(directory / "request_controls_DO_NOT_UPLOAD.jsonl")
    if (export.get("ready") is not True or export.get("n") != len(rows)
            or export.get("requests_sha256") != prep.object_digest(requests)
            or len(requests) != len(rows)):
        raise ValueError("Request export is incomplete or has a stale digest")
    expected_controls = []
    for index, (row, request) in enumerate(zip(rows, requests), 1):
        record = records[row["id"]]
        expand.check_record(record)
        if record["status"] != "ok":
            raise ValueError("Unresolved evidence is not an uploadable success")
        rebuilt, _, size = prep.make_request(row, record["decision"])
        if request != rebuilt or size > 12000:
            raise ValueError("Upload body differs from privacy-filtered reconstruction")
        if (set(request) != {"model", "state", "questions"}
                or request["model"] != prep.MODEL
                or set(request["state"]) != {"transcription", "perception_hypotheses", "evidence_status"}
                or set(request["state"]["perception_hypotheses"]) != set(prep.FEATURES)
                or request["questions"] != {"policy_violation": prep.QUESTION}):
            raise ValueError("Unexpected payload fields/model/policy")
        expected_controls.append({"request_index": index, "local_id": row["id"],
            "gold_label_LOCAL_ONLY": row["label"],
            "qwen_predicted_label_LOCAL_ONLY": record["decision"]["label"],
            "prepared_request_sha256": prep.object_digest(request)})
    if controls != expected_controls:
        raise ValueError("Local evaluation controls do not bind the exact requests to rows")
    return requests, controls


def plan_expansion(root):
    root = root.resolve()
    out = root / "outputs/jev_lora_remaining152_cloud_v1"
    if out.exists():
        raise ValueError("OUTPUT_EXISTS_STOP: this cloud authorization cannot be rerun or resumed")
    inputs, source, source_plan, rows, remaining, seeds, cache, cpu = expand.prepare_plan(
        SimpleNamespace(project_root=root, resume=True, plan_only=True))
    if (source / ".local_worker.lock").exists():
        raise ValueError("LOCAL_EVIDENCE_WORKER_EXISTS_STOP: wait for the local writer to finish")
    expand.verify_sources(source_plan, rows)
    report = prep.read_json(source / "comparison_report.json")
    if (report.get("version") != expand.VERSION or report.get("test_read") is not False
            or report.get("api_calls") != 0 or report.get("cloud_upload") is not False
            or report.get("sources_unchanged") is not True
            or report.get("existing_jev_results_reused_read_only") != 20):
        raise ValueError("Expected the completed local-only 172-row evidence report")
    for stage, name in zip(expand.STAGES, ("base", "lora")):
        if (set(cache[stage]) != {row["id"] for row in rows}
                or any(record["status"] != "ok" for record in cache[stage].values())
                or report[name] != evidence.stage_report(rows, cache[stage])):
            raise ValueError("Full matched evidence must reproduce for all 172 validation rows")
    lora = cache["LORA_EVIDENCE"]
    full_requests, _ = checked_request_files(source / "lora_evidence", rows, lora,
                                             report["local_request_preparation"]["lora"])
    requests, controls = checked_request_files(source / "remaining152_for_jev", remaining, lora,
                                               report["remaining152_request_preparation"])
    if (len(rows) != 172 or len(remaining) != MAX_NEW_POSTS
            or set(row["id"] for row in remaining) & set(source_plan["pilot_ids_LOCAL_ONLY"])):
        raise ValueError("Authorization is only for the frozen 152-ID complement")
    by_id = {row["id"]: request for row, request in zip(rows, full_requests)}
    if any(by_id[row["id"]] != request for row, request in zip(remaining, requests)):
        raise ValueError("Full/remaining request bodies differ")
    old_dir = root / "outputs/jev_lora_val20_cloud_v1"
    old_results = prep.read_json(old_dir / "predictions_LOCAL_ONLY.json")["items"]
    pilot_controls = evidence.read_jsonl(root / "outputs/ltedi_lora_evidence20_v1/lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl")
    if set(old_results) != set(source_plan["pilot_ids_LOCAL_ONLY"]):
        raise ValueError("Reuse must cover the exact original 20 successful results")
    for control in pilot_controls:
        if prep.object_digest(by_id[control["local_id"]]) != control["prepared_request_sha256"]:
            raise ValueError("An original authorized payload changed; never silently call it again")
    hashes_to_old = {control["prepared_request_sha256"]: control["local_id"] for control in pilot_controls}
    request_hashes = [control["prepared_request_sha256"] for control in controls]
    unique_new_posts = len(set(request_hashes) - set(hashes_to_old))
    reserved = cost(unique_new_posts * RESERVED_TOKENS_PER_POST)
    if unique_new_posts > MAX_NEW_POSTS or reserved > SOFTWARE_FORECAST_LIMIT_USD:
        raise ValueError("Reserved input-token forecast exceeds the software limit; no calls allowed")
    previous_tokens = sum(record["input_tokens"] for record in old_results.values())
    files = [source / "plan.json", source / "comparison_report.json",
             source / "base_evidence/predictions.json", source / "lora_evidence/predictions.json",
             source / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl",
             source / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl",
             source / "remaining152_for_jev/jev_requests_LOCAL_ONLY.jsonl",
             source / "remaining152_for_jev/request_controls_DO_NOT_UPLOAD.jsonl", Path(expand.__file__), Path(__file__)]
    plan = {
        "version": VERSION, "requested_model": prep.MODEL, "endpoint": cloud.ENDPOINT,
        "validation_n": 172, "already_completed_and_reused_n": 20, "remaining_n": MAX_NEW_POSTS,
        "remaining_ids_LOCAL_ONLY": [row["id"] for row in remaining],
        "pilot_ids_LOCAL_ONLY": source_plan["pilot_ids_LOCAL_ONLY"],
        "remaining_labels_LOCAL_ONLY": dict(Counter(row["label"] for row in remaining)),
        "requests_sha256": prep.object_digest(requests),
        "max_new_post_attempts": MAX_NEW_POSTS, "unique_new_post_attempts_planned": unique_new_posts,
        "exact_payload_cache_hits_planned": MAX_NEW_POSTS - unique_new_posts,
        "automatic_retries": 0, "redirects_allowed": False, "resume_allowed": False,
        "request_utf8_bytes_total": sum(len(prep.canonical(request).encode()) for request in requests),
        "price_usd_per_million_input_tokens": PRICE_USD_PER_MTOK,
        "price_source": "https://docs.typesafe.ai/models", "price_checked_date": "2026-10-07",
        "output_tokens_charged_at_snapshot": False,
        "reserved_input_tokens_per_post": RESERVED_TOKENS_PER_POST,
        "conservative_reserved_new_inference_cost_usd": reserved,
        "software_forecast_limit_usd": SOFTWARE_FORECAST_LIMIT_USD,
        "reference_new_cost_at_previous_mean_tokens_usd": cost(previous_tokens / 20 * unique_new_posts),
        "provider_enforced_spending_limit": False, "account_bill_verified": False,
        "billing_caution": "Public-price/token forecasts are not a billing guarantee or provider cap. Subscriptions, minimums, top-ups, taxes, other jobs and AutoDL charges are excluded.",
        "privacy": {"only_send": ["masked transcription", "six generated evidence fields", "fixed policy", "pinned model"],
                    "never_send": ["image", "gold label", "Qwen final label", "sample ID", "group ID", "file path", "CPU prediction"],
                    "full_deidentification_guaranteed": False,
                    "authorization_scope": "Remaining 152 validation texts/evidence only; no test examples"},
        "gpu_used": False, "test_read": False, "train_read": False,
        "cpu_name": source_plan["cpu_name"],
        "cached_qwen_lora_mean_seconds": report["lora"]["latency_seconds"]["mean"],
        "source_sha256": {**source_plan["source_sha256"],
                          **{str(path.resolve()): probe.digest(path) for path in files}},
    }
    plan = json.loads(prep.canonical(plan))
    return out, plan, rows, remaining, requests, old_results, hashes_to_old, cache, cpu


def final_report(plan, rows, items, started, cache, cpu, completed_cache_hits):
    valid = [row for row in rows if items.get(row["id"], {}).get("status") == "ok"]
    truth = [row["label"] for row in valid]
    probabilities = [items[row["id"]]["violation_probability"] for row in valid]
    labels = [int(p >= .5) for p in probabilities]
    lora = cache["LORA_EVIDENCE"]
    successful_new = [record for record in items.values()
                      if record.get("status") == "ok" and record.get("result_origin") == "new_api"]
    new_tokens = sum(record["input_tokens"] for record in successful_new)
    old_ids = set(plan["pilot_ids_LOCAL_ONLY"])
    old_tokens = sum(items[key]["input_tokens"] for key in old_ids)
    new_latency = sorted(record["api_roundtrip_seconds"] for record in successful_new)
    all_latency = [items[row["id"]]["api_roundtrip_seconds"] for row in valid]
    gained = sum(lora[row["id"]]["decision"]["label"] != row["label"] and label == row["label"] for row, label in zip(valid, labels))
    lost = sum(lora[row["id"]]["decision"]["label"] == row["label"] and label != row["label"] for row, label in zip(valid, labels))
    probability = cloud.probability_metrics(truth, probabilities) if valid else None
    if probability:
        probability["caution"] = "Descriptive validation results, not proof of in-domain calibration or final test performance"
    return {
        "version": VERSION, "scope": "full_validation_not_final_test", "complete": len(valid) == 172,
        "validation_n": 172, "valid_results": len(valid), "reused_original_results": 20,
        "new_post_attempts_started": started, "new_api_successes": len(successful_new),
        "additional_exact_payload_cache_hits": completed_cache_hits, "max_authorized_new_posts": MAX_NEW_POSTS,
        "automatic_retries": 0, "test_read": False, "train_read": False, "gpu_used": False,
        "pipeline_accuracy_errors_and_unattempted_counted_wrong": sum(y == label for y, label in zip(truth, labels)) / 172,
        "jev_fixed_threshold": .5,
        "jev_classification_valid_only": probe.classification_metrics(truth, labels) if valid else None,
        "probability_metrics_valid_only": probability,
        "qwen_lora_same_all_rows": evidence.stage_report(rows, lora)["classification_on_valid_outputs_only"],
        "matched_base_same_all_rows": evidence.stage_report(rows, cache["BASE_EVIDENCE"])["classification_on_valid_outputs_only"],
        "text_only_same_all_rows": probe.classification_metrics([row["label"] for row in rows], [cpu[row["id"]] for row in rows]),
        "paired_against_qwen_lora_valid_only": {"n": len(valid), "gained": gained, "lost": lost, "net_correct": gained - lost},
        "unresolved_ids_LOCAL_ONLY": [row["id"] for row in rows if row not in valid],
        "risk_bands_not_calibrated": dict(Counter(items[row["id"]]["risk_band"] for row in valid)),
        "routing_counts": dict(Counter(items[row["id"]]["route"] for row in valid)),
        "new_api_roundtrip_seconds": {"n": len(new_latency), "mean": sum(new_latency) / len(new_latency) if new_latency else None,
            "p50": new_latency[len(new_latency) // 2] if new_latency else None,
            "p95": new_latency[max(0, math.ceil(.95 * len(new_latency)) - 1)] if new_latency else None},
        "reconstructed_cached_qwen_plus_api_mean_seconds": plan["cached_qwen_lora_mean_seconds"] + sum(all_latency) / len(all_latency) if all_latency else None,
        "latency_caution": "Sum of historical/cached Qwen and API timings, including reused responses; not a live end-to-end throughput benchmark.",
        "billing_estimates": {"price_usd_per_million_input_tokens": PRICE_USD_PER_MTOK,
            "new_known_reported_input_tokens": new_tokens, "new_known_usage_cost_usd": cost(new_tokens),
            "new_uncertain_attempts": started - len(successful_new),
            "context_reserved_new_started_cost_usd": cost(started * RESERVED_TOKENS_PER_POST),
            "previous20_known_usage_cost_usd": cost(old_tokens),
            "combined_known_usage_cost_usd": cost(old_tokens + new_tokens),
            "provider_enforced_limit": False, "account_bill_verified": False},
        "cautions": ["Classification threshold 0.5 was fixed before this expanded run, not tuned on its labels.",
            "Risk bands are provisional violation-probability bands, not supervised harm severity.",
            "Jev sees generated textual evidence, not the original image; evidence can be incorrect.",
            "Failed or unattempted results are unresolved, never benign defaults; failed calls may be billed.",
            "Only 172 validation examples, including dependent groups. Freeze policy/model/threshold before final test evaluation.",
            "No automatic deletion/block/allow action was performed. Basic masking is not full anonymization."]}


def run(args):
    out, plan, rows, remaining, requests, old, hashes_to_old, cache, cpu = plan_expansion(ROOT)
    print(f"PLAN VAL=172 REUSED=20 REMAINING=152 MAX_NEW_POSTS={MAX_NEW_POSTS} "
          f"PLANNED_UNIQUE_POSTS={plan['unique_new_post_attempts_planned']} MODEL={prep.MODEL}", flush=True)
    print(f"REFERENCE_NEW_ESTIMATE_USD={plan['reference_new_cost_at_previous_mean_tokens_usd']:.8f} "
          f"RESERVED_NEW_ESTIMATE_USD={plan['conservative_reserved_new_inference_cost_usd']:.8f} "
          f"SOFTWARE_FORECAST_LIMIT_USD={SOFTWARE_FORECAST_LIMIT_USD:.2f} NOT_A_PROVIDER_CAP", flush=True)
    for name, stage in zip(("MATCHED_BASE", "LORA"), expand.STAGES):
        metrics = evidence.stage_report(rows, cache[stage])["classification_on_valid_outputs_only"]
        print(f"{name}_VAL N=172 MACRO_F1={metrics['macro_f1']:.4f} ACCURACY={metrics['accuracy']:.4f} "
              f"HARM_RECALL={metrics['harmful_recall']:.4f}", flush=True)
    print("RETRIES=0 REDIRECTS=False RESUME=False GPU_USED=False TEST_READ=False", flush=True)
    if args.plan_only:
        print("PLAN_ONLY API_CALLS=0 KEY_READ=False NETWORK=False OUTPUT_WRITTEN=False", flush=True)
        return 0
    if not args.authorize_152_cloud_calls:
        raise ValueError("EXPLICIT_AUTHORIZATION_FLAG_REQUIRED: no calls made")
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise ValueError("KEY_MISSING: enter privately in this terminal; never paste it into chat")
    if "\r" in key or "\n" in key or len(key) > 4096:
        raise ValueError("KEY_INVALID: key value will not be printed")
    opener = urllib.request.build_opener(cloud.NoRedirect())
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / "started_attempts").mkdir(mode=0o700)
    plan["user_authorized_max_new_posts"] = MAX_NEW_POSTS
    plan["started_utc"] = datetime.now(timezone.utc).isoformat()
    probe.atomic_json(out / "plan.json", plan)
    items = {key: {**copy.deepcopy(record), "result_origin": "reused_original20"} for key, record in old.items()}
    payload_cache = {digest: (old_id, items[old_id]) for digest, old_id in hashes_to_old.items()}
    started, cache_hits = 0, 0
    lora = cache["LORA_EVIDENCE"]
    try:
        probe.atomic_json(out / "predictions_LOCAL_ONLY.json", {"metadata": plan, "items": items})
        for index, (row, request) in enumerate(zip(remaining, requests), 1):
            verify_hashes(plan["source_sha256"])
            digest = prep.object_digest(request)
            if digest in payload_cache:
                from_id, record = payload_cache[digest]
                items[row["id"]] = {**copy.deepcopy(record), "result_origin": "exact_payload_cache",
                                    "cached_from_id_LOCAL_ONLY": from_id}
                cache_hits += 1
                probe.atomic_json(out / "predictions_LOCAL_ONLY.json", {"metadata": plan, "items": items})
                print(f"JEV_REMAINING [{index}/152] EXACT_PAYLOAD_CACHE NO_NEW_POST", flush=True)
                continue
            if started >= MAX_NEW_POSTS or cost((started + 1) * RESERVED_TOKENS_PER_POST) > SOFTWARE_FORECAST_LIMIT_USD:
                print("AUTHORIZATION_OR_FORECAST_LIMIT_STOP", flush=True)
                break
            # Reserve durably before the first network operation. A killed
            # worker or timeout cannot justify calling this request again.
            cloud.write_started(out / "started_attempts" / f"{started + 1:03d}.json", {
                "ordinal": started + 1, "remaining_row_ordinal": index, "local_id_DO_NOT_UPLOAD": row["id"],
                "request_sha256": digest, "status": "reserved_before_network_attempt",
                "automatic_replay_forbidden": True})
            started += 1
            items[row["id"]] = {"status": "uncertain_started_attempt_requires_review", "route": "HUMAN_REVIEW"}
            start = time.perf_counter()
            try:
                result = cloud.post_once(opener, prep.canonical(request).encode("utf-8"), key)
                elapsed = time.perf_counter() - start
                p = result["violation_probability"]
                band = cloud.risk_band(p)
                route = ("HUMAN_REVIEW" if lora[row["id"]]["decision"]["needs_review"] or band == "MEDIUM"
                         else "HIGH_PRIORITY_REVIEW" if band == "HIGH" else "LOW_QUEUE")
                items[row["id"]] = {"status": "ok", **result, "predicted_label_threshold05": int(p >= .5),
                    "risk_band": band, "route": route, "api_roundtrip_seconds": elapsed, "result_origin": "new_api"}
                payload_cache[digest] = (row["id"], items[row["id"]])
                print(f"JEV_REMAINING [{index}/152] POST={started} P={p:.4f} BAND={band} "
                      f"SEC={elapsed:.3f} INPUT_TOKENS={result['input_tokens']}", flush=True)
            except Exception as exc:
                safe_error = f"HTTP_{exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
                items[row["id"]] = {"status": "failed_or_uncertain_no_retry", "error_type": safe_error,
                                    "route": "HUMAN_REVIEW", "possibly_billed": True}
                print(f"STOPPED NEW_POSTS={started} ERROR={safe_error} NO_RETRY; preserve the ledger", flush=True)
                break
            finally:
                probe.atomic_json(out / "predictions_LOCAL_ONLY.json", {"metadata": plan, "items": items})
    finally:
        report = final_report(plan, rows, items, started, cache, cpu, cache_hits)
        probe.atomic_json(out / "report.json", report)
    print(f"JEV_EXPANSION_COMPLETE={report['complete']} REUSED=20 NEW_POSTS={started} "
          f"VALID={report['valid_results']}/172 NEW_KNOWN_COST_USD={report['billing_estimates']['new_known_usage_cost_usd']:.8f}", flush=True)
    print(f"REPORT={out / 'report.json'}; no automatic rerun/resume", flush=True)
    return 0 if report["complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--authorize-152-cloud-calls", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
