#!/usr/bin/env python3
"""Fixed-policy validation ablation: text-only Jev vs cached Qwen-LoRA + Jev.

Only the 172-row validation manifest is read. No GPU, training or test access.
Reuse all existing multimodal results; make at most 172 new text-only POSTs.
Plan mode is read-only and does not read a key or access the network. Cloud
execution is one-use: no automatic retry, redirect, resume or output overwrite.
This tests an evidence bundle, not the causal effect of the image alone.
"""

import argparse
import copy
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import evaluate_ltedi_qwen_base as ev
import prepare_ltedi_jev_pilot as prep
import expand_ltedi_evidence_val172 as expand
import run_ltedi_jev_cloud20 as cloud
import run_ltedi_jev_remaining152 as previous


VERSION = "ltedi_fixed_policy_text_vs_multimodal_jev_val172_v1"
N = MAX_POSTS = 172
PRIMARY_THRESHOLD = .40
CONTROL_THRESHOLD = .50
PRICE = .042
RESERVE_TOKENS = 65536
NEW_FORECAST_LIMIT_USD = .50
USER_BUDGET_USD = 5.00
BOOTSTRAP_SEED = 2064
BOOTSTRAP_RESAMPLES = 1000
OUTPUT_NAME = "jev_text_vs_multimodal_val172_v1"
PINNED_MODEL = "jev-1.13.0"
PINNED_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def cost(tokens):
    return tokens * PRICE / 1_000_000


def verify_sources(hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("SOURCE_HASHES_MISSING_STOP")
    # Reject before hashing anything. Do not accidentally access a new split
    # just because an unrelated run supplied a hash mapping.
    if any(Path(name).name in {"train.jsonl", "test.jsonl"} for name in hashes):
        raise ValueError("TRAIN_OR_TEST_SOURCE_FORBIDDEN_STOP")
    for name, expected in hashes.items():
        if ev.digest(Path(name)) != expected:
            raise ValueError("SOURCE_CHANGED_STOP")


def check_result(record):
    if not isinstance(record, dict) or record.get("status") != "ok":
        raise ValueError("UNRESOLVED_CACHED_RESULT_STOP")
    validated = cloud.parse_response(prep.canonical({
        "model": record.get("model"),
        "answers": {"policy_violation": {"type": "noul", "noul": record.get("violation_probability")}},
        "usage": {field: record.get(field) for field in ("input_tokens", "output_tokens")},
    }))
    seconds = record.get("api_roundtrip_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("INVALID_CACHED_LATENCY_STOP")
    return validated


def text_request(multimodal_request):
    # Byte-identical policy, model and masked transcription. Only remove the
    # model-generated evidence and its availability metadata. The unchanged
    # policy still mentions perception_hypotheses: disclose this limitation,
    # rather than silently optimize the text arm or alter the cached arm.
    return {
        "model": multimodal_request["model"],
        "questions": copy.deepcopy(multimodal_request["questions"]),
        "state": {"transcription": multimodal_request["state"]["transcription"]},
    }


def plan(root):
    root = root.resolve()
    out = root / "outputs" / OUTPUT_NAME
    if out.exists():
        raise ValueError("OUTPUT_EXISTS_STOP: preserve results and the one-use attempt ledger")
    if prep.MODEL != PINNED_MODEL or cloud.ENDPOINT != PINNED_ENDPOINT:
        raise ValueError("PINNED_MODEL_OR_ENDPOINT_CHANGED_STOP")
    data = root / "data/processed/ltedi_grouped_v1"
    local = root / "outputs/ltedi_evidence_val172_v1"
    old = root / "outputs/jev_lora_remaining152_cloud_v1"
    if (local / ".local_worker.lock").exists():
        raise ValueError("EVIDENCE_WRITER_EXISTS_STOP")
    old_plan = prep.read_json(old / "plan.json")
    saved = prep.read_json(old / "predictions_LOCAL_ONLY.json")
    old_report = prep.read_json(old / "report.json")
    if (old_plan.get("version") != previous.VERSION
            or old_plan.get("requested_model") != PINNED_MODEL
            or old_plan.get("endpoint") != PINNED_ENDPOINT
            or old_plan.get("validation_n") != N
            or old_plan.get("test_read") is not False
            or old_plan.get("train_read") is not False
            or saved.get("metadata") != old_plan
            or old_report.get("version") != previous.VERSION
            or old_report.get("complete") is not True
            or old_report.get("validation_n") != N
            or old_report.get("valid_results") != N
            or old_report.get("new_post_attempts_started") != 152
            or old_report.get("reused_original_results") != 20
            or old_report.get("test_read") is not False
            or old_report.get("jev_fixed_threshold") != CONTROL_THRESHOLD):
        raise ValueError("COMPLETED_172_MULTIMODAL_CACHE_REQUIRED_STOP")
    verify_sources(old_plan["source_sha256"])
    manifest = data / "val.jsonl"
    if old_plan["source_sha256"].get(str(manifest.resolve())) != ev.digest(manifest):
        raise ValueError("VALIDATION_MANIFEST_NOT_BOUND_TO_CACHE_STOP")
    rows = ev.load_validation(data)  # Existence checks, no image decoding.
    local_plan = prep.read_json(local / "plan.json")
    local_report = prep.read_json(local / "comparison_report.json")
    qwen = prep.read_json(local / "lora_evidence/predictions.json")["items"]
    multi = saved["items"]
    ids = [row["id"] for row in rows]
    if (len(rows) != N or set(multi) != set(ids) or set(qwen) != set(ids)
            or local_plan.get("version") != expand.VERSION
            or local_plan.get("validation_ids_LOCAL_ONLY") != ids
            or local_plan.get("modality") != "image-text"
            or local_report.get("version") != expand.VERSION
            or local_report.get("test_read") is not False
            or local_report.get("api_calls") != 0
            or local_report.get("sources_unchanged") is not True):
        raise ValueError("MATCHED_VALIDATION_OR_MULTIMODAL_PROVENANCE_CHANGED_STOP")
    multi_requests, _ = previous.checked_request_files(
        local / "lora_evidence", rows, qwen, local_report["local_request_preparation"]["lora"])
    text_requests = [text_request(body) for body in multi_requests]
    for row, multimodal_body, text_body in zip(rows, multi_requests, text_requests):
        check_result(multi[row["id"]])
        if (text_body["model"] != PINNED_MODEL
                or text_body["questions"] != {"policy_violation": prep.QUESTION}
                or set(text_body["state"]) != {"transcription"}
                or text_body["state"]["transcription"] != prep.mask_text(row["text"])[0]
                or text_body["state"]["transcription"] != multimodal_body["state"]["transcription"]
                or len(prep.canonical(text_body).encode("utf-8")) > 12000):
            raise ValueError("TEXT_PAYLOAD_PRIVACY_OR_MATCHING_CHECK_FAILED_STOP")
    truth = [row["label"] for row in rows]
    scores = [multi[key]["violation_probability"] for key in ids]
    if ev.classification_metrics(truth, [int(p >= CONTROL_THRESHOLD) for p in scores]) != old_report["jev_classification_valid_only"]:
        raise ValueError("CACHED_MULTIMODAL_METRICS_DO_NOT_REPRODUCE_STOP")
    unique = len({prep.object_digest(body) for body in text_requests})
    reserved = cost(unique * RESERVE_TOKENS)
    old_known = old_report["billing_estimates"]["combined_known_usage_cost_usd"]
    if type(old_known) not in (int, float) or not math.isfinite(old_known) or old_known < 0:
        raise ValueError("INVALID_PRIOR_COST_STOP")
    if unique > MAX_POSTS or reserved > NEW_FORECAST_LIMIT_USD or reserved + old_known > USER_BUDGET_USD:
        raise ValueError("BUDGET_FORECAST_STOP")
    files = [manifest, old / "plan.json", old / "report.json", old / "predictions_LOCAL_ONLY.json",
             local / "plan.json", local / "comparison_report.json",
             local / "lora_evidence/predictions.json", local / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl",
             local / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl",
             Path(__file__), Path(ev.__file__), Path(prep.__file__), Path(cloud.__file__),
             Path(previous.__file__), Path(expand.__file__)]
    hashes = {**old_plan["source_sha256"], **{str(f.resolve()): ev.digest(f) for f in files}}
    frozen = {
        "version": VERSION, "validation_n": N, "validation_groups": len({r["group_id"] for r in rows}),
        "ids_LOCAL_ONLY": ids, "group_ids_LOCAL_ONLY": [r["group_id"] for r in rows],
        "labels_LOCAL_ONLY": dict(Counter(truth)),
        "requested_model": PINNED_MODEL, "endpoint": PINNED_ENDPOINT,
        "policy_sha256": prep.object_digest(prep.QUESTION),
        "text_requests_sha256": prep.object_digest(text_requests),
        "multimodal_requests_sha256": prep.object_digest(multi_requests),
        "primary_threshold_both_arms": PRIMARY_THRESHOLD,
        "secondary_fixed_threshold_both_arms": CONTROL_THRESHOLD,
        "primary_quality_metric": "macro_f1_at_0.40",
        "primary_effect_direction": "multimodal_plus_jev_minus_text_only_jev",
        "max_new_post_attempts": MAX_POSTS, "unique_new_post_attempts_planned": unique,
        "multimodal_cached_results_reused": N, "new_qwen_inferences": 0,
        "automatic_retries": 0, "redirects_allowed": False, "resume_allowed": False,
        "source_sha256": hashes,
        "price_usd_per_million_input_tokens": PRICE,
        "price_source": "https://docs.typesafe.ai/models", "price_checked_date": "2026-10-07",
        "reserved_new_cost_usd": reserved, "software_new_forecast_limit_usd": NEW_FORECAST_LIMIT_USD,
        "user_budget_usd": USER_BUDGET_USD, "previous_validation_known_cost_usd": old_known,
        "provider_enforced_limit": False, "account_bill_verified": False,
        "billing_caution": "New-run forecast, not an account spending cap. Other calls, subscriptions, taxes, minimums and AutoDL costs are excluded.",
        "only_upload": ["masked transcription", "unchanged policy", "pinned model"],
        "never_upload": ["image", "sample ID", "group ID", "path", "gold label", "Qwen label", "Qwen evidence"],
        "full_deidentification_guaranteed": False,
        "gpu_used": False, "test_read": False, "train_read": False,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "interpretation": "Development-validation evidence-removal ablation; not new final-test evidence or isolated image causality.",
        "policy_caution": "Unchanged historical policy mentions Qwen/perception_hypotheses, absent in the text arm. This is not an optimized text-only prompt benchmark.",
    }
    # Large unchanged weights need checking once before/after the run. Check
    # the small frozen data/cache/policy files before every individual POST.
    fast_hashes = {str(f.resolve()): hashes[str(f.resolve())] for f in files}
    return out, frozen, rows, text_requests, multi, qwen, fast_hashes


def latency(values):
    ordered = sorted(values)
    return {"n": len(ordered), "mean_seconds": sum(ordered) / len(ordered) if ordered else None,
            "p50_seconds": ordered[len(ordered) // 2] if ordered else None,
            "p95_seconds": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)] if ordered else None}


def group_intervals(rows, text_labels, multimodal_labels):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["group_id"]].append(index)
    blocks = list(groups.values())
    rng = random.Random(BOOTSTRAP_SEED)
    differences = {name: [] for name in ("macro_f1", "accuracy", "harmful_recall")}
    for _ in range(BOOTSTRAP_RESAMPLES):
        selected = [i for _ in blocks for i in rng.choice(blocks)]
        truth = [rows[i]["label"] for i in selected]
        a = ev.classification_metrics(truth, [text_labels[i] for i in selected])
        b = ev.classification_metrics(truth, [multimodal_labels[i] for i in selected])
        for name in differences:
            differences[name].append(b[name] - a[name])
    result = {}
    for name, values in differences.items():
        ordered = sorted(values)
        lo, hi = ordered[24], ordered[974]
        result[name] = {"low": lo, "high": hi, "includes_zero": lo <= 0 <= hi}
    return {"direction": "multimodal_minus_text", "group_count": len(blocks),
            "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
            "percentile_95pct": result,
            "caution": "Paired group bootstrap; small, previously used development validation set, not confirmatory evidence."}


def build_report(frozen, rows, text, multi, qwen, started, cached):
    successful = [r for r in text.values() if r.get("status") == "ok" and r.get("result_origin") == "new_api"]
    valid = sum(text.get(r["id"], {}).get("status") == "ok" for r in rows)
    report = {
        "version": VERSION, "complete": valid == N, "scope": "validation_ablation_not_final_test",
        "validation_n": N, "valid_text_results": valid, "reused_multimodal_results": N,
        "new_post_attempts_started": started, "new_api_successes": len(successful),
        "exact_text_payload_cache_hits": cached, "automatic_retries": 0,
        "gpu_used": False, "train_read": False, "test_read": False,
        "billing": {"new_known_input_tokens": sum(r["input_tokens"] for r in successful),
                    "new_known_usage_cost_usd": cost(sum(r["input_tokens"] for r in successful)),
                    "failed_or_uncertain_attempts": started - len(successful),
                    "reserved_started_attempts_usd": cost(started * RESERVE_TOKENS),
                    "software_forecast_limit_usd": NEW_FORECAST_LIMIT_USD,
                    "provider_enforced_limit": False, "account_bill_verified": False},
        "unresolved_ids_LOCAL_ONLY": [r["id"] for r in rows if text.get(r["id"], {}).get("status") != "ok"],
        "cautions": [frozen["interpretation"], frozen["policy_caution"], frozen["billing_caution"],
            "Generated evidence is not ground truth. This does not separate image contribution from Qwen reasoning/extra text.",
            "Cached multimodal calls and new text calls are not contemporaneous; same pinned model does not rule out service variability.",
            "No threshold sweep, rule optimization, retraining or final-test retuning is performed.",
            "Probabilities and 0.2/0.8 bands are not proven calibrated in-domain risks or severity labels.",
            "Basic redaction does not guarantee removal of all personal information. No automatic deletion or moderation action."],
    }
    # Never show a selected-success accuracy after a network failure.
    if not report["complete"]:
        report["quality_metrics_computed"] = False
        return report
    truth = [r["label"] for r in rows]
    a = [text[r["id"]]["violation_probability"] for r in rows]
    b = [multi[r["id"]]["violation_probability"] for r in rows]
    comparisons = {}
    for name, threshold in (("primary_t040", PRIMARY_THRESHOLD), ("fixed_control_t050", CONTROL_THRESHOLD)):
        pa, pb = [int(p >= threshold) for p in a], [int(p >= threshold) for p in b]
        ma, mb = ev.classification_metrics(truth, pa), ev.classification_metrics(truth, pb)
        gained = sum(x != y and z == y for y, x, z in zip(truth, pa, pb))
        lost = sum(x == y and z != y for y, x, z in zip(truth, pa, pb))
        comparisons[name] = {
            "threshold_both_arms": threshold, "pure_jev_text_only": ma, "qwen_lora_image_text_plus_jev": mb,
            "multimodal_minus_text": {key: mb[key] - ma[key] for key in (
                "accuracy", "macro_f1", "harmful_precision", "harmful_recall", "harmful_f1")},
            "paired": {"gained": gained, "lost": lost, "net_correct": gained - lost},
        }
        if name == "primary_t040":
            comparisons[name]["group_bootstrap"] = group_intervals(rows, pa, pb)
    text_probability = cloud.probability_metrics(truth, a)
    multi_probability = cloud.probability_metrics(truth, b)
    for metrics in (text_probability, multi_probability):
        metrics["caution"] = "Descriptive development validation; not proof of operational probability calibration."
    multi_times = [multi[r["id"]]["api_roundtrip_seconds"] for r in rows]
    qwen_times = [qwen[r["id"]]["latency_seconds"] for r in rows]
    primary = comparisons["primary_t040"]
    ci = primary["group_bootstrap"]["percentile_95pct"]["macro_f1"]
    delta = primary["multimodal_minus_text"]["macro_f1"]
    winner = "multimodal_plus_jev" if delta > 0 else "text_only_jev" if delta < 0 else "tie"
    report.update({
        "quality_metrics_computed": True, "fixed_comparisons": comparisons,
        "probability_metrics": {"pure_jev_text_only": text_probability, "multimodal_plus_jev": multi_probability},
        "primary_observation": {"higher_observed_macro_f1": winner, "difference": delta,
            "95pct_group_interval_includes_zero": ci["includes_zero"],
            "stable_superiority_proven": False},
        "latency": {"new_text_jev_http_roundtrip": latency([r["api_roundtrip_seconds"] for r in successful]),
            "historical_multimodal_jev_http_roundtrip": latency(multi_times),
            "historical_qwen_evidence": latency(qwen_times),
            "reconstructed_multimodal_pipeline": latency([x + y for x, y in zip(qwen_times, multi_times)]),
            "caution": "Serial stage timing, not throughput. Historical API results and Qwen timings are cached; no live paired end-to-end benchmark, model-load cost excluded."},
        "multimodal_cached_input_cost_at_price_snapshot_usd": cost(sum(multi[r["id"]]["input_tokens"] for r in rows)),
    })
    return report


def run(args):
    out, frozen, rows, requests, multi, qwen, fast_hashes = plan(args.project_root)
    print(f"PLAN VAL={N} REUSE_MULTIMODAL={N} NEW_TEXT_POSTS_MAX={MAX_POSTS} "
          f"UNIQUE_TEXT_POSTS={frozen['unique_new_post_attempts_planned']} MODEL={PINNED_MODEL}", flush=True)
    print(f"PRIMARY_T=0.40 CONTROL_T=0.50 RESERVED_NEW_USD={frozen['reserved_new_cost_usd']:.8f} "
          "SOFTWARE_NEW_LIMIT_USD=0.50 USER_BUDGET_USD=5.00 NOT_A_PROVIDER_CAP", flush=True)
    print("FIXED_POLICY_EVIDENCE_REMOVAL; NOT_IMAGE_ONLY_CAUSALITY; GPU=False TEST_READ=False RETRIES=0", flush=True)
    if args.plan_only:
        print("PLAN_ONLY API_CALLS=0 KEY_READ=False NETWORK=False OUTPUT_WRITTEN=False", flush=True)
        return 0
    if not args.authorize_text_cloud_calls:
        raise ValueError("EXPLICIT_CLOUD_AUTHORIZATION_FLAG_REQUIRED_STOP")
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise ValueError("KEY_MISSING_STOP: enter privately; do not send it to chat")
    if "\r" in key or "\n" in key or len(key) > 4096:
        raise ValueError("INVALID_PRIVATE_KEY_STOP")
    opener = urllib.request.build_opener(cloud.NoRedirect())
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / "started_attempts").mkdir(mode=0o700)
    frozen["started_utc"] = datetime.now(timezone.utc).isoformat()
    cloud.write_started(out / "frozen_protocol.json", frozen)
    controls = [{"request_index": i, "local_id": r["id"], "group_LOCAL_ONLY": r["group_id"],
                 "gold_label_LOCAL_ONLY": r["label"], "request_sha256": prep.object_digest(body)}
                for i, (r, body) in enumerate(zip(rows, requests), 1)]
    prep.atomic_jsonl(out / "text_requests_LOCAL_ONLY.jsonl", requests)
    prep.atomic_jsonl(out / "controls_DO_NOT_UPLOAD.jsonl", controls)
    items, payload_cache = {}, {}
    started = cached = 0
    checkpoint = out / "text_predictions_LOCAL_ONLY.json"
    ev.atomic_json(checkpoint, {"metadata": frozen, "items": items})
    try:
        for index, (row, body) in enumerate(zip(rows, requests), 1):
            verify_sources(fast_hashes)
            request_hash = prep.object_digest(body)
            if request_hash in payload_cache:
                items[row["id"]] = {**copy.deepcopy(payload_cache[request_hash]), "result_origin": "exact_payload_cache"}
                cached += 1
                ev.atomic_json(checkpoint, {"metadata": frozen, "items": items})
                print(f"TEXT_JEV [{index}/{N}] EXACT_PAYLOAD_CACHE NO_NEW_POST", flush=True)
                continue
            if started >= MAX_POSTS or cost((started + 1) * RESERVE_TOKENS) > NEW_FORECAST_LIMIT_USD:
                raise ValueError("NEW_RUN_BUDGET_OR_CALL_LIMIT_STOP")
            cloud.write_started(out / "started_attempts" / f"{started + 1:03d}.json", {
                "ordinal": started + 1, "local_id_DO_NOT_UPLOAD": row["id"], "request_sha256": request_hash,
                "automatic_replay_forbidden": True, "status": "reserved_before_network_attempt"})
            started += 1
            items[row["id"]] = {"status": "uncertain_started_attempt_requires_review", "possibly_billed": True}
            ev.atomic_json(checkpoint, {"metadata": frozen, "items": items})
            start = time.perf_counter()
            try:
                result = cloud.post_once(opener, prep.canonical(body).encode("utf-8"), key)
                record = {"status": "ok", **result, "api_roundtrip_seconds": time.perf_counter() - start,
                          "request_sha256": request_hash, "result_origin": "new_api"}
                check_result(record)
                items[row["id"]] = record
                payload_cache[request_hash] = record
                print(f"TEXT_JEV [{index}/{N}] POST={started} SEC={record['api_roundtrip_seconds']:.3f} "
                      f"INPUT_TOKENS={result['input_tokens']}", flush=True)
            except Exception as exc:
                safe = f"HTTP_{exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
                items[row["id"]] = {"status": "failed_or_uncertain_no_retry", "error_type": safe, "possibly_billed": True}
                print(f"CLOUD_STOP ERROR_TYPE={safe} NEW_POSTS={started} NO_RETRY; preserve the ledger", flush=True)
                break
            finally:
                ev.atomic_json(checkpoint, {"metadata": frozen, "items": items})
        verify_sources(frozen["source_sha256"])
        report = build_report(frozen, rows, items, multi, qwen, started, cached)
        cloud.write_started(out / "comparison_report.json", report)
    except BaseException:
        # Safe recovery artifact only, no accuracy computed on partial data.
        ev.atomic_json(out / "interrupted_LOCAL_ONLY.json", {
            "complete": False, "new_post_attempts_started": started, "automatic_replay_forbidden": True,
            "quality_metrics_computed": False, "preserve_outputs": True})
        raise
    print(f"COMPARISON_COMPLETE={report['complete']} TEXT_VALID={report['valid_text_results']}/{N} "
          f"NEW_KNOWN_USD={report['billing']['new_known_usage_cost_usd']:.8f}", flush=True)
    if report["complete"]:
        for name, pair in report["fixed_comparisons"].items():
            for arm in ("pure_jev_text_only", "qwen_lora_image_text_plus_jev"):
                m = pair[arm]
                print(f"VAL {name} {arm} ACC={m['accuracy']:.4f} MACRO_F1={m['macro_f1']:.4f} "
                      f"PRECISION={m['harmful_precision']:.4f} RECALL={m['harmful_recall']:.4f}", flush=True)
            print(f"PAIRED {name} {prep.canonical(pair['paired'])}", flush=True)
    print(f"REPORT={out / 'comparison_report.json'} NO_AUTOMATIC_RERUN_OR_RESUME", flush=True)
    return 0 if report["complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ev.DEFAULT_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--authorize-text-cloud-calls", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # No raw HTTP bodies, credentials or exception messages in logs.
        print(f"TEXT_VS_MULTIMODAL_STOP ERROR_TYPE={type(exc).__name__}; preserve outputs; no retry", flush=True)
        raise SystemExit(2)
