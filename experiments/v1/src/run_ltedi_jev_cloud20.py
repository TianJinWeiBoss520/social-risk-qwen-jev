#!/usr/bin/env python3
"""ONE-USE authorized Jev pilot: at most 20 POSTs, no retries/redirects/resume.

Uses already-generated LoRA evidence, not images or Qwen inference. Plan mode
does not read an API key, access the network, import GPU packages, or write files.
Only fixed request bodies go to TypeSafe; labels/IDs/paths stay local. Cost
estimates are NOT a provider-enforced account spending limit or billing promise.
"""

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import evaluate_ltedi_qwen_base as probe
import prepare_ltedi_jev_pilot as prep
import probe_ltedi_lora_evidence20 as evidence


ROOT = probe.DEFAULT_ROOT
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
VERSION = "ltedi_authorized_jev_lora_val20_once_v1"
EXPECTED_REQUESTS = "f4f9413fef75f45b0f193bd342771f765372b67023f2d96433d38b45df903004"
MAX_CALLS = 20
PRICE_USD_PER_MTOK = 0.042
# 64k interpreted generously as 65,536; reserve the WHOLE documented context
# per call, not a guessed Chinese tokenization of our much shorter requests.
RESERVED_INPUT_TOKENS_PER_CALL = 65536
SOFTWARE_ESTIMATE_BUDGET_USD = 0.10
MAX_RESPONSE_BYTES = 65536
TIMEOUT_SECONDS = 30


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line, object_pairs_hook=probe.unique_object) for line in stream if line.strip()]


def quoted_cost(tokens):
    return tokens * PRICE_USD_PER_MTOK / 1_000_000


def plan_pilot(root):
    root = root.resolve()
    source = root / "outputs/ltedi_lora_evidence20_v1"
    out = root / "outputs/jev_lora_val20_cloud_v1"
    if out.exists():
        raise ValueError("OUTPUT_EXISTS_STOP: one-use authorization ledger exists; NEVER delete it to replay requests")
    comparison = prep.read_json(source / "comparison_report.json")
    source_plan = prep.read_json(source / "plan.json")
    if (comparison.get("version") != evidence.VERSION
            or comparison.get("test_read") is not False
            or comparison.get("cloud_upload") is not False
            or comparison.get("api_calls") != 0
            or comparison.get("sources_unchanged") is not True
            or comparison["lora_request_preparation"].get("ready") is not True
            or comparison["lora_request_preparation"].get("n") != MAX_CALLS
            or comparison["lora_request_preparation"].get("requests_sha256") != EXPECTED_REQUESTS):
        raise ValueError("Expected the completed, exact 20-row LoRA evidence preparation")
    if (source_plan.get("version") != evidence.VERSION
            or source_plan.get("test_read") is not False
            or source_plan.get("train_read") is not False
            or source_plan.get("selected_n") != MAX_CALLS
            or source_plan.get("api_calls") != 0
            or source_plan.get("modality") != "image-text"):
        raise ValueError("Unexpected upstream inference plan")
    # The prior matched probe recorded only val/source/cache/adapter/config/code
    # files. Explicitly refuse train/test manifests or any other file kind.
    allowed_names = {"val.jsonl", "predictions.json", "preparation_report.json", "prepared_requests.jsonl",
                     "local_controls_DO_NOT_UPLOAD.jsonl", "validation_report.json", "config.json",
                     "adapter_model.safetensors", "adapter_config.json", "training_metadata.json",
                     "evaluate_ltedi_qwen_base.py", "prepare_ltedi_jev_pilot.py", "probe_ltedi_lora_evidence20.py"}
    for filename, expected in source_plan["source_sha256"].items():
        path = Path(filename)
        if path.name not in allowed_names and path.name != "tfidf_logreg_c4_validation_predictions.jsonl":
            raise ValueError("Unexpected upstream provenance file; no unbounded file reads allowed")
        if probe.digest(path) != expected:
            raise ValueError("Upstream source changed since the matched evidence probe")

    data = root / "data/processed/ltedi_grouped_v1"
    all_rows = probe.load_validation(data)  # Never train.jsonl or test.jsonl.
    identity = source_plan["frozen_preparation_identity"]
    if (identity["source_validation_sha256"] != probe.digest(data / "val.jsonl")
            or identity.get("seed") != 2062):
        raise ValueError("Validation source/selection changed")
    rows = prep.select_rows(all_rows, MAX_CALLS, identity["seed"])
    if [row["id"] for row in rows] != source_plan["selected_ids_LOCAL_ONLY"]:
        raise ValueError("Do not redraw the authorized 20-row selection")
    cpu_name, cpu = probe.load_cpu_predictions(root / "outputs/cpu_text_baselines_v1", data, all_rows)
    predictions = prep.read_json(source / "lora_evidence/predictions.json")
    if predictions.get("metadata") != {**source_plan, "stage": "LORA_EVIDENCE", "adapter_active": True}:
        raise ValueError("LoRA evidence metadata changed")
    if set(predictions["items"]) != {row["id"] for row in rows}:
        raise ValueError("Evidence must cover the entire fixed pilot")
    requests = read_jsonl(source / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl")
    controls = read_jsonl(source / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl")
    if len(requests) != MAX_CALLS or prep.object_digest(requests) != EXPECTED_REQUESTS:
        raise ValueError("AUTHORIZED_PAYLOAD_MISMATCH: no changed/extra requests permitted")
    expected_controls, encoded, decisions, latencies = [], [], {}, []
    for index, (row, request) in enumerate(zip(rows, requests), 1):
        record = predictions["items"][row["id"]]
        if record.get("status") != "ok":
            raise ValueError("Unresolved LoRA evidence cannot be uploaded")
        decision = probe.parse_decision(prep.canonical(record["decision"]), "image-text")
        rebuilt, _, size = prep.make_request(row, decision)
        if request != rebuilt or size > 12000 or set(request) != {"model", "state", "questions"}:
            raise ValueError("Request is not the exact privacy-filtered body for this evidence")
        decisions[row["id"]] = decision
        latencies.append(record["latency_seconds"])
        encoded.append(prep.canonical(request).encode("utf-8"))
        expected_controls.append({"request_index": index, "local_id": row["id"],
                                  "gold_label_LOCAL_ONLY": row["label"],
                                  "qwen_predicted_label_LOCAL_ONLY": decision["label"],
                                  "prepared_request_sha256": prep.object_digest(request)})
    if controls != expected_controls:
        raise ValueError("Local evaluation controls changed")
    reservation = quoted_cost(MAX_CALLS * RESERVED_INPUT_TOKENS_PER_CALL)
    if reservation > SOFTWARE_ESTIMATE_BUDGET_USD:
        raise ValueError("Conservative reservation exceeds software estimate budget; no calls allowed")
    files = [source / "comparison_report.json", source / "plan.json",
             source / "lora_evidence/predictions.json", source / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl",
             source / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl", data / "val.jsonl",
             Path(probe.__file__), Path(prep.__file__), Path(evidence.__file__), Path(__file__)]
    plan = {"version": VERSION, "requested_model": prep.MODEL, "endpoint": ENDPOINT,
            "selected_n": MAX_CALLS, "selected_labels_LOCAL_ONLY": dict(Counter(row["label"] for row in rows)),
            "requests_sha256": EXPECTED_REQUESTS, "request_utf8_bytes_total": sum(map(len, encoded)),
            "request_utf8_bytes_max": max(map(len, encoded)), "max_post_attempts": MAX_CALLS,
            "automatic_retries": 0, "redirects_allowed": False, "resume_allowed": False,
            "price_snapshot_usd_per_million_input_tokens": PRICE_USD_PER_MTOK,
            "price_source": "https://docs.typesafe.ai/models", "price_checked_date": "2026-10-07",
            "output_tokens_charged_at_snapshot": False,
            "reserved_input_tokens_per_call": RESERVED_INPUT_TOKENS_PER_CALL,
            "conservative_reserved_inference_cost_usd": reservation,
            "software_estimate_budget_usd": SOFTWARE_ESTIMATE_BUDGET_USD,
            "provider_enforced_spending_limit": False,
            "billing_caution": "Price/context estimates are not a financial guarantee. Account subscriptions, minimum charges, auto top-ups, other calls, taxes and AutoDL rental are excluded.",
            "only_send": ["transcription", "six model-generated evidence features", "fixed question and pinned model"],
            "never_send": ["image", "gold label", "Qwen final label", "sample ID", "group ID", "file path", "CPU prediction"],
            "privacy_caution": "Basic masking is not full anonymization; user authorized these 20 texts/evidence after this warning.",
            "test_read": False, "train_read": False, "gpu_used": False,
            "source_sha256": {str(path.resolve()): probe.digest(path) for path in files},
            "cpu_name": cpu_name, "cached_qwen_mean_seconds": sum(latencies) / len(latencies)}
    return out, plan, rows, encoded, decisions, cpu


def reject_constant(value):
    raise ValueError("Nonfinite JSON number is not allowed")


def parse_response(raw):
    result = json.loads(raw, object_pairs_hook=probe.unique_object, parse_constant=reject_constant)
    if result.get("model") != prep.MODEL or set(result.get("answers", {})) != {"policy_violation"}:
        raise ValueError("Response model/question does not match the pinned request")
    answer = result["answers"]["policy_violation"]
    probability = answer.get("noul")
    if (answer.get("type") != "noul" or type(probability) not in (int, float)
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("Expected a finite violation probability in [0,1]")
    usage = result["usage"]
    if any(type(usage.get(field)) is not int or not 0 <= usage[field] <= RESERVED_INPUT_TOKENS_PER_CALL
           for field in ("input_tokens", "output_tokens")) or usage["input_tokens"] == 0:
        raise ValueError("Missing/invalid token usage; stop instead of assuming free calls")
    # Save only allowlisted output; raw error/response strings are never logged.
    return {"model": prep.MODEL, "violation_probability": float(probability),
            "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}


def post_once(opener, body, key):
    request = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"})
    with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
        if response.status != 200:
            raise ValueError("Unexpected HTTP success status")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Response exceeds the local safety limit")
    return parse_response(raw)


def risk_band(probability):
    return "LOW" if probability < .2 else "MEDIUM" if probability < .8 else "HIGH"


def probability_metrics(truth, probabilities):
    positive = [p for y, p in zip(truth, probabilities) if y == 1]
    negative = [p for y, p in zip(truth, probabilities) if y == 0]
    auc = sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative)) if positive and negative else None
    ranked = sorted(zip(probabilities, truth), reverse=True)
    ap, tp, count, index = 0.0, 0, 0, 0
    while index < len(ranked):
        score = ranked[index][0]
        group = []
        while index < len(ranked) and ranked[index][0] == score:
            group.append(ranked[index][1])
            index += 1
        gained = sum(group)
        tp += gained
        count += len(group)
        ap += (gained / len(positive)) * (tp / count) if positive else 0
    bounded = [min(1 - 1e-15, max(1e-15, p)) for p in probabilities]
    return {"roc_auc": auc, "pr_auc_average_precision": ap if positive else None,
            "brier_score": sum((p - y) ** 2 for y, p in zip(truth, probabilities)) / len(truth),
            "log_loss": -sum(y * math.log(p) + (1-y) * math.log(1-p) for y, p in zip(truth, bounded)) / len(truth),
            "caution": "Descriptive 20-row pilot only; not proof of in-domain calibration"}


def make_report(plan, rows, items, decisions, cpu):
    valid = [row for row in rows if items.get(row["id"], {}).get("status") == "ok"]
    truth = [row["label"] for row in valid]
    probabilities = [items[row["id"]]["violation_probability"] for row in valid]
    labels = [int(p >= .5) for p in probabilities]  # Fixed beforehand, not tuned on this pilot.
    correct = sum(y == p for y, p in zip(truth, labels))
    started = len(items)
    known_tokens = sum(items[row["id"]]["input_tokens"] for row in valid)
    latency = sorted(items[row["id"]]["api_roundtrip_seconds"] for row in valid)
    gained = sum(decisions[row["id"]]["label"] != row["label"] and p == row["label"] for row, p in zip(valid, labels))
    lost = sum(decisions[row["id"]]["label"] == row["label"] and p != row["label"] for row, p in zip(valid, labels))
    return {"version": VERSION, "complete": len(valid) == len(rows), "selected_n": len(rows),
            "api_post_attempts_started": started, "valid_results": len(valid), "automatic_retries": 0,
            "cloud_upload": started > 0, "test_read": False, "gpu_used": False,
            "pipeline_accuracy_errors_and_unattempted_counted_wrong": correct / len(rows),
            "jev_fixed_threshold": .5, "jev_classification_valid_only": probe.classification_metrics(truth, labels) if valid else None,
            "probability_metrics_valid_only": probability_metrics(truth, probabilities) if valid else None,
            "qwen_lora_same_all_rows": probe.classification_metrics([r["label"] for r in rows], [decisions[r["id"]]["label"] for r in rows]),
            "text_only_same_all_rows": probe.classification_metrics([r["label"] for r in rows], [cpu[r["id"]] for r in rows]),
            "paired_against_qwen_valid_only": {"n": len(valid), "gained": gained, "lost": lost, "net_correct": gained-lost},
            "unresolved_ids_LOCAL_ONLY": [r["id"] for r in rows if r not in valid],
            "risk_bands_not_calibrated": dict(Counter(items[r["id"]]["risk_band"] for r in valid)),
            "human_review_routes": sum(items[r["id"]]["route"] == "HUMAN_REVIEW" for r in valid),
            "api_roundtrip_seconds": {"mean": sum(latency)/len(latency) if latency else None,
                                      "p50": latency[len(latency)//2] if latency else None,
                                      "p95": latency[max(0, math.ceil(len(latency)*.95)-1)] if latency else None},
            "reconstructed_cached_qwen_plus_api_mean_seconds": plan["cached_qwen_mean_seconds"] + sum(latency)/len(latency) if latency else None,
            "latency_caution": "API timing is measured; sum with earlier Qwen timing is a reconstruction, not a live end-to-end benchmark.",
            "billing_estimates": {"price_usd_per_million_input_tokens": PRICE_USD_PER_MTOK,
                                  "known_reported_input_tokens": known_tokens, "known_usage_cost_usd": quoted_cost(known_tokens),
                                  "uncertain_attempts": started-len(valid),
                                  "context_reserved_cost_started_attempts_usd": quoted_cost(started*RESERVED_INPUT_TOKENS_PER_CALL),
                                  "provider_enforced_limit": False, "account_bill_verified": False},
            "cautions": ["20 fixed examples with 3 positives; functionality pilot, not final performance.",
                         "Violation probability is not severity; risk-band thresholds are provisional, not calibrated.",
                         "Jev did not see the image; upstream evidence can be wrong. needs_review routes to human review.",
                         "Invalid API results are unresolved, never defaulted to benign. Failed calls may still be billed.",
                         "No automatic content deletion or unreviewed block/allow action was performed."]}


def write_started(path, record):
    # Durable before sending: a crash/timeout cannot justify replaying a request.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(prep.canonical(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    out, plan, rows, bodies, decisions, cpu = plan_pilot(ROOT)
    print(f"PLAN N=20 MODEL={prep.MODEL} MAX_POSTS=20 RETRIES=0 GPU_USED=False TEST_READ=False", flush=True)
    print(f"RESERVED_ESTIMATE_USD={plan['conservative_reserved_inference_cost_usd']:.8f} "
          f"SOFTWARE_ESTIMATE_BUDGET_USD={SOFTWARE_ESTIMATE_BUDGET_USD:.2f} NOT_A_PROVIDER_BILLING_CAP", flush=True)
    if args.plan_only:
        print("PLAN_ONLY API_CALLS=0; no key read, network access or output written", flush=True)
        return 0
    if not args.authorize_20_cloud_calls:
        raise ValueError("EXPLICIT_AUTHORIZATION_FLAG_REQUIRED; no calls made")
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise ValueError("KEY_MISSING: enter it privately in this terminal; never paste it into chat")
    if "\r" in key or "\n" in key or len(key) > 4096:
        raise ValueError("KEY_INVALID; key value will not be printed")
    opener = urllib.request.build_opener(NoRedirect())  # Direct HTTP, NOT a retrying SDK.
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / "started_attempts").mkdir(mode=0o700)
    plan["user_authorized_max_posts"] = MAX_CALLS
    plan["started_utc"] = datetime.now(timezone.utc).isoformat()
    probe.atomic_json(out / "plan.json", plan)
    items = {}
    try:
        for index, (row, body) in enumerate(zip(rows, bodies), 1):
            if index > MAX_CALLS or quoted_cost(index * RESERVED_INPUT_TOKENS_PER_CALL) > SOFTWARE_ESTIMATE_BUDGET_USD:
                break
            if any(probe.digest(Path(filename)) != expected for filename, expected in plan["source_sha256"].items()):
                print("SOURCE_CHANGED_STOP; no further uploads", flush=True)
                break
            write_started(out / "started_attempts" / f"{index:02d}.json", {
                "ordinal": index, "local_id_DO_NOT_UPLOAD": row["id"], "request_sha256": prep.object_digest(json.loads(body)),
                "status": "reserved_before_network_attempt", "automatic_replay_forbidden": True})
            items[row["id"]] = {"status": "uncertain_started_attempt_requires_review"}
            start = time.perf_counter()
            try:
                result = post_once(opener, body, key)
                elapsed = time.perf_counter() - start
                probability = result["violation_probability"]
                band = risk_band(probability)
                route = "HUMAN_REVIEW" if decisions[row["id"]]["needs_review"] or band == "MEDIUM" else "HIGH_PRIORITY_REVIEW" if band == "HIGH" else "LOW_QUEUE"
                items[row["id"]] = {"status": "ok", **result, "predicted_label_threshold05": int(probability >= .5),
                                     "risk_band": band, "route": route, "api_roundtrip_seconds": elapsed}
                print(f"JEV [{index}/20] P={probability:.4f} BAND={band} ROUTE={route} SEC={elapsed:.3f} "
                      f"INPUT_TOKENS={result['input_tokens']}", flush=True)
            except Exception as exc:
                # Never print error bodies/messages/headers: they could contain
                # sensitive text or echo a key. Even timeout uncertainty STOPs.
                safe_error = f"HTTP_{exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
                items[row["id"]] = {"status": "failed_or_uncertain_no_retry", "error_type": safe_error,
                                     "route": "HUMAN_REVIEW", "possibly_billed": True}
                print(f"STOPPED [{index}/20] ERROR={safe_error} NO_RETRY; preserve the authorization ledger", flush=True)
                break
            finally:
                probe.atomic_json(out / "predictions_LOCAL_ONLY.json", {"metadata": plan, "items": items})
    finally:
        report = make_report(plan, rows, items, decisions, cpu)
        probe.atomic_json(out / "report.json", report)
    print(f"JEV_PILOT_COMPLETE={report['complete']} CALLS_STARTED={len(items)} VALID={report['valid_results']}/20 "
          f"KNOWN_USAGE_ESTIMATE_USD={report['billing_estimates']['known_usage_cost_usd']:.8f}", flush=True)
    print(f"REPORT={out / 'report.json'}; no automatic rerun/resume", flush=True)
    return 0 if report["complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--authorize-20-cloud-calls", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
