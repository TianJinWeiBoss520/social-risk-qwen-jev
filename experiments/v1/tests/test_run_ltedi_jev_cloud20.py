import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import evaluate_ltedi_qwen_base as probe
import prepare_ltedi_jev_pilot as prep
import probe_ltedi_lora_evidence20 as evidence
import run_ltedi_jev_cloud20 as cloud


class CloudTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "data/processed/ltedi_grouped_v1"
        source = root / "outputs/ltedi_lora_evidence20_v1"
        cpu = root / "outputs/cpu_text_baselines_v1"
        for path in (data, source / "lora_evidence", cpu):
            path.mkdir(parents=True)
        rows = []
        for index in range(24):
            image = root / f"{index}.jpg"
            image.write_bytes(b"SYNTHETIC_IMAGE_NOT_DECODED")
            rows.append({"id": f"train:{index}.jpg", "group_id": f"g{index}", "image": str(image),
                         "text": f"程序测试合成文本{index}", "label": int(index < 3)})
        (data / "val.jsonl").write_text("".join(prep.canonical(row) + "\n" for row in rows), encoding="utf-8")
        (data / "test.jsonl").write_text("NEVER_READ_TEST", encoding="utf-8")
        val_hash = probe.digest(data / "val.jsonl")
        selected = prep.select_rows(rows, 20, 2062)
        cpu_name = "tfidf_logreg_c4"
        probe.atomic_json(cpu / "validation_report.json", {"selected_text_baseline": cpu_name,
                                                            "source_split_sha256": {"val": val_hash}})
        (cpu / f"{cpu_name}_validation_predictions.jsonl").write_text("".join(
            prep.canonical({"id": row["id"], "predicted_label": row["label"]}) + "\n" for row in rows), encoding="utf-8")
        plan = {"version": evidence.VERSION, "test_read": False, "train_read": False, "api_calls": 0,
                "selected_n": 20, "modality": "image-text", "selected_ids_LOCAL_ONLY": [r["id"] for r in selected],
                "source_sha256": {str(data / "val.jsonl"): val_hash},
                "frozen_preparation_identity": {"source_validation_sha256": val_hash, "seed": 2062}}
        probe.atomic_json(source / "plan.json", plan)
        items, requests, controls = {}, [], []
        for index, row in enumerate(selected, 1):
            decision = {"label": row["label"], "gender_targeted": bool(row["label"]),
                        "attack_present": bool(row["label"]), "stance": "unclear", "image_role": "neutral",
                        "needs_review": index == 1, "evidence": "仅用于程序测试的合成证据"}
            items[row["id"]] = {"status": "ok", "decision": decision, "latency_seconds": 5.0}
            request, _, _ = prep.make_request(row, decision)
            requests.append(request)
            controls.append({"request_index": index, "local_id": row["id"], "gold_label_LOCAL_ONLY": row["label"],
                             "qwen_predicted_label_LOCAL_ONLY": decision["label"],
                             "prepared_request_sha256": prep.object_digest(request)})
        checksum = prep.object_digest(requests)
        probe.atomic_json(source / "lora_evidence/predictions.json", {
            "metadata": {**plan, "stage": "LORA_EVIDENCE", "adapter_active": True}, "items": items})
        prep.atomic_jsonl(source / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl", requests)
        prep.atomic_jsonl(source / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl", controls)
        probe.atomic_json(source / "comparison_report.json", {"version": evidence.VERSION,
            "test_read": False, "cloud_upload": False, "api_calls": 0, "sources_unchanged": True,
            "lora_request_preparation": {"ready": True, "n": 20, "requests_sha256": checksum}})
        return source, checksum, selected

    def test_plan_no_key_network_gpu_train_test_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, checksum, _ = self.fixture(root)
            original_open = Path.open

            def guard(path, *args, **kwargs):
                if path.name in ("train.jsonl", "test.jsonl"):
                    raise AssertionError("No train/test reads")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(cloud, "ROOT", root), mock.patch.object(cloud, "EXPECTED_REQUESTS", checksum), \
                    mock.patch.object(Path, "open", guard), mock.patch.dict(sys.modules, {"torch": None}), \
                    mock.patch.object(cloud.os, "environ", {}), \
                    mock.patch.object(cloud.urllib.request, "build_opener", side_effect=AssertionError("No network")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cloud.main(["--plan-only"]), 0)
            self.assertFalse((root / "outputs/jev_lora_val20_cloud_v1").exists())
            code = "import sys; from pathlib import Path; import run_ltedi_jev_cloud20 as c; c.ROOT=Path(sys.argv[1]); c.EXPECTED_REQUESTS=sys.argv[2]; raise SystemExit(c.main(['--plan-only']))"
            result = subprocess.run([sys.executable, "-S", "-c", code, str(root), checksum],
                                    cwd=Path(cloud.__file__).parent, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_authorization_key_and_frozen_payload_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, checksum, _ = self.fixture(root)
            with mock.patch.object(cloud, "ROOT", root), mock.patch.object(cloud, "EXPECTED_REQUESTS", checksum), \
                    mock.patch.object(cloud.os, "environ", {}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "AUTHORIZATION"):
                    cloud.main([])
                with self.assertRaisesRegex(ValueError, "KEY_MISSING"):
                    cloud.main(["--authorize-20-cloud-calls"])
                path = source / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl"
                rows = cloud.read_jsonl(path)
                rows[0]["state"]["extra_gold_label"] = 1
                path.write_text("".join(prep.canonical(row) + "\n" for row in rows), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "PAYLOAD_MISMATCH"):
                    cloud.main(["--plan-only"])
            self.assertFalse((root / "outputs/jev_lora_val20_cloud_v1").exists())

    def test_exactly_20_attempts_and_durable_once_only_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, checksum, selected = self.fixture(root)
            calls = []
            out = root / "outputs/jev_lora_val20_cloud_v1"

            def post(opener, body, key):
                index = len(calls)
                self.assertTrue((out / "started_attempts" / f"{index+1:02d}.json").is_file())
                self.assertEqual(key, "SYNTHETIC_SECRET_DO_NOT_PRINT")
                request = json.loads(body)
                self.assertEqual(set(request), {"model", "state", "questions"})
                self.assertNotIn('"label"', body.decode())
                self.assertNotIn(selected[index]["id"], body.decode())
                calls.append(request)
                return {"model": prep.MODEL, "violation_probability": .9 if selected[index]["label"] else .1,
                        "input_tokens": 500, "output_tokens": 23}

            output = io.StringIO()
            with mock.patch.object(cloud, "ROOT", root), mock.patch.object(cloud, "EXPECTED_REQUESTS", checksum), \
                    mock.patch.dict(cloud.os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC_SECRET_DO_NOT_PRINT"}), \
                    mock.patch.object(cloud, "post_once", side_effect=post), contextlib.redirect_stdout(output):
                self.assertEqual(cloud.main(["--authorize-20-cloud-calls"]), 0)
                with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                    cloud.main(["--authorize-20-cloud-calls"])
            self.assertEqual(len(calls), 20)
            self.assertNotIn("SYNTHETIC_SECRET_DO_NOT_PRINT", output.getvalue())
            report = prep.read_json(out / "report.json")
            self.assertTrue(report["complete"])
            self.assertEqual(report["jev_classification_valid_only"]["accuracy"], 1)
            self.assertAlmostEqual(report["billing_estimates"]["known_usage_cost_usd"], .00042)
            self.assertLess(report["billing_estimates"]["context_reserved_cost_started_attempts_usd"], .1)
            predictions = prep.read_json(out / "predictions_LOCAL_ONLY.json")
            self.assertEqual(predictions["items"][selected[0]["id"]]["route"], "HUMAN_REVIEW")
            for path in out.rglob("*.json"):
                self.assertNotIn("SYNTHETIC_SECRET_DO_NOT_PRINT", path.read_text(encoding="utf-8"))

    def test_http_errors_timeout_and_bad_response_stop_without_retry_or_sensitive_log(self):
        failures = [TimeoutError("SYNTHETIC_SECRET_DO_NOT_PRINT"),
                    urllib.error.HTTPError(cloud.ENDPOINT, 429, "SYNTHETIC_SECRET_DO_NOT_PRINT", {}, None),
                    ValueError("SYNTHETIC_SECRET_DO_NOT_PRINT")]
        for failure in failures:
            with self.subTest(error=type(failure).__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, checksum, _ = self.fixture(root)
                output = io.StringIO()
                with mock.patch.object(cloud, "ROOT", root), mock.patch.object(cloud, "EXPECTED_REQUESTS", checksum), \
                        mock.patch.dict(cloud.os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC_SECRET_DO_NOT_PRINT"}), \
                        mock.patch.object(cloud, "post_once", side_effect=failure) as sender, \
                        contextlib.redirect_stdout(output):
                    self.assertEqual(cloud.main(["--authorize-20-cloud-calls"]), 2)
                self.assertEqual(sender.call_count, 1)
                self.assertNotIn("SYNTHETIC_SECRET_DO_NOT_PRINT", output.getvalue())
                report = prep.read_json(root / "outputs/jev_lora_val20_cloud_v1/report.json")
                self.assertFalse(report["complete"])
                self.assertEqual(report["valid_results"], 0)
                self.assertEqual(report["billing_estimates"]["uncertain_attempts"], 1)
                self.assertEqual(len(report["unresolved_ids_LOCAL_ONLY"]), 20)

    def test_response_and_band_boundaries_are_strict(self):
        response = {"model": prep.MODEL, "answers": {"policy_violation": {"type": "noul", "noul": .8}},
                    "usage": {"input_tokens": 388, "output_tokens": 23}}
        self.assertEqual(cloud.parse_response(prep.canonical(response))["violation_probability"], .8)
        for bad in (True, -1, 1.01, "0.8", float("nan")):
            response["answers"]["policy_violation"]["noul"] = bad
            with self.assertRaises((ValueError, TypeError)):
                cloud.parse_response(json.dumps(response))
        self.assertEqual([cloud.risk_band(p) for p in (0, .1999, .2, .7999, .8, 1)],
                         ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"])
        self.assertIsNone(cloud.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.org"))

    def test_network_request_only_to_pinned_host_and_never_redirected(self):
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, limit):
                return prep.canonical({"model": prep.MODEL,
                    "answers": {"policy_violation": {"type": "noul", "noul": .1}},
                    "usage": {"input_tokens": 300, "output_tokens": 23}}).encode()
        opener = mock.Mock()
        opener.open.return_value = Response()
        cloud.post_once(opener, b'{"synthetic":true}', "SYNTHETIC_SECRET")
        self.assertEqual(opener.open.call_count, 1)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, cloud.ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer SYNTHETIC_SECRET")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 30)

    def test_metrics_handle_ties(self):
        scores = cloud.probability_metrics([0, 0, 1, 1], [.5, .5, .5, .5])
        self.assertEqual(scores["roc_auc"], .5)
        self.assertEqual(scores["pr_auc_average_precision"], .5)
        self.assertEqual(scores["brier_score"], .25)


if __name__ == "__main__":
    unittest.main()
