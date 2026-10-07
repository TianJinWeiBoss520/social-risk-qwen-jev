import contextlib
import io
import json
import os
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
import expand_ltedi_evidence_val172 as expand
import run_ltedi_jev_cloud20 as cloud
import run_ltedi_jev_remaining152 as client
import test_expand_ltedi_evidence_val172 as expansion_tests
from test_probe_ltedi_lora_evidence20 import FakeModel


class RemainingCloudTests(unittest.TestCase):
    def fixture(self, root):
        rows, infer, weights, requests = expansion_tests.ExpansionTests().fixture(root)
        with mock.patch.object(evidence, "EXPECTED_WEIGHTS", weights), \
                mock.patch.object(cloud, "EXPECTED_REQUESTS", requests), \
                mock.patch.object(evidence, "load_matched_model", return_value=(object(), FakeModel())), \
                mock.patch.object(probe, "infer_one", side_effect=infer), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(expand.main(["--project-root", str(root)]), 0)
        return rows, weights, requests

    @contextlib.contextmanager
    def guarded(self, root, weights, requests):
        original_open = Path.open

        def guard(path, *args, **kwargs):
            if path.name in ("train.jsonl", "test.jsonl"):
                raise AssertionError("Only validation may be read")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(client, "ROOT", root), \
                mock.patch.object(evidence, "EXPECTED_WEIGHTS", weights), \
                mock.patch.object(cloud, "EXPECTED_REQUESTS", requests), \
                mock.patch.object(Path, "open", guard), \
                mock.patch.dict(sys.modules, {"torch": None, "peft": None}), \
                mock.patch("socket.create_connection", side_effect=AssertionError("No real network")):
            yield

    def test_plan_no_gpu_key_calls_train_test_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            _, weights, requests = self.fixture(root)
            output = io.StringIO()
            original_get = os.environ.get

            def environment_get(name, *args):
                if name == "TYPESAFE_API_KEY":
                    raise AssertionError("No key access")
                return original_get(name, *args)

            with self.guarded(root, weights, requests), \
                    mock.patch.object(os.environ, "get", side_effect=environment_get), \
                    mock.patch.object(cloud, "post_once", side_effect=AssertionError("No POST")), \
                    mock.patch.object(urllib.request, "build_opener", side_effect=AssertionError("No network client")), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(client.main(["--plan-only"]), 0)
            self.assertIn("API_CALLS=0", output.getvalue())
            self.assertIn("RESERVED_NEW_ESTIMATE_USD=0.41838182", output.getvalue())
            self.assertFalse((root / "outputs/jev_lora_remaining152_cloud_v1").exists())
            code = ("import sys, pathlib, run_ltedi_jev_remaining152 as c; "
                    "c.ROOT=pathlib.Path(sys.argv.pop(1)); "
                    "c.evidence.EXPECTED_WEIGHTS=sys.argv.pop(1); "
                    "c.cloud.EXPECTED_REQUESTS=sys.argv.pop(1); raise SystemExit(c.main(sys.argv[1:]))")
            result = subprocess.run([sys.executable, "-S", "-c", code, str(root), weights, requests, "--plan-only"],
                                    cwd=Path(client.__file__).parent, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_exactly152_new_calls_reuse20_and_preserve_originals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            rows, weights, requests = self.fixture(root)
            source = root / "outputs/ltedi_evidence_val172_v1"
            local_plan = prep.read_json(source / "plan.json")
            controls = evidence.read_jsonl(source / "remaining152_for_jev/request_controls_DO_NOT_UPLOAD.jsonl")
            gold = {row["id"]: row["label"] for row in rows}
            old_ids = set(local_plan["pilot_ids_LOCAL_ONLY"])
            old_files = [path for name in ("ltedi_lora_evidence20_v1", "jev_lora_val20_cloud_v1", "ltedi_evidence_val172_v1")
                         for path in (root / "outputs" / name).rglob("*") if path.is_file()]
            hashes = {path: probe.digest(path) for path in old_files}
            out = root / "outputs/jev_lora_remaining152_cloud_v1"
            calls = []

            def post(opener, body, key):
                index = len(calls)
                ledger = prep.read_json(out / "started_attempts" / f"{index + 1:03d}.json")
                self.assertTrue(ledger["automatic_replay_forbidden"])
                self.assertEqual(ledger["local_id_DO_NOT_UPLOAD"], controls[index]["local_id"])
                self.assertNotIn(ledger["local_id_DO_NOT_UPLOAD"], old_ids)
                self.assertEqual(ledger["request_sha256"], prep.object_digest(json.loads(body)))
                self.assertEqual(set(json.loads(body)), {"model", "state", "questions"})
                for forbidden in ('"label"', '"image"', '"local_id"', '"group_id"', "train:"):
                    self.assertNotIn(forbidden, body.decode())
                self.assertEqual(key, "SYNTHETIC_SECRET_NEVER_LOG")
                calls.append(body)
                return {"model": prep.MODEL, "violation_probability": .9 if gold[controls[index]["local_id"]] else .1,
                        "input_tokens": 500, "output_tokens": 23}

            output = io.StringIO()
            with self.guarded(root, weights, requests), \
                    mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC_SECRET_NEVER_LOG"}), \
                    mock.patch.object(cloud, "post_once", side_effect=post), \
                    mock.patch.object(urllib.request, "build_opener", return_value=object()), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(client.main(["--authorize-152-cloud-calls"]), 0)
                with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                    client.main(["--authorize-152-cloud-calls"])
            self.assertEqual(len(calls), 152)
            self.assertNotIn("SYNTHETIC_SECRET_NEVER_LOG", output.getvalue())
            saved = prep.read_json(out / "predictions_LOCAL_ONLY.json")["items"]
            for key in old_ids:
                self.assertEqual(saved[key]["result_origin"], "reused_original20")
            report = prep.read_json(out / "report.json")
            self.assertTrue(report["complete"])
            self.assertEqual(report["valid_results"], 172)
            self.assertEqual(report["reused_original_results"], 20)
            self.assertEqual(report["new_post_attempts_started"], 152)
            self.assertEqual(report["new_api_successes"], 152)
            self.assertEqual(report["new_api_roundtrip_seconds"]["n"], 152)
            self.assertAlmostEqual(report["billing_estimates"]["new_known_usage_cost_usd"], .003192)
            self.assertAlmostEqual(report["billing_estimates"]["previous20_known_usage_cost_usd"], .000588)
            self.assertEqual(hashes, {path: probe.digest(path) for path in old_files})
            self.assertEqual(len(list((out / "started_attempts").glob("*.json"))), 152)
            for path in out.rglob("*.json"):
                self.assertNotIn("SYNTHETIC_SECRET_NEVER_LOG", path.read_text(encoding="utf-8"))

    def test_error_uncertainty_stops_after_one_attempt_no_replay_or_secret(self):
        for failure in (TimeoutError("SYNTHETIC_SECRET_NEVER_LOG"),
                        urllib.error.HTTPError(cloud.ENDPOINT, 429, "SYNTHETIC_SECRET_NEVER_LOG", {}, None),
                        ValueError("SYNTHETIC_SECRET_NEVER_LOG")):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "project"
                _, weights, requests = self.fixture(root)
                output = io.StringIO()
                with self.guarded(root, weights, requests), \
                        mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC_SECRET_NEVER_LOG"}), \
                        mock.patch.object(cloud, "post_once", side_effect=failure) as sender, \
                        mock.patch.object(urllib.request, "build_opener", return_value=object()), \
                        contextlib.redirect_stdout(output):
                    self.assertEqual(client.main(["--authorize-152-cloud-calls"]), 2)
                    with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                        client.main(["--authorize-152-cloud-calls"])
                self.assertEqual(sender.call_count, 1)
                self.assertNotIn("SYNTHETIC_SECRET_NEVER_LOG", output.getvalue())
                report = prep.read_json(root / "outputs/jev_lora_remaining152_cloud_v1/report.json")
                self.assertFalse(report["complete"])
                self.assertEqual(report["valid_results"], 20)
                self.assertEqual(report["new_post_attempts_started"], 1)
                self.assertEqual(report["billing_estimates"]["new_uncertain_attempts"], 1)
                self.assertEqual(len(report["unresolved_ids_LOCAL_ONLY"]), 152)

    def test_missing_auth_missing_key_and_forecast_limit_do_not_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            _, weights, requests = self.fixture(root)
            with self.guarded(root, weights, requests), mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}), \
                    mock.patch.object(cloud, "post_once", side_effect=AssertionError("No calls")), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "AUTHORIZATION_FLAG"):
                    client.main([])
                with self.assertRaisesRegex(ValueError, "KEY_MISSING"):
                    client.main(["--authorize-152-cloud-calls"])
                with mock.patch.object(client, "SOFTWARE_FORECAST_LIMIT_USD", .01), \
                        self.assertRaisesRegex(ValueError, "software limit"):
                    client.main(["--plan-only"])
            self.assertFalse((root / "outputs/jev_lora_remaining152_cloud_v1").exists())

    def test_exact_payload_duplicates_reuse_cache_without_new_posts(self):
        original = prep.make_request

        def same_transcription(row, decision):
            return original({**row, "text": "仅用于自动化测试的完全相同文本"}, decision)

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(prep, "make_request", side_effect=same_transcription):
            root = Path(temporary) / "project"
            _, weights, requests = self.fixture(root)
            with self.guarded(root, weights, requests), \
                    mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC_ONLY"}), \
                    mock.patch.object(cloud, "post_once", side_effect=AssertionError("Exact payload cache must avoid POST")), \
                    mock.patch.object(urllib.request, "build_opener", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(client.main(["--authorize-152-cloud-calls"]), 0)
            report = prep.read_json(root / "outputs/jev_lora_remaining152_cloud_v1/report.json")
            self.assertTrue(report["complete"])
            self.assertEqual(report["new_post_attempts_started"], 0)
            self.assertEqual(report["additional_exact_payload_cache_hits"], 152)
            self.assertEqual(report["billing_estimates"]["new_known_usage_cost_usd"], 0)

    def test_tampered_remaining_payload_and_controls_cannot_reach_api(self):
        for damage in ("body", "controls"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "project"
                _, weights, requests = self.fixture(root)
                directory = root / "outputs/ltedi_evidence_val172_v1/remaining152_for_jev"
                filename = "jev_requests_LOCAL_ONLY.jsonl" if damage == "body" else "request_controls_DO_NOT_UPLOAD.jsonl"
                path = directory / filename
                contents = evidence.read_jsonl(path)
                if damage == "body":
                    contents[0]["state"]["gold_label"] = 1
                else:
                    contents[0]["local_id"] = "test:FORBIDDEN.jpg"
                prep.atomic_jsonl(path, contents)
                with self.guarded(root, weights, requests), \
                        mock.patch.object(cloud, "post_once", side_effect=AssertionError("No API")), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    client.main(["--plan-only"])
                self.assertFalse((root / "outputs/jev_lora_remaining152_cloud_v1").exists())


if __name__ == "__main__":
    unittest.main()
