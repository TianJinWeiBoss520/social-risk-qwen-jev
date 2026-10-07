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

import run_ltedi_jev_text_vs_multimodal172 as run


class ContrastTests(unittest.TestCase):
    def fixture(self, root, duplicate_text=False):
        data = root / "data/processed/ltedi_grouped_v1"
        local = root / "outputs/ltedi_evidence_val172_v1"
        old = root / "outputs/jev_lora_remaining152_cloud_v1"
        for directory in (data, local / "lora_evidence", old):
            directory.mkdir(parents=True)
        rows = []
        for index in range(172):
            image = root / f"image{index}.jpg"
            image.write_bytes(b"SYNTHETIC_BYTES_NEVER_DECODED")
            rows.append({"id": f"train:{index}.jpg", "group_id": f"group{index % 165}",
                         "label": int(index < 49), "image": str(image.resolve()),
                         "text": f"合成文本 {index} 联系 13812345678 @example"})
        if duplicate_text:
            rows[1]["text"] = rows[0]["text"]
        run.prep.atomic_jsonl(data / "val.jsonl", rows)
        for name in ("train.jsonl", "test.jsonl"):
            (data / name).write_text("FORBIDDEN_UNREADABLE_MANIFEST", encoding="utf-8")
        qwen = {}
        multi = {}
        bodies, controls = [], []
        for index, row in enumerate(rows, 1):
            decision = {"label": row["label"], "gender_targeted": bool(row["label"]),
                        "attack_present": bool(row["label"]), "stance": "unclear", "image_role": "neutral",
                        "needs_review": False, "evidence": "合成的图文证据，不用于真实推理"}
            qwen[row["id"]] = {"status": "ok", "decision": decision, "latency_seconds": 5.0}
            multi[row["id"]] = {"status": "ok", "model": run.PINNED_MODEL,
                                "violation_probability": .9 if row["label"] else .1,
                                "input_tokens": 750, "output_tokens": 23, "api_roundtrip_seconds": 1.0}
            body, _, _ = run.prep.make_request(row, decision)
            bodies.append(body)
            controls.append({"request_index": index, "local_id": row["id"],
                             "gold_label_LOCAL_ONLY": row["label"], "qwen_predicted_label_LOCAL_ONLY": decision["label"],
                             "prepared_request_sha256": run.prep.object_digest(body)})
        run.prep.atomic_jsonl(local / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl", bodies)
        run.prep.atomic_jsonl(local / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl", controls)
        run.ev.atomic_json(local / "lora_evidence/predictions.json", {"items": qwen})
        run.ev.atomic_json(local / "plan.json", {"version": run.expand.VERSION, "modality": "image-text",
                                                "validation_ids_LOCAL_ONLY": [r["id"] for r in rows]})
        run.ev.atomic_json(local / "comparison_report.json", {"version": run.expand.VERSION,
            "test_read": False, "api_calls": 0, "sources_unchanged": True,
            "local_request_preparation": {"lora": {"ready": True, "n": 172,
                                                  "requests_sha256": run.prep.object_digest(bodies)}}})
        source_files = [data / "val.jsonl", *(f for f in local.rglob("*") if f.is_file())]
        old_plan = {"version": run.previous.VERSION, "requested_model": run.PINNED_MODEL,
                    "endpoint": run.PINNED_ENDPOINT, "validation_n": 172,
                    "test_read": False, "train_read": False,
                    "source_sha256": {str(f.resolve()): run.ev.digest(f) for f in source_files}}
        run.ev.atomic_json(old / "plan.json", old_plan)
        run.ev.atomic_json(old / "predictions_LOCAL_ONLY.json", {"metadata": old_plan, "items": multi})
        run.ev.atomic_json(old / "report.json", {"version": run.previous.VERSION, "complete": True,
            "validation_n": 172, "valid_results": 172, "new_post_attempts_started": 152,
            "reused_original_results": 20, "test_read": False, "jev_fixed_threshold": .5,
            "jev_classification_valid_only": run.ev.classification_metrics([r["label"] for r in rows], [r["label"] for r in rows]),
            "billing_estimates": {"combined_known_usage_cost_usd": .005214804}})
        return rows, local, old, root / "outputs" / run.OUTPUT_NAME

    @contextlib.contextmanager
    def guards(self):
        original = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.name in {"train.jsonl", "test.jsonl"} or path.suffix.lower() == ".jpg":
                raise AssertionError("No train/test access or image reading")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open), \
                mock.patch("socket.create_connection", side_effect=AssertionError("No real network")), \
                mock.patch.dict(sys.modules, {"torch": None, "peft": None, "PIL": None, "sklearn": None}):
            yield

    def test_plan_is_read_only_no_key_network_gpu_or_split_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            rows, local, old, out = self.fixture(root)
            files = [f for d in (local, old) for f in d.rglob("*") if f.is_file()]
            hashes = {f: run.ev.digest(f) for f in files}
            original_get = os.environ.get

            def guarded_get(name, *args):
                if name == "TYPESAFE_API_KEY":
                    raise AssertionError("No key access")
                return original_get(name, *args)

            output = io.StringIO()
            with self.guards(), mock.patch.object(os.environ, "get", side_effect=guarded_get), \
                    mock.patch.object(run.cloud, "post_once", side_effect=AssertionError("No POST")), \
                    mock.patch.object(run.urllib.request, "build_opener", side_effect=AssertionError("No client")), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(run.main(["--project-root", str(root), "--plan-only"]), 0)
            self.assertFalse(out.exists())
            self.assertIn("RESERVED_NEW_USD=0.47343206", output.getvalue())
            self.assertIn("API_CALLS=0", output.getvalue())
            self.assertEqual(hashes, {f: run.ev.digest(f) for f in files})
            cli = subprocess.run([sys.executable, "-S", str(Path(run.__file__)), "--project-root", str(root), "--plan-only"],
                                 capture_output=True, text=True)
            self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)

    def test_172_calls_identical_policy_and_text_no_evidence_or_secret_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            rows, local, old, out = self.fixture(root)
            old_hashes = {f: run.ev.digest(f) for d in (local, old) for f in d.rglob("*") if f.is_file()}
            bodies = []

            def sender(opener, encoded, key):
                index = len(bodies)
                body = json.loads(encoded)
                self.assertEqual(key, "SYNTHETIC_SECRET_NEVER_LOG")
                self.assertEqual(set(body), {"model", "questions", "state"})
                self.assertEqual(set(body["state"]), {"transcription"})
                self.assertEqual(body["model"], run.PINNED_MODEL)
                self.assertEqual(body["questions"], {"policy_violation": run.prep.QUESTION})
                self.assertEqual(body["state"]["transcription"], run.prep.mask_text(rows[index]["text"])[0])
                ledger = run.prep.read_json(out / "started_attempts" / f"{index + 1:03d}.json")
                self.assertTrue(ledger["automatic_replay_forbidden"])
                self.assertEqual(ledger["request_sha256"], run.prep.object_digest(body))
                checkpoint = run.prep.read_json(out / "text_predictions_LOCAL_ONLY.json")
                self.assertIn("uncertain", checkpoint["items"][rows[index]["id"]]["status"])
                for forbidden in ("train:", '"label"', '"image"', '"group_id"', "13812345678", "@example"):
                    self.assertNotIn(forbidden, encoded.decode("utf-8"))
                bodies.append(body)
                label = rows[index]["label"] if index not in (0, 50) else 1 - rows[index]["label"]
                return {"model": run.PINNED_MODEL, "violation_probability": .9 if label else .1,
                        "input_tokens": 500, "output_tokens": 23}

            output = io.StringIO()
            with self.guards(), mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC_SECRET_NEVER_LOG"}), \
                    mock.patch.object(run.cloud, "post_once", side_effect=sender), \
                    mock.patch.object(run.urllib.request, "build_opener", return_value=object()), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(run.main(["--project-root", str(root), "--authorize-text-cloud-calls"]), 0)
                with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                    run.main(["--project-root", str(root), "--authorize-text-cloud-calls"])
            report = run.prep.read_json(out / "comparison_report.json")
            self.assertEqual(len(bodies), 172)
            self.assertTrue(report["complete"])
            self.assertEqual(report["fixed_comparisons"]["primary_t040"]["paired"], {"gained": 2, "lost": 0, "net_correct": 2})
            self.assertEqual(report["fixed_comparisons"]["primary_t040"]["group_bootstrap"]["group_count"], 165)
            self.assertAlmostEqual(report["billing"]["new_known_usage_cost_usd"], .003612)
            self.assertEqual(report["latency"]["reconstructed_multimodal_pipeline"]["mean_seconds"], 6.0)
            self.assertEqual(old_hashes, {f: run.ev.digest(f) for f in old_hashes})
            self.assertNotIn("SYNTHETIC_SECRET_NEVER_LOG", output.getvalue())
            for f in out.rglob("*"):
                if f.is_file():
                    self.assertNotIn("SYNTHETIC_SECRET_NEVER_LOG", f.read_text(encoding="utf-8"))

    def test_identical_text_payloads_are_paid_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            _, _, _, out = self.fixture(root, duplicate_text=True)
            with self.guards(), mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC"}), \
                    mock.patch.object(run.cloud, "post_once", return_value={"model": run.PINNED_MODEL,
                        "violation_probability": .2, "input_tokens": 500, "output_tokens": 23}) as post, \
                    mock.patch.object(run.urllib.request, "build_opener", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(["--project-root", str(root), "--authorize-text-cloud-calls"]), 0)
            self.assertEqual(post.call_count, 171)
            report = run.prep.read_json(out / "comparison_report.json")
            self.assertEqual(report["exact_text_payload_cache_hits"], 1)
            self.assertEqual(report["latency"]["new_text_jev_http_roundtrip"]["n"], 171)

    def test_failures_stop_after_one_call_and_suppress_partial_accuracy(self):
        failures = (TimeoutError("SYNTHETIC_SECRET_NEVER_LOG"),
                    urllib.error.HTTPError(run.PINNED_ENDPOINT, 429, "SYNTHETIC_SECRET_NEVER_LOG", {}, None),
                    ValueError("SYNTHETIC_SECRET_NEVER_LOG"))
        for failure in failures:
            with self.subTest(kind=type(failure).__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "project"
                _, _, _, out = self.fixture(root)
                output = io.StringIO()
                with self.guards(), mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC_SECRET_NEVER_LOG"}), \
                        mock.patch.object(run.cloud, "post_once", side_effect=failure) as post, \
                        mock.patch.object(run.urllib.request, "build_opener", return_value=object()), \
                        contextlib.redirect_stdout(output):
                    self.assertEqual(run.main(["--project-root", str(root), "--authorize-text-cloud-calls"]), 2)
                    with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                        run.main(["--project-root", str(root), "--authorize-text-cloud-calls"])
                self.assertEqual(post.call_count, 1)
                self.assertNotIn("SYNTHETIC_SECRET_NEVER_LOG", output.getvalue())
                report = run.prep.read_json(out / "comparison_report.json")
                self.assertFalse(report["complete"])
                self.assertFalse(report["quality_metrics_computed"])
                self.assertNotIn("fixed_comparisons", report)
                self.assertEqual(report["billing"]["failed_or_uncertain_attempts"], 1)

    def test_missing_key_auth_and_budget_never_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            _, _, _, out = self.fixture(root)
            with self.guards(), mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}), \
                    mock.patch.object(run.cloud, "post_once", side_effect=AssertionError("No POST")), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "AUTHORIZATION"):
                    run.main(["--project-root", str(root)])
                with self.assertRaisesRegex(ValueError, "KEY_MISSING"):
                    run.main(["--project-root", str(root), "--authorize-text-cloud-calls"])
                with mock.patch.object(run, "NEW_FORECAST_LIMIT_USD", .10):
                    with self.assertRaisesRegex(ValueError, "BUDGET_FORECAST"):
                        run.main(["--project-root", str(root), "--plan-only"])
            self.assertFalse(out.exists())

    def test_tampered_source_or_mismatched_cache_stops_before_post(self):
        for target in ("source", "cache", "policy"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "project"
                _, local, old, out = self.fixture(root)
                if target == "source":
                    with (local / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl").open("a", encoding="utf-8") as f:
                        f.write("\n")
                elif target == "cache":
                    saved = run.prep.read_json(old / "predictions_LOCAL_ONLY.json")
                    saved["items"]["train:0.jpg"]["violation_probability"] = .1
                    run.ev.atomic_json(old / "predictions_LOCAL_ONLY.json", saved)
                else:
                    saved = run.prep.read_json(old / "predictions_LOCAL_ONLY.json")
                    saved["items"]["train:0.jpg"]["model"] = "jev-latest"
                    run.ev.atomic_json(old / "predictions_LOCAL_ONLY.json", saved)
                with self.guards(), mock.patch.object(run.cloud, "post_once", side_effect=AssertionError("No POST")):
                    with self.assertRaises(ValueError):
                        run.main(["--project-root", str(root), "--plan-only"])
                self.assertFalse(out.exists())

    def test_source_change_during_run_stops_before_next_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            _, local, _, out = self.fixture(root)

            def sender(*args):
                with (local / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl").open("a", encoding="utf-8") as f:
                    f.write("\n")
                return {"model": run.PINNED_MODEL, "violation_probability": .1, "input_tokens": 500, "output_tokens": 23}

            with self.guards(), mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC"}), \
                    mock.patch.object(run.cloud, "post_once", side_effect=sender) as post, \
                    mock.patch.object(run.urllib.request, "build_opener", return_value=object()), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "SOURCE_CHANGED"):
                    run.main(["--project-root", str(root), "--authorize-text-cloud-calls"])
            self.assertEqual(post.call_count, 1)
            self.assertTrue((out / "interrupted_LOCAL_ONLY.json").is_file())
            self.assertFalse((out / "comparison_report.json").exists())

    def test_split_hash_guard_rejects_before_any_hash_and_redirects_disabled(self):
        with mock.patch.object(run.ev, "digest", side_effect=AssertionError("Should not hash")):
            with self.assertRaisesRegex(ValueError, "TRAIN_OR_TEST"):
                run.verify_sources({"arbitrary/val.jsonl": "a", "forbidden/test.jsonl": "b"})
        self.assertIsNone(run.cloud.NoRedirect().redirect_request(None, None, 307, "Moved", {}, "https://other.example"))


if __name__ == "__main__":
    unittest.main()
