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

import run_ltedi_frozen_test170 as run
from test_probe_ltedi_lora_evidence20 import FakeModel


class FrozenTestTests(unittest.TestCase):
    def fixture(self, root):
        p = run.paths(root)
        for directory in (p.data, p.cpu, p.validation, p.local_validation / "lora_evidence", p.adapter_path, p.model_path):
            directory.mkdir(parents=True, exist_ok=True)
        splits = {}
        for name, size, prefix, positives in (("val", 172, "train", 49), ("test", 170, "dev", 47)):
            rows = []
            for index in range(size):
                image = root / f"{name}_{index}.jpg"
                image.write_bytes(f"SYNTHETIC_IMAGE_NOT_DECODED_{name}_{index}".encode())
                label = int(index < positives)
                rows.append({"id": f"{prefix}:{index}.jpg", "group_id": f"{name}-g{index}",
                    "image": str(image.resolve()), "text": f"合成{name} {'正例' if label else '负例'} {index}", "label": label})
            splits[name] = rows
            run.prep.atomic_jsonl(p.data / f"{name}.jsonl", rows)
        (p.data / "train.jsonl").write_text("FORBIDDEN_TRAIN", encoding="utf-8")
        run.ev.atomic_json(p.data / "preparation_report.json", {"group_overlap_between_splits": 0, "splits": {"test": {"rows": 170}}})
        (p.model_path / "config.json").write_text('{}', encoding="utf-8")
        weights = p.adapter_path / "adapter_model.safetensors"
        weights.write_bytes(b"SYNTHETIC_NO_MODEL")
        sha = run.ev.digest(weights)
        source_splits = {"train": "synthetic-training-hash", "val": run.ev.digest(p.data / "val.jsonl")}
        cpu_report = {"selected_text_baseline": "tfidf_logreg_c4", "source_split_sha256": source_splits,
            "models": {name: {"validation_selected_threshold": {"threshold": .7}} for name in run.MODEL_NAMES}}
        run.ev.atomic_json(p.cpu / "validation_report.json", cpu_report)
        run.prep.atomic_jsonl(p.cpu / "tfidf_logreg_c4_validation_predictions.jsonl",
            [{"id": r["id"], "predicted_label": r["label"]} for r in splits["val"]])
        for name in run.MODEL_NAMES:
            (p.cpu / f"{name}.joblib").write_bytes(b"SYNTHETIC_PICKLE_PLACEHOLDER_NEVER_LOADED")
        decisions = {r["id"]: self.decision(r["label"]) for r in splits["val"]}
        records = {key: {"status": "ok", "decision": value, "latency_seconds": .2} for key, value in decisions.items()}
        run.ev.atomic_json(p.local_validation / "lora_evidence/predictions.json", {"items": records})
        requests, controls = [], []
        for index, row in enumerate(splits["val"], 1):
            request, _, _ = run.prep.make_request(row, decisions[row["id"]])
            requests.append(request)
            controls.append({"local_id": row["id"], "prepared_request_sha256": run.prep.object_digest(request)})
        run.prep.atomic_jsonl(p.local_validation / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl", requests)
        run.prep.atomic_jsonl(p.local_validation / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl", controls)
        items = {r["id"]: {"status": "ok", "model": run.prep.MODEL, "violation_probability": .85 if r["label"] else .1,
            "input_tokens": 720, "output_tokens": 23, "api_roundtrip_seconds": .1} for r in splits["val"]}
        source_files = [p.data / "val.jsonl", p.cpu / "validation_report.json", weights,
            p.local_validation / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl",
            p.local_validation / "lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl"]
        source_sha = {str(f.resolve()): run.ev.digest(f) for f in source_files}
        run.ev.atomic_json(p.validation / "plan.json", {"synthetic": True})
        run.ev.atomic_json(p.validation / "predictions_LOCAL_ONLY.json", {"metadata": {"source_sha256": source_sha}, "items": items})
        run.ev.atomic_json(p.validation / "report.json", {"version": run.previous.VERSION, "complete": True,
            "validation_n": 172, "valid_results": 172, "test_read": False, "new_post_attempts_started": 152,
            "jev_classification_valid_only": run.ev.classification_metrics([r["label"] for r in splits["val"]], [r["label"] for r in splits["val"]]),
            "billing_estimates": {"combined_known_usage_cost_usd": .005214804, "new_known_usage_cost_usd": .004606602}})
        return p, sha, splits

    @staticmethod
    def decision(label):
        return {"label": label, "gender_targeted": bool(label), "attack_present": bool(label),
            "stance": "unclear", "image_role": "neutral", "needs_review": False,
            "evidence": "仅用于离线自动化测试的合成证据"}

    def cpu(self, p, frozen, inputs):
        return ({name: {r["id"]: {"score": .9 if "正例" in r["text"] else .1,
            "predicted_label": int("正例" in r["text"])} for r in inputs} for name in run.MODEL_NAMES},
            {name: {"mean_seconds": .001, "n": len(inputs)} for name in run.MODEL_NAMES})

    def infer(self, processor, model, row, modality, budget):
        self.assertEqual(set(row), {"image", "text"})
        self.assertEqual((modality, budget), ("image-text", 384))
        return {"status": "ok", "decision": self.decision(int("正例" in row["text"])), "latency_seconds": .2}

    @contextlib.contextmanager
    def mocks(self, sha):
        with mock.patch.object(run.evidence, "EXPECTED_WEIGHTS", sha), \
                mock.patch.object(run, "cpu_predictions", side_effect=self.cpu), \
                mock.patch.object(run.evidence, "load_matched_model", return_value=(object(), FakeModel())), \
                mock.patch.object(run.ev, "infer_one", side_effect=self.infer), \
                mock.patch("socket.create_connection", side_effect=AssertionError("No real network")), \
                mock.patch.dict(sys.modules, {"torch": None, "peft": None}), \
                contextlib.redirect_stdout(io.StringIO()):
            yield

    def prepare_and_infer(self, p, sha):
        with self.mocks(sha):
            self.assertEqual(run.main(["--project-root", str(p.root), "--stage", "prepare"]), 0)
            self.assertEqual(run.main(["--project-root", str(p.root), "--stage", "evidence"]), 0)

    def test_plan_no_test_train_key_gpu_network_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, sha, splits = self.fixture(Path(tmp) / "project")
            original = Path.open
            original_get = os.environ.get
            def safe_open(path, *a, **kw):
                self.assertNotIn(path.name, ("train.jsonl", "test.jsonl"))
                return original(path, *a, **kw)
            def safe_env(name, *a):
                if name == "TYPESAFE_API_KEY":
                    raise AssertionError("Plan must not read a key")
                return original_get(name, *a)
            with self.mocks(sha), mock.patch.object(Path, "open", safe_open), \
                    mock.patch.object(os.environ, "get", side_effect=safe_env), \
                    mock.patch.object(run, "cpu_predictions", side_effect=AssertionError("No CPU deps in plan")), \
                    mock.patch.object(run.cloud, "post_once", side_effect=AssertionError("No API")), \
                    mock.patch.dict(sys.modules, {"joblib": None, "sklearn": None}):
                self.assertEqual(run.main(["--project-root", str(p.root), "--plan-only"]), 0)
            self.assertFalse(p.out.exists())
            code = "import sys, run_ltedi_frozen_test170 as r; r.evidence.EXPECTED_WEIGHTS=sys.argv.pop(1); raise SystemExit(r.main(sys.argv[1:]))"
            process = subprocess.run([sys.executable, "-S", "-c", code, sha, "--project-root", str(p.root), "--plan-only"],
                cwd=Path(run.__file__).parent, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)

    def test_freeze_precedes_first_test_read_labels_isolated_no_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, sha, splits = self.fixture(Path(tmp) / "project")
            original = Path.open
            reads = []
            def safe_open(path, *a, **kw):
                self.assertNotEqual(path.name, "train.jsonl")
                if path.name == "test.jsonl":
                    self.assertTrue((p.out / "frozen_protocol.json").is_file())
                    reads.append(path)
                return original(path, *a, **kw)
            with self.mocks(sha), mock.patch.object(Path, "open", safe_open):
                self.assertEqual(run.main(["--project-root", str(p.root)]), 0)
                with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                    run.main(["--project-root", str(p.root)])
            self.assertTrue(reads)
            rows = run.evidence.read_jsonl(p.out / "inputs_LOCAL_ONLY.jsonl")
            self.assertTrue(all(set(r) == {"id", "image", "text", "group_id"} for r in rows))
            done = run.prep.read_json(p.out / "preparation_complete.json")
            self.assertFalse(done["test_quality_computed"])
            self.assertFalse((p.out / "final_test_report.json").exists())

    def test_full170_pipeline_post_ledger_privacy_cost_and_one_final_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, sha, splits = self.fixture(Path(tmp) / "project")
            originals = {f: run.ev.digest(f) for f in p.root.rglob("*") if f.is_file()}
            self.prepare_and_infer(p, sha)
            original = Path.open
            def no_quality_open(path, *a, **kw):
                if path.name == "test_controls_DO_NOT_UPLOAD.jsonl" and a and a[0] != "rb":
                    raise AssertionError("Cloud must not evaluate test labels")
                return original(path, *a, **kw)
            calls = []
            def post(opener, body, key):
                self.assertEqual(key, "FAKE_SECRET_NEVER_PRINT")
                parsed = json.loads(body)
                self.assertEqual(set(parsed), {"model", "state", "questions"})
                for field in ('"label"', '"image"', '"local_id"', '"group_id"', "dev:", "train:"):
                    self.assertNotIn(field, body.decode())
                ledger = run.prep.read_json(p.out / "cloud/started_attempts" / f"{len(calls)+1:03d}.json")
                self.assertTrue(ledger["automatic_replay_forbidden"])
                self.assertEqual(ledger["request_sha256"], run.prep.object_digest(parsed))
                calls.append(body)
                return {"model": run.prep.MODEL, "violation_probability": .85 if "正例" in parsed["state"]["transcription"] else .1,
                    "input_tokens": 720, "output_tokens": 23}
            log = io.StringIO()
            with self.mocks(sha), mock.patch.object(Path, "open", no_quality_open), \
                    mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "FAKE_SECRET_NEVER_PRINT"}), \
                    mock.patch.object(run.urllib.request, "build_opener", return_value=object()), \
                    mock.patch.object(run.cloud, "post_once", side_effect=post), contextlib.redirect_stdout(log):
                self.assertEqual(run.main(["--project-root", str(p.root), "--stage", "cloud", "--plan-only"]), 0)
                self.assertEqual(calls, [])
                self.assertEqual(run.main(["--project-root", str(p.root), "--stage", "cloud", "--authorize-test-cloud"]), 0)
                with self.assertRaisesRegex(ValueError, "CLOUD_OUTPUT_EXISTS"):
                    run.main(["--project-root", str(p.root), "--stage", "cloud", "--authorize-test-cloud"])
            self.assertEqual(len(calls), 170)
            self.assertNotIn("FAKE_SECRET_NEVER_PRINT", log.getvalue())
            self.assertFalse((p.out / "final_test_report.json").exists())
            with self.mocks(sha):
                self.assertEqual(run.main(["--project-root", str(p.root), "--stage", "report"]), 0)
                sha_report = run.ev.digest(p.out / "final_test_report.json")
                self.assertEqual(run.main(["--project-root", str(p.root), "--stage", "report"]), 0)
                self.assertEqual(sha_report, run.ev.digest(p.out / "final_test_report.json"))
            report = run.prep.read_json(p.out / "final_test_report.json")
            self.assertFalse(report["test_threshold_tuning"])
            self.assertEqual(report["primary"]["n"], 170)
            self.assertAlmostEqual(report["billing"]["new_known_usage_cost_usd"], .0051408)
            self.assertEqual(originals, {f: run.ev.digest(f) for f in originals})
            for f in p.out.rglob("*.json"):
                self.assertNotIn("FAKE_SECRET_NEVER_PRINT", f.read_text(encoding="utf-8"))

    def test_local_interruption_resumes_only_unsaved_inferences(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, sha, splits = self.fixture(Path(tmp) / "project")
            with self.mocks(sha):
                run.main(["--project-root", str(p.root)])
                calls = []
                def interrupted(processor, model, row, modality, budget):
                    calls.append(row)
                    if len(calls) == 11:
                        raise RuntimeError("Synthetic local interruption")
                    return self.infer(processor, model, row, modality, budget)
                with mock.patch.object(run.ev, "infer_one", side_effect=interrupted), self.assertRaises(RuntimeError):
                    run.main(["--project-root", str(p.root), "--stage", "evidence"])
                self.assertFalse((p.out / ".evidence_worker.lock").exists())
                count = len(run.prep.read_json(p.out / "base_evidence.json")["items"])
                self.assertEqual(count, 10)
                with mock.patch.object(run.ev, "infer_one", side_effect=self.infer) as generator:
                    self.assertEqual(run.main(["--project-root", str(p.root), "--stage", "evidence", "--resume-local"]), 0)
                    self.assertEqual(generator.call_count, 330)

    def test_cloud_failure_no_retry_and_no_partial_test_evaluation(self):
        for failure in (TimeoutError("FAKE_SECRET_NEVER_PRINT"), urllib.error.HTTPError(run.cloud.ENDPOINT, 429, "FAKE_SECRET_NEVER_PRINT", {}, None)):
            with self.subTest(error=type(failure).__name__), tempfile.TemporaryDirectory() as tmp:
                p, sha, splits = self.fixture(Path(tmp) / "project")
                self.prepare_and_infer(p, sha)
                with self.mocks(sha), mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "FAKE_SECRET_NEVER_PRINT"}), \
                        mock.patch.object(run.urllib.request, "build_opener", return_value=object()), \
                        mock.patch.object(run.cloud, "post_once", side_effect=failure) as sender:
                    self.assertEqual(run.main(["--project-root", str(p.root), "--stage", "cloud", "--authorize-test-cloud"]), 2)
                    self.assertEqual(sender.call_count, 1)
                    with self.assertRaisesRegex(ValueError, "COMPLETE_TEST_REQUIRED"):
                        run.main(["--project-root", str(p.root), "--stage", "report"])
                summary = run.prep.read_json(p.out / "cloud/completion.json")
                self.assertEqual(summary["uncertain_attempts"], 1)
                self.assertFalse(summary["test_quality_computed"])
                self.assertFalse((p.out / "final_test_report.json").exists())

    def test_budget_and_changed_rule_or_payload_stop_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, sha, splits = self.fixture(Path(tmp) / "project")
            with self.mocks(sha), mock.patch.object(run, "SOFTWARE_NEW_FORECAST_USD", .01), self.assertRaisesRegex(ValueError, "BUDGET_FORECAST"):
                run.main(["--project-root", str(p.root), "--plan-only"])
            self.assertFalse(p.out.exists())
            self.prepare_and_infer(p, sha)
            path = p.out / "requests_LOCAL_ONLY.jsonl"
            request = run.evidence.read_jsonl(path)
            request[0]["state"]["gold_label"] = 1
            run.prep.atomic_jsonl(path, request)
            with self.mocks(sha), mock.patch.object(run.cloud, "post_once", side_effect=AssertionError("No API")), self.assertRaises(ValueError):
                run.main(["--project-root", str(p.root), "--stage", "cloud", "--plan-only"])
            self.assertFalse((p.out / "cloud").exists())

    def test_real_cpu_prediction_loads_artifacts_without_fit_or_test_metrics(self):
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        with tempfile.TemporaryDirectory() as tmp:
            p, sha, splits = self.fixture(Path(tmp) / "project")
            # Train only independent, synthetic unit-test fixtures. Never
            # fit anything inside the actual frozen-test implementation.
            texts = ["正例 合成风险攻击" + str(i) if i % 2 else "负例 合成文明安全" + str(i) for i in range(20)]
            labels = [i % 2 for i in range(20)]
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
            features = vectorizer.fit_transform(texts)
            classifier = LogisticRegression(solver="liblinear").fit(features, labels)
            source = run.prep.read_json(p.cpu / "validation_report.json")["source_split_sha256"]
            for name in run.MODEL_NAMES:
                joblib.dump({"input_modality": "text_only_ablation", "source_split_sha256": source,
                    "threshold": .7, "vectorizer": vectorizer, "classifier": classifier}, p.cpu / f"{name}.joblib")
            real_cpu = run.cpu_predictions
            with self.mocks(sha), mock.patch.object(run, "cpu_predictions", side_effect=real_cpu), \
                    mock.patch.object(TfidfVectorizer, "fit", side_effect=AssertionError("No fitting in final test")), \
                    mock.patch.object(TfidfVectorizer, "fit_transform", side_effect=AssertionError("No fitting in final test")), \
                    mock.patch.object(LogisticRegression, "fit", side_effect=AssertionError("No fitting in final test")):
                self.assertEqual(run.main(["--project-root", str(p.root)]), 0)
            result = run.prep.read_json(p.out / "cpu_predictions_LOCAL_ONLY.json")
            self.assertEqual(set(result["models"]), set(run.MODEL_NAMES))
            self.assertTrue(all(len(items) == 170 for items in result["models"].values()))
            self.assertFalse((p.out / "final_test_report.json").exists())

    def test_exact_payload_cache_does_not_charge_or_call_again(self):
        original = run.prep.make_request
        def identical_text(row, decision):
            return original({**row, "text": "完全相同的合成测试文本"}, decision)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(run.prep, "make_request", side_effect=identical_text):
            p, sha, splits = self.fixture(Path(tmp) / "project")
            self.prepare_and_infer(p, sha)
            with self.mocks(sha), mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC_ONLY"}), \
                    mock.patch.object(run.urllib.request, "build_opener", return_value=object()), \
                    mock.patch.object(run.cloud, "post_once", side_effect=AssertionError("Exact cache must not call")):
                self.assertEqual(run.main(["--project-root", str(p.root), "--stage", "cloud", "--authorize-test-cloud"]), 0)
            summary = run.prep.read_json(p.out / "cloud/completion.json")
            self.assertEqual(summary["exact_payload_cache_hits"], 170)
            self.assertEqual(summary["new_post_attempts"], 0)
            self.assertEqual(summary["new_known_usage_cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
