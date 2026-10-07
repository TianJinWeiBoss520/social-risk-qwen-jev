import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evaluate_ltedi_qwen_base as probe
import prepare_ltedi_jev_pilot as prep
import probe_ltedi_lora_evidence20 as evidence
import run_ltedi_jev_cloud20 as cloud
import expand_ltedi_evidence_val172 as expand
from test_probe_ltedi_lora_evidence20 import FakeModel


class ExpansionTests(unittest.TestCase):
    def fixture(self, root):
        out = root / "outputs/ltedi_evidence_val172_v1"
        inputs = expand.settings(root, out)
        for path in (inputs.data_dir, inputs.evidence_source.parent, inputs.cpu_baseline_dir,
                     inputs.adapter_path, inputs.model_path):
            path.mkdir(parents=True)
        rows = []
        for index in range(172):
            image = root / f"synthetic_image_{index}.jpg"
            image.write_bytes(f"SYNTHETIC_BYTES_NOT_DECODED_{index}".encode())
            rows.append({"id": f"train:{index}.jpg", "group_id": f"g{index}", "image": str(image),
                         "text": f"中文程序测试样本{index}", "label": int(index < 49)})
        prep.atomic_jsonl(inputs.data_dir / "val.jsonl", rows)
        (inputs.data_dir / "train.jsonl").write_text("FORBIDDEN_TRAIN", encoding="utf-8")
        (inputs.data_dir / "test.jsonl").write_text("FORBIDDEN_TEST", encoding="utf-8")
        (inputs.model_path / "config.json").write_text('{"synthetic":true}', encoding="utf-8")
        val_sha = probe.digest(inputs.data_dir / "val.jsonl")
        image_hashes = [(row["id"], probe.digest(Path(row["image"]))) for row in rows]
        metadata = {"adapter": None, "modality": "image-text", "val_sha256": val_sha,
                    "prompt_version": probe.PROMPT_VERSION,
                    "prompt_sha256": hashlib.sha256(probe.SYSTEM_PROMPT.encode()).hexdigest(),
                    "validation_image_bytes_sha256": hashlib.sha256(
                        expand.json.dumps(image_hashes, sort_keys=True).encode()).hexdigest(),
                    "model_path": str(inputs.model_path),
                    "model_config_sha256": probe.digest(inputs.model_path / "config.json"),
                    "max_pixels": evidence.PIXELS, "max_new_tokens": evidence.NEW_TOKENS, "do_sample": False}
        decisions = {row["id"]: {"label": row["label"], "gender_targeted": bool(row["label"]),
                                "attack_present": bool(row["label"]), "stance": "unclear", "image_role": "neutral",
                                "needs_review": False, "evidence": "仅用于自动化测试的合成证据"} for row in rows}
        probe.atomic_json(inputs.evidence_source, {"metadata": metadata, "items": {
            key: {"status": "ok", "decision": decision} for key, decision in decisions.items()}})
        probe.atomic_json(inputs.cpu_baseline_dir / "validation_report.json", {
            "selected_text_baseline": "tfidf_logreg_c4", "source_split_sha256": {"val": val_sha}})
        prep.atomic_jsonl(inputs.cpu_baseline_dir / "tfidf_logreg_c4_validation_predictions.jsonl", [
            {"id": row["id"], "predicted_label": row["label"]} for row in rows])
        (inputs.adapter_path / "adapter_model.safetensors").write_bytes(b"FAKE_NO_REAL_MODEL")
        weight_sha = probe.digest(inputs.adapter_path / "adapter_model.safetensors")
        probe.atomic_json(inputs.adapter_path / "adapter_config.json", {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 16,
            "bias": "none", "modules_to_save": None})
        probe.atomic_json(inputs.adapter_path / "training_metadata.json", {
            "task_version": "ltedi_binary_json_label_v1", "trainable_scope": "language_attention_lora_only",
            "visual_encoder_frozen": True, "complete": True, "completed_steps": 245,
            "used_train_records": 978, "max_pixels": evidence.PIXELS, "base_model": str(inputs.model_path),
            "source_sha256": {"base_config": metadata["model_config_sha256"], "val": val_sha}})
        rows_by_image = {row["image"]: row for row in rows}

        def infer(processor, model, row, modality, budget):
            return {"status": "ok", "decision": dict(decisions[rows_by_image[row["image"]]["id"]]),
                    "first_pass_schema_valid": True, "attempts": 1, "latency_seconds": .2}

        prep_args = ["--data-dir", str(inputs.data_dir), "--evidence-source", str(inputs.evidence_source),
                     "--cpu-baseline-dir", str(inputs.cpu_baseline_dir), "--output-dir", str(inputs.prepared_dir)]
        evidence_args = prep_args[:-2] + ["--prepared-dir", str(inputs.prepared_dir),
            "--adapter-path", str(inputs.adapter_path), "--model-path", str(inputs.model_path),
            "--output-dir", str(root / "outputs/ltedi_lora_evidence20_v1")]
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(evidence, "EXPECTED_WEIGHTS", weight_sha), \
                mock.patch.object(evidence, "load_matched_model", return_value=(object(), FakeModel())), \
                mock.patch.object(probe, "infer_one", side_effect=infer):
            self.assertEqual(prep.main(prep_args), 0)
            self.assertEqual(evidence.main(evidence_args), 0)
        pilot = root / "outputs/ltedi_lora_evidence20_v1"
        request_sha = prep.object_digest(evidence.read_jsonl(pilot / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl"))
        calls = []

        def fake_post(opener, body, key):
            calls.append(body)
            return {"model": prep.MODEL, "violation_probability": .1, "input_tokens": 700, "output_tokens": 23}

        # Build the old 20 result/ledger fixtures through the real writer,
        # intercepting every POST. No test uses a network service or real key.
        with mock.patch.object(cloud, "ROOT", root), mock.patch.object(cloud, "EXPECTED_REQUESTS", request_sha), \
                mock.patch.object(cloud, "post_once", side_effect=fake_post), \
                mock.patch.object(cloud.urllib.request, "build_opener", return_value=object()), \
                mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "SYNTHETIC_ONLY_NOT_A_KEY"}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cloud.main(["--authorize-20-cloud-calls"]), 0)
        self.assertEqual(len(calls), 20)
        return rows, infer, weight_sha, request_sha

    @contextlib.contextmanager
    def guarded(self, weight_sha, request_sha):
        original_open = Path.open

        def guard(path, *args, **kwargs):
            if path.name in ("train.jsonl", "test.jsonl"):
                raise AssertionError("Train/test must stay unread")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(evidence, "EXPECTED_WEIGHTS", weight_sha), \
                mock.patch.object(cloud, "EXPECTED_REQUESTS", request_sha), \
                mock.patch.object(Path, "open", guard), \
                mock.patch.object(cloud, "post_once", side_effect=AssertionError("No cloud calls")), \
                mock.patch("socket.create_connection", side_effect=AssertionError("No network")), \
                mock.patch.object(os, "getenv", side_effect=AssertionError("No key read")), \
                contextlib.redirect_stdout(io.StringIO()):
            yield

    def test_plan_is_stdlib_read_only_and_complement_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            _, _, weights, requests = self.fixture(root)
            with self.guarded(weights, requests), mock.patch.dict(sys.modules, {"torch": None, "peft": None}), \
                    mock.patch.object(evidence, "load_matched_model", side_effect=AssertionError("No model")):
                self.assertEqual(expand.main(["--project-root", str(root), "--plan-only"]), 0)
            self.assertFalse((root / "outputs/ltedi_evidence_val172_v1").exists())
            code = ("import sys, expand_ltedi_evidence_val172 as e; "
                    "e.evidence.EXPECTED_WEIGHTS=sys.argv.pop(1); "
                    "e.cloud.EXPECTED_REQUESTS=sys.argv.pop(1); raise SystemExit(e.main(sys.argv[1:]))")
            result = subprocess.run([sys.executable, "-S", "-c", code, weights, requests,
                                     "--project-root", str(root), "--plan-only"],
                                    cwd=Path(expand.__file__).parent, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"BASE_EVIDENCE":152', result.stdout)

    def run_mock(self, root, rows, infer, weights, requests, interrupted=False, invalid=False):
        model, calls = FakeModel(), []
        interruption_fired = False
        pilot = prep.read_json(root / "outputs/ltedi_lora_evidence20_v1/plan.json")["selected_ids_LOCAL_ONLY"]
        remaining = [row for row in rows if row["id"] not in pilot]
        by_image = {row["image"]: row["id"] for row in rows}

        def tracked(processor, runtime, row, modality, budget):
            nonlocal interruption_fired
            self.assertEqual(set(row), {"image", "text"})
            self.assertEqual((modality, budget), ("image-text", 384))
            key = by_image[row["image"]]
            self.assertNotIn(key, pilot)
            calls.append((runtime.enabled, key))
            if interrupted and not interruption_fired and len(calls) == 9:
                interruption_fired = True
                raise RuntimeError("Synthetic interruption")
            if invalid and runtime.enabled and key == remaining[0]["id"]:
                return {"status": "invalid_json", "raw_responses": ["{}"], "latency_seconds": .2}
            return infer(processor, runtime, row, modality, budget)

        with self.guarded(weights, requests), \
                mock.patch.object(evidence, "load_matched_model", return_value=(object(), model)), \
                mock.patch.object(probe, "infer_one", side_effect=tracked):
            argv = ["--project-root", str(root)]
            if interrupted:
                with self.assertRaisesRegex(RuntimeError, "Synthetic interruption"):
                    expand.main(argv)
                self.assertTrue(model.enabled)
                out = root / "outputs/ltedi_evidence_val172_v1"
                self.assertFalse((out / ".local_worker.lock").exists())
                self.assertEqual(len(prep.read_json(out / "base_evidence/predictions.json")["items"]), 28)
                calls.clear()
                self.assertEqual(expand.main(argv + ["--resume", "--plan-only"]), 0)
                self.assertEqual(calls, [])
                self.assertEqual(expand.main(argv + ["--resume"]), 0)
                self.assertEqual(len(calls), 296)
                self.assertEqual([key for enabled, key in calls if not enabled], [r["id"] for r in remaining[8:]])
            else:
                self.assertEqual(expand.main(argv), 2 if invalid else 0)
                self.assertEqual(calls, [(enabled, r["id"]) for enabled in (False, True) for r in remaining])
            with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                expand.main(argv)
        return root / "outputs/ltedi_evidence_val172_v1"

    def test_full_304_generations_exports152_no_replay_no_privacy_leak(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            rows, infer, weights, requests = self.fixture(root)
            old_files = [path for directory in ("ltedi_lora_evidence20_v1", "jev_lora_val20_cloud_v1")
                         for path in (root / "outputs" / directory).rglob("*") if path.is_file()]
            before = {path: probe.digest(path) for path in old_files}
            out = self.run_mock(root, rows, infer, weights, requests)
            report = prep.read_json(out / "comparison_report.json")
            self.assertEqual(report["lora"]["schema_valid_n"], 172)
            self.assertEqual(report["base"]["schema_valid_n"], 172)
            self.assertEqual(report["api_calls"], 0)
            self.assertFalse(report["jev_full_validation_results_available"])
            bodies = evidence.read_jsonl(out / "remaining152_for_jev/jev_requests_LOCAL_ONLY.jsonl")
            self.assertEqual(len(bodies), 152)
            for body in bodies:
                self.assertEqual(set(body), {"model", "state", "questions"})
                self.assertEqual(set(body["state"]["perception_hypotheses"]), set(prep.FEATURES))
                self.assertNotIn('"label"', prep.canonical(body))
                self.assertNotIn('"image"', prep.canonical(body))
                self.assertNotIn("train:", prep.canonical(body))
            self.assertEqual(before, {path: probe.digest(path) for path in old_files})
            with self.guarded(weights, requests), \
                    mock.patch.object(evidence, "load_matched_model", side_effect=AssertionError("Completed resume needs no GPU")):
                self.assertEqual(expand.main(["--project-root", str(root), "--resume"]), 0)

    def test_interrupted_resume_skips_all_completed_local_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            rows, infer, weights, requests = self.fixture(root)
            self.run_mock(root, rows, infer, weights, requests, interrupted=True)

    def test_invalid_output_blocks_full_and_remaining_lora_request_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            rows, infer, weights, requests = self.fixture(root)
            out = self.run_mock(root, rows, infer, weights, requests, invalid=True)
            report = prep.read_json(out / "comparison_report.json")
            self.assertEqual(report["lora"]["schema_valid_n"], 171)
            self.assertFalse(report["remaining152_request_preparation"]["ready"])
            self.assertFalse((out / "remaining152_for_jev").exists())
            self.assertFalse((out / "lora_evidence/jev_requests_LOCAL_ONLY.jsonl").exists())

    def test_cloud_ledger_tampering_and_gpu_failure_stop_without_output(self):
        for scenario in ("ledger", "gpu"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "project"
                _, _, weights, requests = self.fixture(root)
                if scenario == "ledger":
                    path = root / "outputs/jev_lora_val20_cloud_v1/started_attempts/01.json"
                    value = prep.read_json(path)
                    value["request_sha256"] = "changed"
                    probe.atomic_json(path, value)
                with self.guarded(weights, requests), \
                        mock.patch.object(evidence, "load_matched_model", side_effect=RuntimeError("GPU_REQUIRED")), \
                        self.assertRaisesRegex((ValueError, RuntimeError), "ledger|GPU_REQUIRED"):
                    expand.main(["--project-root", str(root)])
                self.assertFalse((root / "outputs/ltedi_evidence_val172_v1").exists())

    def test_source_change_during_generation_blocks_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            rows, infer, weights, requests = self.fixture(root)
            changed = False

            def changing(processor, model, row, modality, budget):
                nonlocal changed
                if not changed:
                    Path(rows[-1]["image"]).write_bytes(b"CHANGED_DURING_INFERENCE")
                    changed = True
                return infer(processor, model, row, modality, budget)

            with self.guarded(weights, requests), \
                    mock.patch.object(evidence, "load_matched_model", return_value=(object(), FakeModel())), \
                    mock.patch.object(probe, "infer_one", side_effect=changing), \
                    self.assertRaisesRegex(RuntimeError, "image bytes changed"):
                expand.main(["--project-root", str(root)])
            out = root / "outputs/ltedi_evidence_val172_v1"
            self.assertFalse((out / "comparison_report.json").exists())
            self.assertFalse((out / "remaining152_for_jev").exists())
            self.assertFalse((out / ".local_worker.lock").exists())

    def test_resume_rejects_modified_seed_and_concurrent_writer(self):
        for damage in ("seed", "lock"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "project"
                rows, infer, weights, requests = self.fixture(root)
                out = self.run_mock(root, rows, infer, weights, requests)
                if damage == "seed":
                    path = out / "lora_evidence/predictions.json"
                    value = prep.read_json(path)
                    plan = prep.read_json(out / "plan.json")
                    value["items"][plan["pilot_ids_LOCAL_ONLY"][0]]["decision"]["needs_review"] = True
                    probe.atomic_json(path, value)
                else:
                    (out / ".local_worker.lock").write_text("EXISTING_WORKER", encoding="utf-8")
                with self.guarded(weights, requests), \
                        mock.patch.object(evidence, "load_matched_model", side_effect=AssertionError("No model on failed resume")), \
                        self.assertRaisesRegex(ValueError, "reused 20|LOCK_EXISTS"):
                    expand.main(["--project-root", str(root), "--resume"])


if __name__ == "__main__":
    unittest.main()
