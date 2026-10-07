import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evaluate_ltedi_qwen_base as probe
import prepare_ltedi_jev_pilot as prep
import probe_ltedi_lora_evidence20 as evidence
import test_prepare_ltedi_jev_pilot as preparation_tests


class FakeModel:
    def __init__(self):
        self.enabled = True

    @contextlib.contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


class EvidenceTests(unittest.TestCase):
    def fixture(self, root):
        rows, source, prep_argv = preparation_tests.PreparationTests().fixture(root)
        model, adapter = root / "model", root / "adapter"
        model.mkdir()
        adapter.mkdir()
        (model / "config.json").write_text('{"synthetic":true}', encoding="utf-8")
        cache = prep.read_json(source)
        cache["metadata"].update({"model_path": str(model),
                                  "model_config_sha256": probe.digest(model / "config.json"),
                                  "max_pixels": evidence.PIXELS, "max_new_tokens": evidence.NEW_TOKENS,
                                  "do_sample": False})
        source.write_text(prep.canonical(cache), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prep.main(prep_argv), 0)
        (adapter / "adapter_model.safetensors").write_bytes(b"SYNTHETIC_NO_REAL_WEIGHTS")
        (adapter / "adapter_config.json").write_text(prep.canonical({
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8,
            "lora_alpha": 16, "bias": "none", "modules_to_save": None}), encoding="utf-8")
        (adapter / "training_metadata.json").write_text(prep.canonical({
            "task_version": "ltedi_binary_json_label_v1", "trainable_scope": "language_attention_lora_only",
            "visual_encoder_frozen": True, "complete": True, "completed_steps": 245,
            "used_train_records": 978, "max_pixels": evidence.PIXELS, "base_model": str(model),
            "source_sha256": {"base_config": probe.digest(model / "config.json"),
                              "val": probe.digest(root / "data/val.jsonl")}}), encoding="utf-8")
        argv = ["--data-dir", str(root / "data"), "--evidence-source", str(source),
                "--cpu-baseline-dir", str(root / "cpu"), "--prepared-dir", str(root / "output"),
                "--adapter-path", str(adapter), "--model-path", str(model),
                "--output-dir", str(root / "new_output")]
        return rows, cache, argv, probe.digest(adapter / "adapter_model.safetensors")

    def test_plan_no_gpu_key_network_writes_or_train_test_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, argv, checksum = self.fixture(root)
            original_open = Path.open

            def guard(path, *args, **kwargs):
                if path.name in ("test.jsonl", "train.jsonl"):
                    raise AssertionError("Train/test read is not authorized")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(evidence, "EXPECTED_WEIGHTS", checksum), \
                    mock.patch.object(Path, "open", guard), \
                    mock.patch.dict(sys.modules, {"torch": None, "peft": None}), \
                    mock.patch("socket.create_connection", side_effect=AssertionError("No network")), \
                    mock.patch.object(prep.os, "getenv", side_effect=AssertionError("No key read")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evidence.main(argv + ["--plan-only"]), 0)
            self.assertFalse((root / "new_output").exists())
            code = "import sys, probe_ltedi_lora_evidence20 as e; e.EXPECTED_WEIGHTS=sys.argv.pop(1); raise SystemExit(e.main(sys.argv[1:]))"
            result = subprocess.run([sys.executable, "-S", "-c", code, checksum, *argv, "--plan-only"],
                                    cwd=Path(evidence.__file__).parent, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("BASE=20 LORA=20", result.stdout)

    def test_stale_preparation_files_and_adapter_are_rejected(self):
        for damage in ("weights", "config", "metadata", "controls", "requests", "report", "image"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows, _, argv, checksum = self.fixture(root)
                if damage == "weights":
                    (root / "adapter/adapter_model.safetensors").write_bytes(b"changed")
                elif damage == "config":
                    path = root / "adapter/adapter_config.json"
                    value = prep.read_json(path)
                    value["r"] = 16
                    path.write_text(prep.canonical(value), encoding="utf-8")
                elif damage == "metadata":
                    path = root / "adapter/training_metadata.json"
                    value = prep.read_json(path)
                    value["visual_encoder_frozen"] = False
                    path.write_text(prep.canonical(value), encoding="utf-8")
                elif damage == "report":
                    path = root / "output/preparation_report.json"
                    value = prep.read_json(path)
                    value["identity"]["selected_ids_LOCAL_ONLY"].reverse()
                    path.write_text(prep.canonical(value), encoding="utf-8")
                elif damage == "image":
                    Path(rows[0]["image"]).write_bytes(b"changed image")
                else:
                    name = "local_controls_DO_NOT_UPLOAD.jsonl" if damage == "controls" else "prepared_requests.jsonl"
                    path = root / "output" / name
                    contents = evidence.read_jsonl(path)
                    contents.pop()
                    path.write_text("".join(prep.canonical(v) + "\n" for v in contents), encoding="utf-8")
                with mock.patch.object(evidence, "EXPECTED_WEIGHTS", checksum), \
                        self.assertRaises((ValueError, KeyError)):
                    evidence.main(argv + ["--plan-only"])
                self.assertFalse((root / "new_output").exists())

    def run_mocked(self, root, invalid_lora=False, interrupted=False):
        _, cache, argv, checksum = self.fixture(root)
        original_report = prep.read_json(root / "output/preparation_report.json")
        selected = original_report["identity"]["selected_ids_LOCAL_ONLY"]
        model, calls = FakeModel(), []
        source_before = probe.digest(root / "output/prepared_requests.jsonl")

        def infer(processor, runtime, model_input, modality, budget):
            self.assertEqual(set(model_input), {"image", "text"})
            self.assertEqual(modality, "image-text")
            self.assertEqual(budget, 384)
            index = len(calls) % 20
            row_id = selected[index]
            calls.append((runtime.enabled, model_input["text"]))
            if interrupted and len(calls) == 3:
                raise RuntimeError("Synthetic interruption")
            if invalid_lora and runtime.enabled and index == 0:
                return {"status": "invalid_json", "raw_responses": ['{"label":0}'],
                        "error": "Unexpected JSON fields", "latency_seconds": .1}
            return {"status": "ok", "decision": dict(cache["items"][row_id]["decision"]),
                    "first_pass_schema_valid": True, "attempts": 1, "latency_seconds": .1}

        with mock.patch.object(evidence, "EXPECTED_WEIGHTS", checksum), \
                mock.patch.object(evidence, "load_matched_model", return_value=(object(), model)), \
                mock.patch.object(probe, "infer_one", side_effect=infer), \
                mock.patch("socket.create_connection", side_effect=AssertionError("No network")), \
                contextlib.redirect_stdout(io.StringIO()):
            if interrupted:
                with self.assertRaisesRegex(RuntimeError, "Synthetic interruption"):
                    evidence.main(argv)
                self.assertTrue(model.enabled)
                partial = prep.read_json(root / "new_output/base_evidence/predictions.json")
                self.assertEqual(len(partial["items"]), 2)
                self.assertFalse((root / "new_output/lora_evidence").exists())
            else:
                self.assertEqual(evidence.main(argv), 2 if invalid_lora else 0)
                self.assertEqual([enabled for enabled, _ in calls], [False] * 20 + [True] * 20)
                self.assertEqual([text for _, text in calls[:20]], [text for _, text in calls[20:]])
                with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                    evidence.main(argv)
        self.assertEqual(probe.digest(root / "output/prepared_requests.jsonl"), source_before)

    def test_matched_40_inferences_and_stripped_local_request_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_mocked(root)
            report = prep.read_json(root / "new_output/comparison_report.json")
            self.assertEqual(report["base"]["schema_valid_n"], 20)
            self.assertEqual(report["lora"]["schema_valid_n"], 20)
            self.assertEqual(report["paired_both_schema_valid"]["net_correct"], 0)
            self.assertEqual(report["api_calls"], 0)
            self.assertFalse(report["test_read"])
            for stage in ("base", "lora"):
                requests = evidence.read_jsonl(root / f"new_output/{stage}_evidence/jev_requests_LOCAL_ONLY.jsonl")
                self.assertEqual(len(requests), 20)
                for request in requests:
                    self.assertNotIn('"label"', prep.canonical(request))
                    self.assertEqual(set(request["state"]["perception_hypotheses"]), set(prep.FEATURES))

    def test_invalid_json_never_defaulted_or_exported_as_success_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_mocked(root, invalid_lora=True)
            report = prep.read_json(root / "new_output/comparison_report.json")
            self.assertEqual(report["lora"]["schema_valid_n"], 19)
            self.assertEqual(report["lora"]["pipeline_accuracy_errors_counted_wrong"], 19 / 20)
            self.assertEqual(report["paired_both_schema_valid"]["n"], 19)
            self.assertFalse(report["lora_request_preparation"]["ready"])
            directory = root / "new_output/lora_evidence"
            self.assertFalse((directory / "jev_requests_LOCAL_ONLY.jsonl").exists())
            records = prep.read_json(directory / "predictions.json")["items"]
            self.assertNotIn("decision", next(iter(records.values())))

    def test_interruption_preserves_partial_and_reenables_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.run_mocked(Path(temporary), interrupted=True)

    def test_gpu_load_failure_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, argv, checksum = self.fixture(root)
            with mock.patch.object(evidence, "EXPECTED_WEIGHTS", checksum), \
                    mock.patch.object(evidence, "load_matched_model", side_effect=RuntimeError("GPU_REQUIRED")), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "GPU_REQUIRED"):
                evidence.main(argv)
            self.assertFalse((root / "new_output").exists())

    def test_output_source_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, argv, checksum = self.fixture(root)
            with mock.patch.object(evidence, "EXPECTED_WEIGHTS", checksum), self.assertRaisesRegex(ValueError, "separate"):
                evidence.main(argv + ["--output-dir", str(root / "adapter/new") , "--plan-only"])

    def test_loaded_weights_must_match_and_be_frozen(self):
        class Tensor:
            requires_grad = False

            def detach(self):
                return self

            def cpu(self):
                return self

        class Runtime:
            def named_parameters(self):
                return live.items()

            def parameters(self):
                return live.values()

        torch = mock.Mock()
        torch.isfinite.return_value.all.return_value = True
        torch.equal.side_effect = lambda a, b: a is b
        live = {f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj.lora_A.default.weight": Tensor()
                for i in range(512)}
        saved = {name.replace(".default.", "."): value for name, value in live.items()}
        evidence.verify_loaded_adapter(Runtime(), saved, torch)
        saved[next(iter(saved))] = Tensor()
        with self.assertRaisesRegex(RuntimeError, "exactly match"):
            evidence.verify_loaded_adapter(Runtime(), saved, torch)
        saved = {name.replace(".default.", "."): value for name, value in live.items()}
        next(iter(live.values())).requires_grad = True
        with self.assertRaisesRegex(RuntimeError, "trainable"):
            evidence.verify_loaded_adapter(Runtime(), saved, torch)


if __name__ == "__main__":
    unittest.main()
