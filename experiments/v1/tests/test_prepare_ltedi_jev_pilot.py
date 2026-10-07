import contextlib
import hashlib
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


class PreparationTests(unittest.TestCase):
    def fixture(self, root):
        data, cache, cpu = root / "data", root / "cache", root / "cpu"
        for directory in (data, cache, cpu):
            directory.mkdir()
        rows = []
        for index in range(24):
            image = root / f"synthetic{index}.jpg"
            image.write_bytes(f"SYNTHETIC_NOT_A_REAL_IMAGE_{index}".encode())
            rows.append({"id": f"train:{index}.jpg", "group_id": f"g{index}",
                         "text": f"中文合成样本{index}", "label": index % 2, "image": str(image)})
        (data / "val.jsonl").write_text("".join(prep.canonical(row) + "\n" for row in rows), encoding="utf-8")
        (data / "test.jsonl").write_text("TEST_MUST_NOT_BE_READ", encoding="utf-8")
        image_hashes = [(row["id"], probe.digest(Path(row["image"]))) for row in rows]
        metadata = {
            "adapter": None, "modality": "image-text", "val_sha256": probe.digest(data / "val.jsonl"),
            "prompt_version": probe.PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(probe.SYSTEM_PROMPT.encode()).hexdigest(),
            "validation_image_bytes_sha256": hashlib.sha256(json.dumps(image_hashes, sort_keys=True).encode()).hexdigest(),
        }
        items = {}
        for row in rows:
            items[row["id"]] = {"status": "ok", "decision": {
                "label": row["label"], "gender_targeted": bool(row["label"]),
                "attack_present": bool(row["label"]), "stance": "support" if row["label"] else "unclear",
                "image_role": "neutral", "needs_review": False, "evidence": "程序测试用的合成证据",
            }}
        source = cache / "predictions.json"
        source.write_text(prep.canonical({"metadata": metadata, "items": items}), encoding="utf-8")
        (cpu / "validation_report.json").write_text(prep.canonical({
            "selected_text_baseline": "tfidf_logreg_c4", "source_split_sha256": {"val": metadata["val_sha256"]},
        }), encoding="utf-8")
        (cpu / "tfidf_logreg_c4_validation_predictions.jsonl").write_text("".join(
            prep.canonical({"id": row["id"], "predicted_label": 0}) + "\n" for row in rows), encoding="utf-8")
        args = ["--data-dir", str(data), "--evidence-source", str(source), "--cpu-baseline-dir", str(cpu),
                "--output-dir", str(root / "output")]
        return rows, source, args

    def test_selection_is_label_blind_deterministic_and_grouped(self):
        rows = [{"id": str(i), "group_id": f"g{i // 2}", "label": i % 2} for i in range(50)]
        selected = prep.select_rows(rows, 20, 2062)
        changed = [{**row, "label": 1 - row["label"]} for row in reversed(rows)]
        self.assertEqual([r["id"] for r in selected], [r["id"] for r in prep.select_rows(changed, 20, 2062)])
        self.assertEqual(len({row["group_id"] for row in selected}), 20)
        for limit in (0, 21, True):
            with self.assertRaises(ValueError):
                prep.select_rows(rows, limit, 2062)

    def test_request_has_no_labels_identifiers_paths_or_cpu_outputs(self):
        row = {"id": "LOCAL_SAMPLE_DO_NOT_SEND", "group_id": "LOCAL_GROUP_DO_NOT_SEND", "label": 1,
               "image": "/root/LOCAL_PATH_DO_NOT_SEND.jpg", "text": "普通日常分享"}
        decision = {"label": 0, "gender_targeted": False, "attack_present": False, "stance": "unclear",
                    "image_role": "neutral", "needs_review": False, "evidence": "普通生活内容"}
        request, _, _ = prep.make_request(row, decision)
        rendered = prep.canonical(request)
        for marker in ("LOCAL_SAMPLE_DO_NOT_SEND", "LOCAL_GROUP_DO_NOT_SEND", "LOCAL_PATH_DO_NOT_SEND", '"label"'):
            self.assertNotIn(marker, rendered)
        self.assertEqual(request["model"], "jev-1.13.0")
        self.assertEqual(set(request["state"]["perception_hypotheses"]), set(prep.FEATURES))
        self.assertEqual(request["questions"]["policy_violation"]["type"], "noul")

    def test_basic_contact_masking_but_no_claim_of_full_deidentification(self):
        clean, counts = prep.mask_text("13912345678 a@example.com https://example.org/a @测试号 110101199001011234 张三")
        self.assertEqual(dict(counts), {"url": 1, "email": 1, "cn_id": 1, "mobile": 1, "handle": 1})
        self.assertNotIn("13912345678", clean)
        self.assertIn("张三", clean)

    def test_plan_has_no_gpu_network_key_access_test_read_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, argv = self.fixture(root)
            original_open = Path.open

            def guard(path, *args, **kwargs):
                if path.name in ("test.jsonl", "train.jsonl"):
                    raise AssertionError("Only validation may be read")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", guard), mock.patch.dict(sys.modules, {"torch": None}), \
                    mock.patch("socket.create_connection", side_effect=AssertionError("No network")), \
                    mock.patch.object(prep.os, "getenv", side_effect=AssertionError("No key access")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prep.main(argv + ["--plan-only"]), 0)
            self.assertFalse((root / "output").exists())
            result = subprocess.run([sys.executable, "-S", str(Path(prep.__file__)), *argv, "--plan-only"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("API_CALLS=0", result.stdout)

    def test_written_requests_controls_and_provenance_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, argv = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prep.main(argv), 0)
            out = root / "output"
            requests = [json.loads(line) for line in (out / "prepared_requests.jsonl").read_text(encoding="utf-8").splitlines()]
            controls = [json.loads(line) for line in (out / "local_controls_DO_NOT_UPLOAD.jsonl").read_text(encoding="utf-8").splitlines()]
            report = prep.read_json(out / "preparation_report.json")
            self.assertEqual(len(requests), 20)
            self.assertEqual(len(controls), 20)
            for request, control in zip(requests, controls):
                self.assertEqual(control["prepared_request_sha256"], prep.object_digest(request))
                self.assertNotIn(control["local_id"], prep.canonical(request))
            self.assertEqual(report["api_calls"], 0)
            self.assertFalse(report["cloud_upload"])
            self.assertFalse(report["test_read"])
            self.assertIsNone(report["identity"]["perception_adapter"])
            self.assertFalse(report["privacy"]["manual_payload_review_done"])
            with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                prep.main(argv)

    def test_missing_rows_wrong_modality_adapter_and_stale_hash_are_rejected(self):
        for damage in ("missing", "text-only", "adapter", "manifest", "prompt", "image", "invalid_json"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                rows, source, argv = self.fixture(root)
                cache = prep.read_json(source)
                if damage == "missing":
                    cache["items"].pop(rows[0]["id"])
                elif damage == "text-only":
                    cache["metadata"]["modality"] = "text-only"
                elif damage == "adapter":
                    cache["metadata"]["adapter"] = "a_lora_directory"
                elif damage == "manifest":
                    cache["metadata"]["val_sha256"] = "changed"
                elif damage == "prompt":
                    cache["metadata"]["prompt_sha256"] = "changed"
                elif damage == "image":
                    Path(rows[0]["image"]).write_bytes(b"changed source bytes")
                else:
                    cache["items"][rows[0]["id"]]["decision"]["label"] = True
                source.write_text(prep.canonical(cache), encoding="utf-8")
                with self.assertRaises(ValueError):
                    prep.main(argv + ["--plan-only"])
                self.assertFalse((root / "output").exists())

    def test_test_source_ids_and_input_output_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, _, argv = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "separate"):
                prep.main(argv + ["--output-dir", str(root / "cache/export"), "--plan-only"])
            rows[0]["id"] = "dev:0.jpg"
            (root / "data/val.jsonl").write_text("".join(prep.canonical(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dev/test"):
                prep.main(argv + ["--plan-only"])

    def test_oversized_requests_fail_instead_of_silent_truncation(self):
        row = {"text": "字" * 12000}
        decision = {"gender_targeted": False, "attack_present": False, "stance": "unclear",
                    "image_role": "neutral", "needs_review": False, "evidence": "测试"}
        with self.assertRaisesRegex(ValueError, "REQUEST_TOO_LARGE"):
            prep.make_request(row, decision)


if __name__ == "__main__":
    unittest.main()
