import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prepare_ltedi_chain_review23 as run
from test_run_ltedi_jev_text_vs_multimodal172 import ContrastTests


class ReviewTests(unittest.TestCase):
    def fixture(self, root, unsafe_text=False):
        rows, local, old, _ = ContrastTests().fixture(root)
        both_wrong = set(range(13)) | {50, 51, 52}
        qwen_only_wrong = {17, 53, 54}
        jev_only_wrong = {13, 14, 15, 16}
        qwen = run.prep.read_json(local / "lora_evidence/predictions.json")
        saved = run.prep.read_json(old / "predictions_LOCAL_ONLY.json")
        for index, row in enumerate(rows):
            key = row["id"]
            if index in both_wrong | qwen_only_wrong:
                qwen["items"][key]["decision"]["label"] = 1 - row["label"]
            if index in both_wrong | jev_only_wrong:
                saved["items"][key]["violation_probability"] = .1 if row["label"] else .9
        if unsafe_text:
            rows[0]["text"] = '<script>fetch("https://never.example")</script> <img src=x onerror=alert(1)>'
            run.prep.atomic_jsonl(root / "data/processed/ltedi_grouped_v1/val.jsonl", rows)
        run.ev.atomic_json(local / "lora_evidence/predictions.json", qwen)
        plan = run.prep.read_json(local / "plan.json")
        image_hashes = [(r["id"], run.ev.digest(Path(r["image"]))) for r in rows]
        plan["source_validation_image_bytes_sha256"] = hashlib.sha256(json.dumps(image_hashes, sort_keys=True).encode()).hexdigest()
        run.ev.atomic_json(local / "plan.json", plan)
        old_plan = run.prep.read_json(old / "plan.json")
        old_plan["source_sha256"] = {name: run.ev.digest(Path(name)) for name in old_plan["source_sha256"]}
        saved["metadata"] = old_plan
        run.ev.atomic_json(old / "plan.json", old_plan)
        run.ev.atomic_json(old / "predictions_LOCAL_ONLY.json", saved)
        report = run.prep.read_json(old / "report.json")
        report["jev_classification_valid_only"] = run.ev.classification_metrics([r["label"] for r in rows],
            [int(saved["items"][r["id"]]["violation_probability"] >= .5) for r in rows])
        run.ev.atomic_json(old / "report.json", report)
        return rows, local, old, root / "reports" / run.OUT_NAME

    @contextlib.contextmanager
    def guards(self, no_image_read=False):
        original_open = Path.open
        original_get = os.environ.get

        def safe_open(path, *args, **kwargs):
            if path.name in {"train.jsonl", "test.jsonl"}:
                raise AssertionError("No train or test manifests")
            if no_image_read and path.suffix.lower() == ".jpg":
                raise AssertionError("Plan must not decode or read image bytes")
            return original_open(path, *args, **kwargs)

        def safe_get(name, *args):
            if name == "TYPESAFE_API_KEY":
                raise AssertionError("No key access")
            return original_get(name, *args)

        with mock.patch.object(Path, "open", safe_open), \
                mock.patch.object(os.environ, "get", side_effect=safe_get), \
                mock.patch("socket.create_connection", side_effect=AssertionError("No network")), \
                mock.patch.object(run.contrast.cloud, "post_once", side_effect=AssertionError("No POST")), \
                mock.patch.dict(sys.modules, {"torch": None, "peft": None}), \
                contextlib.redirect_stdout(io.StringIO()):
            yield

    def test_plan_selects23_with_priority_without_images_key_gpu_test_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            _, _, _, out = self.fixture(root)
            with self.guards(no_image_read=True), mock.patch.dict(sys.modules, {"PIL": None}):
                self.assertEqual(run.main(["--project-root", str(root), "--plan-only"]), 0)
                _, meta, rows, selected, _ = run.load_plan(root)
            self.assertFalse(out.exists())
            self.assertEqual(meta["category_counts"], {
                "BOTH_WRONG": 16, "QWEN_CORRECT_JEV_WRONG": 4,
                "QWEN_WRONG_JEV_CORRECT": 3, "BOTH_CORRECT": 149})
            self.assertEqual(meta["jev_errors"], {"FN": 17, "FP": 3})
            self.assertEqual(len(selected), 23)
            self.assertEqual([r["category"] for r in selected[:4]], [run.ORDER[0]] * 4)
            self.assertTrue(all(r["source_split"] == "val" and not r["eligible_for_training"] for r in selected))
            self.assertEqual(len(rows), 172)

    def test_export_private_self_contained_escaped_report_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            _, local, old, out = self.fixture(root, unsafe_text=True)
            sources = [f for directory in (local, old) for f in directory.rglob("*") if f.is_file()]
            hashes = {f: run.ev.digest(f) for f in sources}
            with self.guards(), mock.patch.object(run, "jpeg_data", return_value="data:image/jpeg;base64,AA==") as encoder:
                self.assertEqual(run.main(["--project-root", str(root)]), 0)
                with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                    run.main(["--project-root", str(root)])
            self.assertEqual(encoder.call_count, 23)
            document = (out / "review_PRIVATE.html").read_text(encoding="utf-8")
            self.assertEqual(document.count("<article data-id="), 23)
            self.assertIn("Content-Security-Policy", document)
            self.assertIn("connect-src &#x27;none&#x27;", document)
            self.assertNotIn('<script>fetch(', document)
            self.assertIn('&lt;script&gt;fetch(', document)
            self.assertNotIn('<img src=x onerror=', document)
            self.assertIn("eligible_for_training:false", document)
            self.assertIn("VALIDATION_DIAGNOSIS_ONLY_NOT_TRAINING_DATA", document)
            metadata = run.prep.read_json(out / "manifest_DIAGNOSIS_ONLY.json")["metadata"]
            self.assertEqual(metadata["training_records_created"], 0)
            self.assertEqual(metadata["api_calls"], 0)
            self.assertEqual(hashes, {f: run.ev.digest(f) for f in sources})

    def test_changed_image_stops_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            rows, _, _, out = self.fixture(root)
            Path(rows[0]["image"]).write_bytes(b"CHANGED_BYTES")
            with self.guards(), mock.patch.object(run, "jpeg_data", side_effect=AssertionError("No encoding")):
                with self.assertRaisesRegex(ValueError, "IMAGES_CHANGED"):
                    run.main(["--project-root", str(root)])
            self.assertFalse(out.exists())

    def test_changed_cache_metric_and_source_fail_in_plan(self):
        for target in ("source", "metric", "model"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "project"
                _, local, old, out = self.fixture(root)
                if target == "source":
                    with (local / "lora_evidence/predictions.json").open("a", encoding="utf-8") as f:
                        f.write("\n")
                elif target == "metric":
                    report = run.prep.read_json(old / "report.json")
                    report["jev_classification_valid_only"]["accuracy"] = .123
                    run.ev.atomic_json(old / "report.json", report)
                else:
                    saved = run.prep.read_json(old / "predictions_LOCAL_ONLY.json")
                    saved["items"]["train:0.jpg"]["model"] = "jev-latest"
                    run.ev.atomic_json(old / "predictions_LOCAL_ONLY.json", saved)
                with self.guards(no_image_read=True):
                    with self.assertRaises(ValueError):
                        run.main(["--project-root", str(root), "--plan-only"])
                self.assertFalse(out.exists())

    def test_jpeg_embedding_reads_real_image_without_gpu(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.png"
            Image.new("RGBA", (10, 12), (20, 40, 60, 255)).save(path)
            with self.guards():
                data = run.jpeg_data(path)
            self.assertTrue(data.startswith("data:image/jpeg;base64,"))
            self.assertGreater(len(data), 100)


if __name__ == "__main__":
    unittest.main()
