import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np

# Local test-only dependency directory; never required by the AutoDL script.
test_dependencies = Path(__file__).parent / ".cpu_testdeps"
if test_dependencies.is_dir():
    sys.path.insert(0, str(test_dependencies))

import train_ltedi_cpu_baselines as baseline


class BaselineTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "data"
        data.mkdir()
        image = root / "placeholder.jpg"
        image.write_bytes(b"placeholder; baseline never decodes images")
        for name, count, offset in (("train", 24, 0), ("val", 8, 100)):
            with (data / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for index in range(count):
                    label = index % 2
                    row = {"id": f"train:{offset + index}.jpg", "image": str(image.resolve()),
                           "text": ("负类模拟 文明尊重 平等 " if label == 0 else "正类模拟 风险攻击 MOCK ") + str(offset + index),
                           "label": label, "group_id": f"group-{offset + index}"}
                    if name == "val":
                        row["text"] += " 🦉"
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (data / "test.jsonl").write_bytes(b"UNREADABLE_TEST_PLACEHOLDER")
        return data

    def test_metrics_expose_majority_accuracy_trap(self):
        truth = [0] * 123 + [1] * 49
        result = baseline.metrics(truth, [0.] * 172)
        self.assertAlmostEqual(result["accuracy"], 123 / 172)
        self.assertAlmostEqual(result["macro_f1"], .5 * 246 / 295)
        self.assertEqual(result["harmful_recall"], 0)
        self.assertAlmostEqual(result["pr_auc_average_precision"], 49 / 172)
        self.assertEqual(result["confusion_matrix_true_rows_predicted_columns_0_1"], [[123, 0], [49, 0]])

    def test_invalid_scores_do_not_produce_fake_metrics(self):
        for scores in ([0, float("nan")], [0, 1.1], [0]):
            with self.assertRaises(ValueError):
                baseline.metrics([0, 1], scores)

    def test_threshold_ties_prefer_point_five(self):
        result = baseline.select_threshold([0, 0, 1, 1], [.1, .2, .8, .9])
        self.assertEqual(result["threshold"], .5)
        self.assertEqual(result["macro_f1"], 1)

    def test_positive_score_respects_class_order(self):
        classifier = mock.Mock()
        classifier.classes_ = np.array([1, 0])
        classifier.predict_proba.return_value = np.array([[.8, .2], [.3, .7]])
        np.testing.assert_allclose(baseline.positive_scores(classifier, object()), [.8, .3])

    def test_overlap_or_dev_in_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = self.fixture(Path(temporary))
            val = baseline.load_rows(data / "val.jsonl")
            val[0]["group_id"] = "group-0"
            (data / "val.jsonl").write_text("".join(json.dumps(row) + "\n" for row in val), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "group_id"):
                baseline.load_train_validation(data)
            val[0]["group_id"] = "group-100"
            val[0]["id"] = "dev:100.jpg"
            (data / "val.jsonl").write_text("".join(json.dumps(row) + "\n" for row in val), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "original dev"):
                baseline.load_train_validation(data)

    def test_plan_only_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self.fixture(root)
            output = root / "output"
            result = baseline.train_baselines(data, output, plan_only=True)
            self.assertIsNone(result)
            self.assertFalse(output.exists())

    def run_end_to_end(self, skip_xgboost):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = self.fixture(root)
            output = root / "output"
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.iterdir()}
            original_open = Path.open

            def guard(path, *args, **kwargs):
                if path.name == "test.jsonl":
                    raise AssertionError("Final test must never be opened during training/selection")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", guard):
                report = baseline.train_baselines(data, output, skip_xgboost=skip_xgboost)
            after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.iterdir()}
            self.assertEqual(before, after)
            self.assertFalse(report["test_read"])
            self.assertEqual(report["stage"], "validation_model_selection_not_final_test")
            self.assertEqual(len(report["models"]), 2 if skip_xgboost else 4)
            for name in report["models"]:
                bundle = joblib.load(output / f"{name}.joblib")
                self.assertNotIn("🦉", bundle["vectorizer"].vocabulary_)
                rows = baseline.load_rows(data / "val.jsonl")
                scores = baseline.positive_scores(bundle["classifier"], bundle["vectorizer"].transform([r["text"] for r in rows]))
                self.assertEqual(len(scores), 8)
                predictions = [json.loads(line) for line in (output / f"{name}_validation_predictions.jsonl").read_text().splitlines()]
                self.assertEqual(set(predictions[0]), {"id", "score", "predicted_label"})
            self.assertTrue((output / "validation_report.json").is_file())
            with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS_STOP"):
                baseline.train_baselines(data, output, skip_xgboost=skip_xgboost)

    def test_logistic_end_to_end_train_only_vocabulary_and_untouched_test(self):
        self.run_end_to_end(skip_xgboost=True)

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("xgboost"), "Optional XGBoost not installed")
    def test_xgboost_end_to_end_cpu_only(self):
        self.run_end_to_end(skip_xgboost=False)


if __name__ == "__main__":
    unittest.main()
