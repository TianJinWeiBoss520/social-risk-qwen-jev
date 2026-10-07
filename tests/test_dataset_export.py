"""All content below is synthetic; never reads the owner's dataset or API key."""

import csv
import importlib.util
import json
import socket
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from social_risk_jev.decision.policy import QUESTION, prepare_request

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("private_export", ROOT / "scripts/export_dataset_badcases.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def put(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(ex.canonical(r) + "\n" for r in value) if jsonl else ex.canonical(value)
    path.write_text(text, encoding="utf-8")


def fixture(root):
    splits = {}
    number = 0
    for split, (count, positives) in ex.COUNTS.items():
        rows = []
        for i in range(count):
            origin = "dev" if split == "test" else "train"
            name = f"{i if origin == 'dev' else number}.jpg"
            path = root / f"data/raw/ltedi_cn_misogyny/Datasets/{origin}/{origin} images/{name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Byte-preservation fixture, deliberately not a claim of image decoding.
            path.write_bytes(b"SYNTHETIC_IMAGE_BYTES_" + name.encode())
            rows.append({"id": f"{origin}:{name}", "image": str(path.resolve()),
                         "text": '合成内容，含逗号,引号"与\n换行。联系 example@example.com',
                         "label": int(i >= count - positives), "group_id": f"{split}-group-{i}"})
            number += 1
        splits[split] = rows
        put(root / f"data/processed/ltedi_grouped_v1/{split}.jsonl", rows, True)
    put(root / "data/processed/ltedi_grouped_v1/preparation_report.json",
        {"source_commit": ex.COMMIT, "label_map": {"Not-Misogyny": 0, "Misogyny": 1},
         "splits": {k: {"rows": len(v)} for k, v in splits.items()}})
    for split in ("val", "test"):
        is_val = split == "val"
        directory = root / ("outputs/ltedi_evidence_val172_v1/lora_evidence" if is_val else
                            "outputs/ltedi_frozen_test170_v1")
        qpath = directory / ("predictions.json" if is_val else "lora_evidence.json")
        rpath = directory / ("jev_requests_LOCAL_ONLY.jsonl" if is_val else "requests_LOCAL_ONLY.jsonl")
        cpath = directory / "request_controls_DO_NOT_UPLOAD.jsonl"
        cloud_path = (root / "outputs/jev_lora_remaining152_cloud_v1/predictions_LOCAL_ONLY.json"
                      if is_val else directory / "cloud/predictions_LOCAL_ONLY.json")
        qwen, results, requests, controls = {}, {}, [], []
        positives_seen = negatives_seen = 0
        for index, row in enumerate(splits[split], 1):
            decision = {"label": row["label"], "gender_targeted": True, "attack_present": True,
                        "stance": "unclear", "image_role": "neutral", "needs_review": False,
                        "evidence": "完全合成的程序测试，不是实际样本或证据。"}
            qwen[row["id"]] = {"status": "ok", "decision": decision}
            body, _ = prepare_request(row["text"], decision)
            requests.append(body)
            controls.append({"request_index": index, "local_id": row["id"],
                             "gold_label_LOCAL_ONLY": row["label"],
                             "qwen_predicted_label_LOCAL_ONLY": decision["label"],
                             "prepared_request_sha256": ex.object_digest(body)} if is_val else
                            {"index": index, "local_id_DO_NOT_UPLOAD": row["id"],
                             "request_sha256": ex.object_digest(body)})
            if row["label"]:
                prediction = int(positives_seen >= ex.MATRICES[split][1][0])
                positives_seen += 1
            else:
                prediction = int(negatives_seen < ex.MATRICES[split][0][1])
                negatives_seen += 1
            results[row["id"]] = {"status": "ok", "model": ex.MODEL,
                                  "violation_probability": .8 if prediction else .1,
                                  "input_tokens": 20, "output_tokens": 2, "api_roundtrip_seconds": .02,
                                  "authorization_header": "SYNTHETIC_DO_NOT_EXPORT"}
        put(qpath, {"metadata": {"stage": "LORA_EVIDENCE", "adapter_active": True, "modality": "image-text"},
                    "items": qwen} if is_val else
                   {"stage": "LORA_EVIDENCE", "protocol_sha256": "placeholder", "items": qwen})
        put(rpath, requests, True)
        put(cpath, controls, True)
        if is_val:
            hashes = {str(p.resolve()): ex.file_digest(p) for p in
                      (qpath, rpath, cpath, root / "data/processed/ltedi_grouped_v1/val.jsonl")}
            put(cloud_path, {"metadata": {"source_sha256": hashes}, "items": results})
            put(cloud_path.parent / "report.json", {"complete": True, "valid_results": 172})
        else:
            protocol_path = directory / "frozen_protocol.json"
            put(protocol_path, {"jev_threshold": ex.THRESHOLD, "jev_model": ex.MODEL,
                                "jev_policy_sha256": ex.QUESTION_SHA256})
            digest = ex.file_digest(protocol_path)
            put(qpath, {"stage": "LORA_EVIDENCE", "protocol_sha256": digest, "items": qwen})
            put(cloud_path, {"protocol_sha256": digest, "items": results})
            put(directory / "cloud/completion.json", {"complete": True, "valid_n": 170})
            put(directory / "final_test_report.json", {"complete": True, "test_n": 170,
                "protocol_sha256": digest, "report_source_sha256": {str(cloud_path.resolve()): ex.file_digest(cloud_path)},
                "fixed_comparators": {"JEV_T040": {"confusion_matrix_true_rows_predicted_columns_0_1": ex.MATRICES[split]}}})
            put(directory / "evidence_complete.json", {"n": 170, "protocol_sha256": digest,
                "requests_sha256": ex.object_digest(requests),
                "source_sha256": {str(p.resolve()): ex.file_digest(p) for p in (qpath, rpath, cpath)}})
            test_path = root / "data/processed/ltedi_grouped_v1/test.jsonl"
            put(directory / "preparation_complete.json", {"prepared_source_sha256": {str(test_path.resolve()): ex.file_digest(test_path)}})
    return splits


class PrivateExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.splits = fixture(self.root)
        self.out = self.root / "outputs/private_dataset_badcases_v1"

    def plan(self, archive=False):
        return ex.build_plan(self.root, self.out, archive)

    def test_plan_read_only_and_no_network(self):
        before = {p.relative_to(self.root).as_posix(): ex.file_digest(p) for p in self.root.rglob("*") if p.is_file()}
        with patch.object(socket, "socket", side_effect=AssertionError("No network permitted")):
            plan = self.plan(True)
        self.assertEqual(len(plan["cases"]), 47)
        self.assertFalse(self.out.exists())
        after = {p.relative_to(self.root).as_posix(): ex.file_digest(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(ex.object_digest(QUESTION), ex.QUESTION_SHA256)

    def test_export_exact_csv_images_requests_and_private_archive(self):
        plan = self.plan(True)
        with patch.object(socket, "socket", side_effect=AssertionError("No network permitted")):
            summary = ex.export(plan)
        self.assertEqual(summary["csv_rows"], 1320)
        self.assertEqual(summary["badcase_counts"], {"val": {"false_positive": 3, "false_negative": 17},
                                                   "test": {"false_positive": 12, "false_negative": 15}})
        with (self.out / "data/labels.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["image_name", "label"])
        self.assertEqual(len(rows), 1321)
        self.assertTrue(all(len(r) == 2 for r in rows))
        self.assertEqual(len({r[0] for r in rows[1:]}), 1320)
        self.assertIn(["train/0.jpg", "0"], rows)
        self.assertIn(["test/0.jpg", "0"], rows)
        for split, inputs in self.splits.items():
            for row in inputs:
                image = self.out / "data" / split / Path(row["image"]).name
                self.assertEqual(image.read_bytes(), Path(row["image"]).read_bytes())
                self.assertEqual(image.with_suffix(".txt").read_text(encoding="utf-8"), row["text"])
        for case in plan["cases"]:
            folder = self.out / "badcase" / case["split"] / case["error_type"] / Path(case["row"]["image"]).stem
            self.assertEqual(ex.decode((folder / "jev_input.json").read_text(encoding="utf-8")), case["body"])
            self.assertNotIn("label", case["body"]["state"]["perception_hypotheses"])
            self.assertNotIn("SYNTHETIC_DO_NOT_EXPORT", (folder / "jev_output.json").read_text())
            self.assertFalse(ex.decode((folder / "result.json").read_text(encoding="utf-8"))["eligible_for_training"])
        with zipfile.ZipFile(plan["archive"]) as archive:
            self.assertIn("data/labels.csv", archive.namelist())
            self.assertEqual(archive.read("completion_PRIVATE.json"), (self.out / "completion_PRIVATE.json").read_bytes())
            self.assertTrue(all(not n.startswith(("/", "..")) for n in archive.namelist()))
        plan["sources"].verify()
        with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS"):
            self.plan()

    def test_source_changed_after_plan_stops_before_output(self):
        plan = self.plan()
        Path(self.splits["train"][0]["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            ex.export(plan)
        self.assertFalse(self.out.exists())

    def test_archive_failure_leaves_no_successful_completion(self):
        plan = self.plan(True)
        with patch.object(zipfile, "ZipFile", side_effect=OSError("synthetic archive failure")):
            with self.assertRaises(OSError):
                ex.export(plan)
        self.assertTrue(self.out.exists())  # Recoverable partial output remains.
        self.assertFalse((self.out / "completion_PRIVATE.json").exists())

    def test_corrupted_request_mapping_is_rejected(self):
        path = self.root / "outputs/ltedi_evidence_val172_v1/lora_evidence/request_controls_DO_NOT_UPLOAD.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.plan()
        self.assertFalse(self.out.exists())

    def test_incomplete_cache_is_not_success_only_export(self):
        path = self.root / "outputs/jev_lora_remaining152_cloud_v1/predictions_LOCAL_ONLY.json"
        value = ex.decode(path.read_text(encoding="utf-8"))
        value["items"].pop(next(iter(value["items"])))
        put(path, value)
        with self.assertRaises(ValueError):
            self.plan()

    def test_label_group_and_unsafe_id_guards(self):
        path = self.root / "data/processed/ltedi_grouped_v1/train.jsonl"
        rows = self.splits["train"]
        for key, value in (("label", True), ("group_id", "val-group-0"), ("id", "train:../unsafe.jpg")):
            original = rows[0][key]
            rows[0][key] = value
            put(path, rows, True)
            with self.assertRaises(ValueError):
                self.plan()
            rows[0][key] = original
        put(path, rows, True)

    def test_sources_cannot_be_export_targets(self):
        for out in (self.root, self.root.parent, self.root / "data/new", self.root / "src/new",
                    self.root / "outputs/ltedi_frozen_test170_v1/new"):
            with self.assertRaises(ValueError):
                ex.build_plan(self.root, out)

    def test_policy_labels_scores_and_nonfinite_json_guards(self):
        row = self.splits["val"][0]
        decision = {"label": 0, "gender_targeted": False, "attack_present": False, "stance": "unclear",
                    "image_role": "neutral", "needs_review": False, "evidence": "合成测试"}
        body, _ = prepare_request(row["text"], decision)
        body["state"]["perception_hypotheses"]["label"] = 0
        with self.assertRaises(ValueError):
            ex.checked_request(row, decision, body)
        for text in ('{"label":0,"label":1}', '{"value":NaN}'):
            with self.assertRaises(ValueError):
                ex.decode(text)
        for value in (True, float("nan"), 1.01, -.01):
            with self.assertRaises(ValueError):
                ex.checked_result({"status": "ok", "model": ex.MODEL, "violation_probability": value})

    def test_publication_only_allows_exact_document_paths(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import check_publication
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in check_publication.PUBLIC_SCAFFOLD:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Public documentation only", encoding="utf-8")
            self.assertEqual(check_publication.scan(root), [])
            for relative in ("data/labels.csv", "data/train/1.txt", "badcase/test/1/result.json", "badcase/other.md"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("private synthetic fixture", encoding="utf-8")
            self.assertEqual(len(check_publication.scan(root)), 4)
            (root / "data/README.md").write_text("ghp_" + "A"*36)
            self.assertEqual(len(check_publication.scan(root)), 5)


if __name__ == "__main__":
    unittest.main()
