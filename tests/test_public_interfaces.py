import hashlib
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from social_risk_jev.cli import main
from social_risk_jev.decision.policy import prepare_request
from social_risk_jev.decision.rules import probability, risk_band, review_decision
from social_risk_jev.evaluation.leaderboard import load_results, render_table
from social_risk_jev.evaluation.metrics import confusion_metrics
from social_risk_jev.perception.schema import consistency_flags, parse_evidence
from social_risk_jev.privacy import mask_text


def sample():
    return {"label": 0, "gender_targeted": False, "attack_present": False,
            "stance": "unclear", "image_role": "neutral", "needs_review": False, "evidence": "合成程序测试"}


class PublicInterfaces(unittest.TestCase):
    def test_strict_schema(self):
        self.assertEqual(parse_evidence(json.dumps(sample())), sample())
        for value in ({**sample(), "label": True}, {**sample(), "extra": 1}, {**sample(), "evidence": ""}):
            with self.assertRaises(ValueError):
                parse_evidence(json.dumps(value))
        for value in ('{"label":0,"label":1}', '{"label":NaN}'):
            with self.assertRaises(ValueError):
                parse_evidence(value)

    def test_text_only_cannot_claim_image(self):
        with self.assertRaises(ValueError):
            parse_evidence(json.dumps(sample()), modality="text-only")

    def test_boundaries_and_invalid_scores(self):
        for value, expected in [(0,"LOW"),(.19999,"LOW"),(.2,"MEDIUM"),(.79999,"MEDIUM"),(.8,"HIGH"),(1,"HIGH")]:
            self.assertEqual(risk_band(value), expected)
        for value in (True, float("nan"), float("inf"), -.1, 1.1, ".5"):
            with self.assertRaises(ValueError):
                probability(value)

    def test_review_never_deletes(self):
        self.assertTrue(review_decision(.01, text_label=1)["flagged"])
        self.assertFalse(review_decision(.9)["automatic_content_deletion"])
        self.assertEqual(review_decision(.01, needs_review=True)["action"], "queue_review")

    def test_request_excludes_labels_identifiers_images(self):
        value = sample()
        request, _ = prepare_request("联系 test@example.com", value)
        self.assertEqual(set(request), {"model", "state", "questions"})
        self.assertNotIn("label", request["state"]["perception_hypotheses"])
        self.assertNotIn("test@example.com", json.dumps(request))
        self.assertEqual(value, sample())
        request["questions"]["policy_violation"]["criteria"]["true"] = "changed"
        self.assertNotEqual(prepare_request("新合成示例")[0]["questions"]["policy_violation"]["criteria"]["true"], "changed")

    def test_flags_are_not_relabeling(self):
        value = {**sample(), "label": 1, "stance": "oppose"}
        self.assertEqual(len(consistency_flags(value)), 3)
        self.assertEqual(value["label"], 1)

    def test_metrics_and_complete_public_sets(self):
        test, val = load_results("test"), load_results("validation")
        self.assertEqual(len(test["methods"]), 13)
        self.assertEqual(len(val["methods"]), 38)
        for document in (test, val):
            for row in document["methods"]:
                matrix = row["confusion_matrix"]
                self.assertEqual(sum(matrix[0]), document["class_support"]["not_misogyny"])
                self.assertEqual(sum(matrix[1]), document["class_support"]["misogyny"])
        best = max(test["methods"], key=lambda r:r["metrics"]["macro_f1"])
        self.assertEqual(best["id"], "tfidf_logreg_c4")
        self.assertAlmostEqual(best["metrics"]["macro_f1"], .9035734543391946)
        with self.assertRaises(ValueError):
            confusion_metrics([[1,True],[1,1]])

    def test_readme_is_generated_from_same_results(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "README.md").exists():
            self.skipTest("README not available in wheel-only installation")
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn(render_table(), readme)
        self.assertIn(render_table("validation"), (root / "docs/VALIDATION_RESULTS.md").read_text(encoding="utf-8"))
        comparison_ids = {"multi_jev_t0.40", "multi_jev_t0.50",
                          "pure_jev_text_t040", "pure_jev_text_t050"}
        validation = load_results("validation")
        methods = sorted((row for row in validation["methods"] if row["id"] in comparison_ids),
                         key=lambda row: (-row["metrics"]["macro_f1"], row["id"]))
        self.assertEqual(len(methods), 4)
        self.assertEqual(validation["n"], 172)
        comparison = readme.split('## 纯 Jev 与多模态＋Jev：验证集对照', 1)[1].split('\n## ', 1)[0]
        for rank, row in enumerate(methods, 1):
            m = row["metrics"]
            scores = " | ".join(f"{100*m[key]:.2f}%" for key in
                                ("macro_f1", "accuracy", "harmful_precision", "harmful_recall"))
            self.assertIn(f"| {rank} | {row['name']} | {scores} | {m['false_positives']} | {m['false_negatives']} |",
                          comparison)
        self.assertIn('172 条验证样本', comparison)
        self.assertIn('170 条纯 Jev 补充测试已加入上方主表', comparison)
        self.assertFalse(any(row["id"] in comparison_ids for row in load_results("test")["methods"]))

    def test_supplementary_pure_jev_complete_scope_and_matched_thresholds(self):
        result = load_results("test")
        metadata = result["supplementary_test"]
        self.assertTrue(metadata["complete"])
        self.assertEqual(metadata["n"], 170)
        self.assertEqual(metadata["original_successes_reused"] + metadata["new_successful_posts"], 170)
        self.assertEqual(metadata["total_post_attempts_across_runs"], 171)
        self.assertTrue(metadata["old_timeout_billing_unknown"])
        self.assertRegex(metadata["owner_supplied_completion_log_sha256"], r"^[0-9a-f]{64}$")
        rows = {row["id"]: row for row in result["methods"]}
        frozen = [row for row in rows.values() if row["protocol"] == "frozen_test170_v1"]
        self.assertEqual(len(frozen), 11)
        for name, matrix, threshold, logged in (
            ("PURE_JEV_TEXT_T040_SUPPLEMENTARY", [[121, 2], [35, 12]], .4, (.6304, .7824, .8571, .2553)),
            ("PURE_JEV_TEXT_T050_SUPPLEMENTARY_CONTROL", [[122, 1], [35, 12]], .5, (.6357, .7882, .9231, .2553)),
        ):
            row = rows[name]
            self.assertEqual(row["confusion_matrix"], matrix)
            self.assertEqual(row["protocol"], "posthoc_supplementary_test_no_retuning")
            self.assertEqual(row["input"], "text-only-jev")
            self.assertEqual(row["threshold"], threshold)
            for metric, expected in zip(("macro_f1", "accuracy", "harmful_precision", "harmful_recall"), logged):
                self.assertEqual(round(row["metrics"][metric], 4), expected)
        self.assertAlmostEqual(rows["JEV_T040"]["metrics"]["harmful_recall"] -
                               rows["PURE_JEV_TEXT_T040_SUPPLEMENTARY"]["metrics"]["harmful_recall"], 20/47)

    def test_synthetic_demo_without_api(self):
        out = StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["demo"]), 0)
        self.assertIn("SYNTHETIC_CACHE_NOT_MODEL_INFERENCE", out.getvalue())
        self.assertIn('"api_calls": 0', out.getvalue())

    def test_archived_policy_matches_public_policy(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/v1/src"))
        import prepare_ltedi_jev_pilot as archived
        from social_risk_jev.decision.policy import QUESTION
        self.assertEqual(QUESTION, archived.QUESTION)

    def test_archived_source_bytes_match_manifest(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "experiments/v1/source_manifest.json").read_text(encoding="utf-8"))
        paths = {item["path"] for item in manifest["files"]}
        actual = {p.relative_to(root).as_posix() for directory in ("src", "tests")
                  for p in (root / "experiments/v1" / directory).glob("*.py")}
        self.assertEqual(paths, actual)
        for item in manifest["files"]:
            self.assertEqual(hashlib.sha256((root / item["path"]).read_bytes()).hexdigest(), item["sha256"])

    def test_local_document_links_exist(self):
        root = Path(__file__).resolve().parents[1]
        for path in (list(root.glob("*.md")) + list((root / "docs").glob("*.md"))):
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                relative = target.split("#", 1)[0]
                self.assertTrue((path.parent / relative).exists(), f"Broken link in {path.name}: {relative}")

    def test_secret_checker_reports_only_locations(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import check_publication
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = "ghp_" + "A"*36
            (root / "bad.txt").write_text(secret)
            (root / ".env").write_text("private fixture")
            findings = check_publication.scan(root)
            self.assertEqual(len(findings), 2)
            self.assertNotIn(secret, str(findings))
