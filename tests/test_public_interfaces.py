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
        self.assertEqual(len(test["methods"]), 11)
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
        self.assertIn(render_table(), (root / "README.md").read_text(encoding="utf-8"))
        self.assertIn(render_table("validation"), (root / "docs/VALIDATION_RESULTS.md").read_text(encoding="utf-8"))

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
