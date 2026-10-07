import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import audit_ltedi_images as audit


class AuditTests(unittest.TestCase):
    def test_pixel_duplicates_even_when_file_encodings_differ(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = Image.new("RGB", (32, 32), (30, 80, 140))
            a, b = root / "a.jpg", root / "b.jpg"
            image.save(a, format="PNG", compress_level=0)
            image.save(b, format="PNG", compress_level=9)
            ra = audit.decode_record(a, "train:a", "train", "Misogyny")
            rb = audit.decode_record(b, "dev:b", "dev", "Not-Misogyny")
            self.assertNotEqual(ra.file_sha256, rb.file_sha256)
            self.assertEqual(ra.pixel_sha256, rb.pixel_sha256)
            summary = audit.duplicate_summary([ra, rb], "pixel_sha256", 3)
            self.assertEqual(summary["cross_split_groups"], 1)
            self.assertEqual(summary["mixed_label_groups_for_review"], 1)

    def test_difference_hash_retains_gradient_direction(self):
        left, right = Image.new("L", (90, 80)), Image.new("L", (90, 80))
        left.putdata([x * 2 for _ in range(80) for x in range(90)])
        right.putdata([(89 - x) * 2 for _ in range(80) for x in range(90)])
        self.assertEqual((audit.difference_hash(left) ^ audit.difference_hash(right)).bit_count(), 64)

    def make_fixture(self, root, broken=False, unsafe=False):
        for split in ("train", "dev"):
            folder = root / f"Datasets/{split}/{split} images"
            folder.mkdir(parents=True)
            image = Image.new("RGB", (32, 32), (20, 70, 130))
            image.save(folder / "1.jpg", format="PNG", compress_level=0)
            if broken and split == "dev":
                (folder / "1.jpg").write_bytes(b"not an image")
            with (folder.parent / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_name", "labels", "transcriptions"])
                writer.writeheader()
                writer.writerow({"image_name": "../1.jpg" if unsafe else "1.jpg", "labels": "Misogyny", "transcriptions": "test text"})

    def test_full_audit_is_read_only_and_finds_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            before = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            with mock.patch.object(audit, "EXPECTED", {"train": 1, "dev": 1}):
                report = audit.audit(root)
            after = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertTrue(report["readability_check_passed"])
            self.assertEqual(report["exact_file_duplicates"]["cross_split_groups"], 1)
            self.assertEqual(report["near_duplicate_candidates"]["pairs"], 0)
            self.assertIn("NOT_DONE", report["political_content_screening"])

    def test_broken_file_is_reported_without_repair_or_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root, broken=True)
            with mock.patch.object(audit, "EXPECTED", {"train": 1, "dev": 1}):
                report = audit.audit(root)
            self.assertFalse(report["readability_check_passed"])
            self.assertEqual(report["unreadable_or_unsupported_count"], 1)
            self.assertEqual((root / "Datasets/dev/dev images/1.jpg").read_bytes(), b"not an image")

    def test_unsafe_annotation_name_stops_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root, unsafe=True)
            with mock.patch.object(audit, "EXPECTED", {"train": 1, "dev": 1}):
                with self.assertRaises(RuntimeError):
                    audit.audit(root)


if __name__ == "__main__":
    unittest.main()
