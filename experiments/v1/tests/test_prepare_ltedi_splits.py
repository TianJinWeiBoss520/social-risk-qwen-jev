import csv
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import prepare_ltedi_splits as prep
from download_ltedi_images import ImageSpec, file_blob_sha1


def row(index, label=None, split="train", **changes):
    rng = random.Random(index + 100)
    result = {
        "id": f"{split}:{index}.jpg", "original_split": split,
        "image": f"/fixture/{split}/{index}.jpg", "text": f"unique sample {index}",
        "label": index % 2 if label is None else label, "group_id": f"original-{index}",
        "pixel_sha256": f"pixels-{index}", "file_sha256": f"file-{index}",
        "dhash": rng.getrandbits(64), "size": (32, 32),
    }
    result.update(changes)
    return result


class SplitTests(unittest.TestCase):
    def pool(self):
        return [row(i) for i in range(40)]

    def test_normalization_retains_semantic_words(self):
        self.assertEqual(prep.normalize_text(" ＡBC\n\t 中文  "), "abc 中文")
        self.assertNotEqual(prep.normalize_text("不同文字"), prep.normalize_text("相同文字"))

    def test_group_split_is_reproducible_and_class_balanced(self):
        source = self.pool() + [row(100, split="dev"), row(101, split="dev")]
        a, _, _ = prep.make_splits(source, seed=123)
        b, _, _ = prep.make_splits(source, seed=123)
        self.assertEqual(a, b)
        self.assertEqual(len(a["val"]), 6)
        self.assertEqual(len(a["test"]), 2)
        for name, rows in a.items():
            self.assertEqual({r["label"] for r in rows}, {0, 1}, name)
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse({r["group_id"] for r in a[left]} & {r["group_id"] for r in a[right]})

    def test_exact_cross_split_input_prefers_dev_and_retains_originals(self):
        source = self.pool()
        source += [row(100, split="dev", text=source[0]["text"], pixel_sha256=source[0]["pixel_sha256"],
                       dhash=source[0]["dhash"], label=source[0]["label"]), row(101, split="dev")]
        splits, _, excluded = prep.make_splits(source)
        self.assertIn("dev:100.jpg", {r["id"] for r in splits["test"]})
        self.assertNotIn("train:0.jpg", {r["id"] for part in splits.values() for r in part})
        self.assertEqual(excluded[0]["reason"], "exact_image_and_normalized_text_duplicate")
        self.assertEqual(len(source), 42)

    def test_similar_images_are_split_guards_not_automatic_duplicate_removal(self):
        source = self.pool()
        source[1]["dhash"] = source[0]["dhash"] ^ 1
        assigned, summary = prep.assign_groups(source)
        self.assertEqual(len(assigned), 40)
        self.assertEqual(assigned[0]["group_id"], assigned[1]["group_id"])
        self.assertGreaterEqual(summary["near_image_candidate_pairs"], 1)
        self.assertTrue(summary["similarity_is_not_a_duplicate_or_harm_label"])

    def test_cross_split_candidate_guard_excludes_train_not_dev(self):
        source = self.pool()
        source += [row(100, split="dev", dhash=source[0]["dhash"] ^ 1), row(101, split="dev")]
        splits, _, exclusions = prep.make_splits(source)
        self.assertIn("train:0.jpg", {r["id"] for r in exclusions})
        self.assertIn("dev:100.jpg", {r["id"] for r in splits["test"]})
        self.assertTrue(all(r["original_split"] == "train" for name in ("train", "val") for r in splits[name]))

    def test_same_image_different_text_not_collapsed_and_identical_text_is_grouped(self):
        source = self.pool()
        source[1]["pixel_sha256"] = source[0]["pixel_sha256"]
        source[2]["text"] = source[3]["text"]
        splits, _, exclusions = prep.make_splits(source + [row(100, split="dev"), row(101, split="dev")])
        self.assertEqual(sum(map(len, splits.values())), 42)
        self.assertEqual(exclusions, [])
        by_id = {r["id"]: r for part in splits.values() for r in part}
        self.assertEqual(by_id["train:2.jpg"]["group_id"], by_id["train:3.jpg"]["group_id"])

    def test_conflicting_gold_labels_stop_instead_of_relabeling(self):
        source = self.pool()
        source[1].update(pixel_sha256=source[0]["pixel_sha256"], text=source[0]["text"])
        with self.assertRaisesRegex(RuntimeError, "LABEL_CONFLICT"):
            prep.make_splits(source)
        self.assertEqual(source[0]["label"], 0)
        self.assertEqual(source[1]["label"], 1)

    def fixture(self, root):
        specs = []
        for split, count in (("train", 24), ("dev", 4)):
            directory = root / f"Datasets/{split}/{split} images"
            directory.mkdir(parents=True)
            originals = []
            for index in range(count):
                path = directory / f"{index}.jpg"
                image = Image.new("RGB", (32, 32))
                rng = random.Random(index + (10000 if split == "dev" else 0))
                image.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(1024)])
                if split == "train" and index == 3:
                    image.save(path, format="GIF", save_all=True,
                               append_images=[Image.new("RGB", (32, 32), "white")], duration=100)
                else:
                    image.save(path, format="PNG")
                originals.append({"image_name": path.name, "labels": "Misogyny" if index % 2 else "Not-Misogyny",
                                  "transcriptions": f"{split} original text {index}"})
                specs.append(ImageSpec(split, path.relative_to(root).as_posix(), file_blob_sha1(path)))
            with (directory.parent / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_name", "labels", "transcriptions"])
                writer.writeheader()
                writer.writerows(originals)
        return specs

    def test_preparation_keeps_source_unchanged_and_records_no_full_screen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "raw", Path(temporary) / "prepared"
            specs = self.fixture(root)
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            with mock.patch.object(prep, "read_specs", return_value=specs):
                report = prep.prepare(root, output, user_spot_check=True)
            after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
            self.assertEqual(before, after)
            self.assertFalse(report["political_screening"]["full_dataset_verified"])
            self.assertEqual(len(report["quality_exclusions"]), 1)
            self.assertEqual(report["quality_exclusions"][0]["id"], "train:3.jpg")
            self.assertTrue((output / "train.jsonl").is_file())
            self.assertTrue((output / "val.jsonl").is_file())
            self.assertTrue((output / "test.jsonl").is_file())
            with self.assertRaisesRegex(RuntimeError, "OUTPUT_EXISTS_STOP"):
                prep.prepare(root, output, user_spot_check=True)

    def test_dry_run_writes_nothing_and_waiver_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "raw", Path(temporary) / "prepared"
            specs = self.fixture(root)
            with mock.patch.object(prep, "read_specs", return_value=specs):
                with self.assertRaisesRegex(RuntimeError, "accept-user-spot-check"):
                    prep.prepare(root, output)
                prep.prepare(root, output, user_spot_check=True, dry_run=True)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
