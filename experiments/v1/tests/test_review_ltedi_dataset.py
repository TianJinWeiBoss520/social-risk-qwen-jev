import csv
import hashlib
import http.client
import json
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import unittest
from contextlib import closing
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from PIL import Image

import review_ltedi_dataset as review
from download_ltedi_images import ImageSpec, file_blob_sha1


class ReviewTests(unittest.TestCase):
    def fixture(self, root, animated=False):
        specs = []
        for split in ("train", "dev"):
            directory = root / f"Datasets/{split}/{split} images"
            directory.mkdir(parents=True)
            path = directory / "1.jpg"
            image = Image.new("RGB", (32, 32), (120, 80, 140))
            if animated and split == "train":
                image.save(path, format="GIF", save_all=True,
                           append_images=[Image.new("RGB", (32, 32), "yellow")], duration=100, loop=0)
            else:
                image.save(path, format="PNG")
            with (directory.parent / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_name", "labels", "transcriptions"])
                writer.writeheader()
                writer.writerow({"image_name": "1.jpg", "labels": "Misogyny",
                                 "transcriptions": "<img src=x onerror=alert(1)> 待审核内容"})
            specs.append(ImageSpec(split, path.relative_to(root).as_posix(), file_blob_sha1(path)))
        return specs

    def prepared(self, temporary, animated=False):
        root, output = Path(temporary) / "raw", Path(temporary) / "review"
        specs = self.fixture(root, animated)
        with mock.patch.object(review, "read_specs", return_value=specs):
            review.prepare(root, output)
        return root, output, specs

    def test_prepare_preserves_all_source_files_and_finds_exact_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "raw", Path(temporary) / "review"
            specs = self.fixture(root)
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            with mock.patch.object(review, "read_specs", return_value=specs):
                review.prepare(root, output)
            after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
            self.assertEqual(before, after)
            store = review.ReviewStore(output)
            self.assertEqual(len(store.pairs), 1)
            self.assertTrue(next(iter(store.pairs.values()))["cross_split"])
            self.assertEqual(store.status()["content_pending"], 2)
            self.assertFalse(store.status()["training_ready"])
            with self.assertRaises(RuntimeError):
                review.prepare(root, output)

    def test_multi_frame_is_quality_excluded_not_deleted_or_first_frame_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, _ = self.prepared(temporary, animated=True)
            store = review.ReviewStore(output)
            self.assertEqual(set(store.rows), {"dev:1.jpg"})
            self.assertEqual(store.manifest["quality_exclusions"][0]["id"], "train:1.jpg")
            with Image.open(root / "Datasets/train/train images/1.jpg") as image:
                self.assertEqual(image.n_frames, 2)
            with self.assertRaises(ValueError):
                store.preview("train:1.jpg")

    def test_source_bytes_changed_stop_before_metadata_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "raw", Path(temporary) / "review"
            specs = self.fixture(root)
            (root / specs[0].relative).write_bytes(b"changed")
            with mock.patch.object(review, "read_specs", return_value=specs):
                with self.assertRaisesRegex(RuntimeError, "SOURCE_BYTES_CHANGED"):
                    review.prepare(root, output)
            self.assertFalse(output.exists())

    def test_decisions_persist_and_revisions_append_audit_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, output, _ = self.prepared(temporary)
            store = review.ReviewStore(output)
            store.save("content", "train:1.jpg", "keep", "first review")
            store.save("content", "train:1.jpg", "exclude_uncertain", "rechecked")
            restored = review.ReviewStore(output)
            self.assertEqual(restored.status()["content_pending"], 1)
            self.assertEqual(restored.decisions()[0]["decision"], "exclude_uncertain")
            with closing(sqlite3.connect(store.database)) as db, db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 2)

    def test_invalid_decisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, output, _ = self.prepared(temporary)
            store = review.ReviewStore(output)
            for args in (("content", "bad", "keep"), ("content", "train:1.jpg", "Misogyny"),
                         ("pair", "pair:0001", "keep"), ("bad", "train:1.jpg", "keep")):
                with self.assertRaises(ValueError):
                    store.save(*args)
            self.assertEqual(store.decisions(), [])

    def test_preview_changed_source_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, specs = self.prepared(temporary)
            store = review.ReviewStore(output)
            self.assertTrue(store.preview("train:1.jpg").startswith(b"\xff\xd8"))
            (root / specs[0].relative).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Source image changed"):
                store.preview("train:1.jpg")
            with self.assertRaises(ValueError):
                review.safe_image(root, "../outside.jpg")

    def test_ui_api_never_exposes_gold_labels_and_keeps_text_as_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, output, _ = self.prepared(temporary)
            store = review.ReviewStore(output)
            state = store.public_state()
            for row in state["rows"]:
                self.assertEqual(set(row), {"id", "text", "size"})
            self.assertNotIn("source_label", json.dumps(state))
            self.assertNotIn("innerHTML", review.PAGE)
            self.assertIn("text.textContent", review.PAGE)

    @unittest.skipUnless(shutil.which("node"), "Optional local JavaScript parser unavailable")
    def test_javascript_parses_after_python_template_render(self):
        script = review.PAGE.split('<script nonce=', 1)[1].split('>', 1)[1].split('</script>', 1)[0]
        script = script.replace("__TOKEN__", '"test-token"')
        result = subprocess.run(
            ["node", "-e", "new (require('vm').Script)(require('fs').readFileSync(0,'utf8'));"],
            input=script, text=True, encoding="utf-8", capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_export_is_new_snapshot_not_training_data_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output, _ = self.prepared(temporary)
            store = review.ReviewStore(output)
            target = Path(temporary) / "snapshot.json"
            store.export(target)
            snapshot = json.loads(target.read_text(encoding="utf-8"))
            self.assertFalse(snapshot["status"]["training_ready"])
            self.assertEqual(snapshot["manifest_sha256"], hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                store.export(target)
            with self.assertRaises(ValueError):
                store.export(root / "new.json")

    def test_http_preview_read_api_and_csrf_host_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, output, _ = self.prepared(temporary)
            store = review.ReviewStore(output)
            server = ThreadingHTTPServer(("127.0.0.1", 0), review.handler_for(store))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                connection.request("GET", "/api/state")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["status"]["readable_rows"], 2)
                connection.request("GET", "/image?id=train%3A1.jpg")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read().startswith(b"\xff\xd8"))
                connection.request("POST", "/api/decision", body='{}', headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 403)
                response.read()
                connection.request("GET", "/api/state", headers={"Host": "evil.example"})
                response = connection.getresponse()
                self.assertEqual(response.status, 403)
                response.read()
                connection.request("GET", "/")
                response = connection.getresponse()
                html = response.read().decode()
                token = html.split('const token="', 1)[1].split('"', 1)[0]
                body = json.dumps({"kind": "content", "target": "train:1.jpg", "decision": "keep"})
                connection.request("POST", "/api/decision", body=body,
                                   headers={"Content-Type": "application/json", "X-Review-Token": token})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                self.assertEqual(store.status()["content_reviewed"], 1)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
