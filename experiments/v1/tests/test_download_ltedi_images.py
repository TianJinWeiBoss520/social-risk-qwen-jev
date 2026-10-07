import hashlib
import http.client
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import download_ltedi_images as dl


class DownloadTests(unittest.TestCase):
    def test_git_blob_hash_and_streaming_file_agree(self):
        data = b"example\x00\xff\x01"
        expected = hashlib.sha1(b"blob 10\x00" + data).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.jpg"
            path.write_bytes(data)
            self.assertEqual(dl.blob_sha1(data), expected)
            self.assertEqual(dl.file_blob_sha1(path), expected)

    def test_valid_file_is_preserved(self):
        data = b"verified-source-data"
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            spec = dl.ImageSpec("train", "Datasets/train/train images/1.jpg", dl.blob_sha1(data))
            path = dl.image_path(repo, spec)
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
            before = path.stat().st_mtime_ns
            self.assertFalse(dl.publish_payload(repo, spec, data, repo / "backup"))
            self.assertEqual(path.stat().st_mtime_ns, before)

    def test_invalid_file_is_backed_up_before_repair(self):
        old, data = b"incomplete", b"correct-source-data"
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            backup = repo / "backups"
            spec = dl.ImageSpec("train", "Datasets/train/train images/1.jpg", dl.blob_sha1(data))
            path = dl.image_path(repo, spec)
            path.parent.mkdir(parents=True)
            path.write_bytes(old)
            self.assertTrue(dl.publish_payload(repo, spec, data, backup))
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual((backup / spec.relative).read_bytes(), old)

    def test_wrong_download_never_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            spec = dl.ImageSpec("train", "Datasets/train/train images/1.jpg", dl.blob_sha1(b"correct"))
            path = dl.image_path(repo, spec)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"existing")
            with self.assertRaises(RuntimeError):
                dl.publish_payload(repo, spec, b"wrong", repo / "backup")
            self.assertEqual(path.read_bytes(), b"existing")

    def test_escaped_image_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            spec = dl.ImageSpec("train", "../outside.jpg", "0" * 40)
            with self.assertRaises(RuntimeError):
                dl.image_path(repo, spec)

    def test_direct_route_after_503_and_url_encoding(self):
        data = b"correct-source-data"
        spec = dl.ImageSpec("train", "Datasets/train/train images/1005.jpg", dl.blob_sha1(data))
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = data
        proxy, direct = mock.Mock(), mock.Mock()
        proxy.open.side_effect = urllib.error.HTTPError("https://example.test", 503, "Unavailable", {}, io.BytesIO())
        direct.open.return_value = response
        with mock.patch.object(dl.urllib.request, "build_opener", side_effect=[proxy, direct]), mock.patch.object(dl.time, "sleep"):
            payload, route = dl.fetch_payload(spec, 20, 2)
        self.assertEqual(payload, data)
        self.assertEqual(route, "direct")
        request = direct.open.call_args.args[0]
        self.assertIn("train%20images/1005.jpg", request.full_url)
        self.assertIn(dl.COMMIT, request.full_url)

    def test_incomplete_http_body_is_retried(self):
        data = b"correct-source-data"
        spec = dl.ImageSpec("train", "Datasets/train/train images/1.jpg", dl.blob_sha1(data))
        broken = mock.MagicMock()
        broken.__enter__.return_value = broken
        broken.status = 200
        broken.read.side_effect = http.client.IncompleteRead(b"partial", 10)
        complete = mock.MagicMock()
        complete.__enter__.return_value = complete
        complete.status = 200
        complete.read.return_value = data
        proxy, direct = mock.Mock(), mock.Mock()
        proxy.open.return_value = broken
        direct.open.return_value = complete
        with mock.patch.object(dl.urllib.request, "build_opener", side_effect=[proxy, direct]), mock.patch.object(dl.time, "sleep"):
            payload, route = dl.fetch_payload(spec, 20, 2)
        self.assertEqual((payload, route), (data, "direct"))

    def test_read_manifest_validates_pinned_annotations_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            (repo / ".git").mkdir()
            tree = []
            for split in ("train", "dev"):
                path = repo / f"Datasets/{split}/{split}.csv"
                path.parent.mkdir(parents=True)
                content = b"image_name,labels,transcriptions\n1.jpg,Misogyny,example text\n"
                path.write_bytes(content)
                relative = f"Datasets/{split}/{split}.csv"
                tree.append(f"100644 blob {dl.blob_sha1(content)}\t{relative}\0".encode())
                tree.append(f"100644 blob {dl.blob_sha1(b'fake-image')}\tDatasets/{split}/{split} images/1.jpg\0".encode())
            with mock.patch.object(dl, "EXPECTED", {"train": 1, "dev": 1}), mock.patch.object(dl.subprocess, "check_output", side_effect=[dl.COMMIT + "\n", b"".join(tree)]):
                specs = dl.read_specs(repo)
            self.assertEqual(len(specs), 2)
            self.assertEqual({spec.split for spec in specs}, {"train", "dev"})

    def test_targeted_stop_does_not_signal_unrelated_process(self):
        alive = {701: ["git", "sparse-checkout"], 702: ["git", "fetch"], 703: ["git-remote-https"]}
        killed = []

        def kill(pid, sig):
            killed.append(pid)
            alive.pop(pid, None)

        with mock.patch.object(dl, "find_old_git_roots", side_effect=[{701}, set()]), mock.patch.object(dl.subprocess, "check_output", return_value="701 1\n702 701\n703 702\n999 1\n"), mock.patch.object(dl, "process_args", side_effect=lambda pid: alive.get(pid, [])), mock.patch.object(dl.os, "kill", side_effect=kill):
            dl.stop_old_git(Path("/example/repo"))
        self.assertEqual(killed, [703, 702, 701])
        self.assertNotIn(999, killed)


if __name__ == "__main__":
    unittest.main()
