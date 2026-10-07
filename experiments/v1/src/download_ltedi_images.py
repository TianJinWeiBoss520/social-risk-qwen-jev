#!/usr/bin/env python3
"""Resume the pinned LT-EDI train/dev image download; no labels are changed.

Run on AutoDL/Linux. Uses Python's standard library only. Existing files are
accepted only if their Git blob SHA-1 matches the pinned repository tree.
This verifies source bytes, NOT image readability or political-content safety.
"""

import argparse
import csv
import hashlib
import http.client
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


COMMIT = "0179687f197a0b2babddcb4eb32547a07cba0933"
DEFAULT_REPO = "/root/autodl-tmp/social_risk_jev/data/raw/ltedi_cn_misogyny"
RAW_BASE = f"https://raw.githubusercontent.com/AJFaisal002/Mysogyny-Meme-Detection/{COMMIT}/"
EXPECTED = {"train": 1190, "dev": 170}
MAX_IMAGE_BYTES = 64 * 1024 * 1024
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


@dataclass(frozen=True)
class ImageSpec:
    split: str
    relative: str
    sha1: str


def blob_sha1(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def file_blob_sha1(path):
    digest = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(repo, spec):
    path = repo / spec.relative
    if path.is_symlink() or not path.resolve().is_relative_to(repo):
        raise RuntimeError(f"Unsafe image path: {spec.relative}")
    if path.exists() and not path.is_file():
        raise RuntimeError(f"Not a regular file: {spec.relative}")
    return path


def matches_file(path, sha1):
    return path.is_file() and file_blob_sha1(path) == sha1


def read_specs(repo):
    if not (repo / ".git").is_dir():
        raise RuntimeError(f"Repository missing: {repo}")
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=15, text=True
    ).strip()
    if head != COMMIT:
        raise RuntimeError(f"Unexpected commit: {head}; expected {COMMIT}")
    tree = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", COMMIT, "--",
         "Datasets/train", "Datasets/dev"], timeout=30
    )
    blobs = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        meta, relative = entry.split(b"\t", 1)
        mode, kind, sha1 = meta.decode().split()
        if kind == "blob" and mode == "100644":
            blobs[relative.decode("utf-8")] = sha1
    specs = []
    for split, expected in EXPECTED.items():
        relative_csv = f"Datasets/{split}/{split}.csv"
        csv_path = repo / relative_csv
        if csv_path.is_symlink() or not csv_path.resolve().is_relative_to(repo):
            raise RuntimeError(f"Unsafe annotation path: {csv_path}")
        if not matches_file(csv_path, blobs.get(relative_csv, "")):
            raise RuntimeError(f"Original annotation checksum mismatch: {relative_csv}")
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"image_name", "labels", "transcriptions"}.issubset(reader.fieldnames or []):
                raise RuntimeError(f"Unexpected annotation fields: {relative_csv}")
            rows = list(reader)
        if len(rows) != expected:
            raise RuntimeError(f"Unexpected row count: {split}={len(rows)}")
        names = set()
        for row in rows:
            name = row["image_name"].strip()
            if not name or Path(name).name != name or "\\" in name or name in names:
                raise RuntimeError(f"Unsafe or duplicate image name in {split}: {name}")
            if row["labels"].strip() not in {"Misogyny", "Not-Misogyny"}:
                raise RuntimeError(f"Unexpected original label in {split}: {name}")
            if not row["transcriptions"].strip():
                raise RuntimeError(f"Empty transcription in {split}: {name}")
            names.add(name)
            relative = f"Datasets/{split}/{split} images/{name}"
            if relative not in blobs:
                raise RuntimeError(f"Image not in pinned tree: {relative}")
            spec = ImageSpec(split, relative, blobs[relative])
            image_path(repo, spec)
            specs.append(spec)
    return specs


def process_args(pid):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []


def find_old_git_roots(repo):
    roots = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        args = process_args(int(proc.name))
        if not args or Path(args[0]).name != "git" or "-C" not in args:
            continue
        at = args.index("-C")
        if at + 1 >= len(args) or Path(args[at + 1]).resolve() != repo:
            continue
        if any(args[i:i+2] == ["sparse-checkout", "add"] for i in range(len(args) - 1)):
            roots.add(int(proc.name))
    return roots


def stop_old_git(repo):
    roots = find_old_git_roots(repo)
    if not roots:
        log("OLD_GIT_DOWNLOAD_ALREADY_EXITED")
        return
    rows = subprocess.check_output(["ps", "-e", "-o", "pid=,ppid="], text=True, timeout=10)
    pairs = [tuple(map(int, line.split())) for line in rows.splitlines()]
    targets = set(roots)
    while True:
        expanded = targets | {pid for pid, parent in pairs if parent in targets}
        if expanded == targets:
            break
        targets = expanded
    identities = {pid: process_args(pid) for pid in targets}
    log(f"STOPPING_THIS_REPO_OLD_GIT ROOTS={sorted(roots)} DESCENDANTS={len(targets)-len(roots)}")
    for pid in sorted(targets - roots, reverse=True) + sorted(roots):
        if identities[pid] and process_args(pid) == identities[pid]:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        alive = [pid for pid, args in identities.items() if args and process_args(pid) == args]
        if not alive:
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(f"Old download still alive; no new download started: {alive}")
    if find_old_git_roots(repo):
        raise RuntimeError("Another old sparse-checkout appeared; stop and inspect")
    log("OLD_GIT_DOWNLOAD_STOPPED; existing files and Git objects retained")


def fetch_payload(spec, timeout, attempts):
    url = RAW_BASE + urllib.parse.quote(spec.relative, safe="/")
    routes = [("environment", urllib.request.build_opener()),
              ("direct", urllib.request.build_opener(urllib.request.ProxyHandler({})))]
    last_error = None
    for attempt in range(attempts):
        route, opener = routes[attempt % len(routes)]
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ltedi-research-download/1.0"})
            with opener.open(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"Unexpected HTTP status: {response.status}")
                payload = response.read(MAX_IMAGE_BYTES + 1)
            if len(payload) > MAX_IMAGE_BYTES:
                raise RuntimeError("Image exceeds the 64 MiB download limit")
            if blob_sha1(payload) != spec.sha1:
                raise RuntimeError("Payload does not match pinned Git blob SHA-1")
            return payload, route
        except (OSError, RuntimeError, urllib.error.URLError, http.client.HTTPException) as exc:
            last_error = exc
            # Do not log proxy URLs, credentials, response bodies, or labels.
            code = getattr(exc, "code", None)
            detail = f"HTTP={code}" if code is not None else type(exc).__name__
            log(f"RETRY {spec.relative} ATTEMPT={attempt+1}/{attempts} ROUTE={route} ERROR={detail}")
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Network/content validation failed after {attempts} attempts: {type(last_error).__name__}")


def publish_payload(repo, spec, payload, backup_root):
    if blob_sha1(payload) != spec.sha1:
        raise RuntimeError("Refusing to publish data with wrong source checksum")
    path = image_path(repo, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    if matches_file(path, spec.sha1):
        return False
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.download-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if path.exists():
            image_path(repo, spec)
            if matches_file(path, spec.sha1):
                return False
            backup = backup_root / spec.relative
            if not backup.resolve().is_relative_to(backup_root) or backup.exists():
                raise RuntimeError("Unsafe or existing backup destination")
            backup.parent.mkdir(parents=True, exist_ok=True)
            path.rename(backup)
            log(f"INVALID_EXISTING_FILE_BACKED_UP={backup}")
        # Hard link publishes the completed file without overwriting a competing file.
        os.link(temporary, path)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def run_download(repo, specs, workers, timeout, attempts):
    pending = []
    invalid = 0
    for spec in specs:
        path = image_path(repo, spec)
        if not matches_file(path, spec.sha1):
            pending.append(spec)
            invalid += path.exists()
    log(f"PLAN TOTAL={len(specs)} VERIFIED_EXISTING={len(specs)-len(pending)} PENDING={len(pending)} INVALID_EXISTING={invalid} WORKERS={workers}")
    if not pending:
        log("COMPLETE VERIFIED=1360/1360 TRAIN=1190/1190 DEV=170/170")
        return 0
    backup_root = (repo.parent / "ltedi_invalid_backups" / uuid.uuid4().hex).resolve()
    if not backup_root.is_relative_to(repo.parent):
        raise RuntimeError("Backup directory resolves outside the dataset parent; stop and inspect")
    start = time.monotonic()
    log("PREFLIGHT_RAW_IMAGE_DOWNLOAD")
    first = pending[0]
    payload, route = fetch_payload(first, timeout, attempts)
    publish_payload(repo, first, payload, backup_root)
    log(f"[1/{len(pending)}] {first.relative} OK ROUTE={route}")
    done = 1
    failures = []
    abort = threading.Event()

    def one(spec):
        if abort.is_set():
            return spec, "cancelled", ""
        try:
            payload, route = fetch_payload(spec, timeout, attempts)
            publish_payload(repo, spec, payload, backup_root)
            time.sleep(0.1)
            return spec, "ok", route
        except Exception as exc:
            return spec, "failed", type(exc).__name__

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, spec) for spec in pending[1:]]
        for future in as_completed(futures):
            spec, status, detail = future.result()
            if status == "cancelled":
                continue
            done += 1
            if status == "failed":
                failures.append(spec.relative)
                if len(failures) >= 5:
                    abort.set()
                log(f"[{done}/{len(pending)}] FAILED {spec.relative} ERROR={detail}")
            else:
                rate = (done - len(failures)) * 60 / max(time.monotonic() - start, 0.01)
                log(f"[{done}/{len(pending)}] {spec.relative} OK ROUTE={detail} RATE_PER_MIN={rate:.1f}")
    verified = {split: 0 for split in EXPECTED}
    remaining = []
    for spec in specs:
        if matches_file(image_path(repo, spec), spec.sha1):
            verified[spec.split] += 1
        else:
            remaining.append(spec.relative)
    log(f"FINAL_VERIFIED TRAIN={verified['train']}/1190 DEV={verified['dev']}/170")
    log(f"REMAINING_COUNT={len(remaining)} FIRST_10=" + json.dumps(remaining[:10], ensure_ascii=False))
    if remaining:
        log("INCOMPLETE; valid files retained; do not start training")
        return 2
    log("COMPLETE VERIFIED=1360/1360 TRAIN=1190/1190 DEV=170/170")
    log("NOTE: source-byte checks passed; image readability, deduplication and political screening are still required")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(DEFAULT_REPO))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--attempts", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--stop-old-git", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0 or args.timeout > 120:
        parser.error("--timeout must be in (0, 120]")
    repo = args.repo.resolve()
    specs = read_specs(repo)
    if args.plan_only:
        verified = sum(matches_file(image_path(repo, spec), spec.sha1) for spec in specs)
        log(f"PLAN_ONLY COMMIT={COMMIT} TOTAL={len(specs)} VERIFIED_EXISTING={verified} PENDING={len(specs)-verified}; no processes stopped or files written")
        return 0
    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("Run the download on the AutoDL Linux server, not Windows")
    import fcntl
    lock_path = repo.parent / "ltedi_raw_download.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another copy of this downloader is running")
        if args.stop_old_git:
            stop_old_git(repo)
        elif find_old_git_roots(repo):
            raise RuntimeError("Old Git download still running; use --stop-old-git or wait")
        return run_download(repo, specs, args.workers, args.timeout, args.attempts)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"STOPPED {type(exc).__name__}: {exc}")
        raise SystemExit(2)
