"""Pre-publication check: print only finding locations/types, never secret values.

This is a conservative guard, not a complete secret/PII detection guarantee.
No network access, key lookup or files changed. Run before staging AND pushing.
"""

import argparse
import re
import subprocess
from pathlib import Path

SKIP = {".git", ".venv", "venv", "__pycache__", ".cpu_testdeps", ".pytest_cache", "build", "dist"}
PRIVATE_ROOTS = {"data", "outputs", "reports", "logs", "models", "checkpoints", "private"}
PRIVATE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".pkl", ".joblib", ".zip", ".rar", ".part", ".jpg", ".jpeg", ".png", ".gif", ".htm", ".html"}
SECRET_PATTERNS = (
    ("github_token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("literal_api_key_assignment", re.compile(rb"(?:TYPESAFE_API_KEY|GITHUB_TOKEN|GH_TOKEN|OPENAI_API_KEY)\s*=\s*['\"][A-Za-z0-9_.-]{12,}['\"]")),
    ("literal_bearer_token", re.compile(rb"Bearer [A-Za-z0-9_.-]{24,}")),
)


def candidate_files(root):
    root = root.resolve()
    if (root / ".git").exists():
        process = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
        tracked = [root / p.decode("utf-8") for p in process.stdout.split(b"\0") if p]
        if tracked:
            return tracked
    return [p for p in root.rglob("*") if p.is_file() and not set(p.relative_to(root).parts) & SKIP]


def scan(root):
    root = root.resolve()
    findings = []
    for path in candidate_files(root):
        relative = path.relative_to(root)
        if path.is_symlink():
            findings.append((str(relative), "symlink_not_allowed"))
            continue
        parts, name = relative.parts, relative.name
        if (parts[0] in PRIVATE_ROOTS or (name.startswith(".env") and name != ".env.example")
                or path.suffix.lower() in PRIVATE_SUFFIXES
                or any(term in name for term in ("PRIVATE", "LOCAL_ONLY", "DO_NOT_UPLOAD", "NOT_TRAINING"))):
            findings.append((str(relative), "private_artifact_not_allowed"))
            continue
        if path.stat().st_size > 1_000_000:
            findings.append((str(relative), "unexpected_large_public_file"))
            continue
        content = path.read_bytes()
        for reason, pattern in SECRET_PATTERNS:
            if pattern.search(content):
                findings.append((str(relative), reason))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-license", action="store_true")
    args = parser.parse_args()
    findings = scan(args.root)
    if args.require_license and not (args.root / "LICENSE").is_file():
        findings.append(("LICENSE", "owner_license_choice_required_before_publication"))
    for path, reason in findings:
        print(f"BLOCK {path}: {reason}")
    print(f"PUBLICATION_CHECK {'BLOCKED' if findings else 'PASS'} FINDINGS={len(findings)} API_CALLS=0")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
