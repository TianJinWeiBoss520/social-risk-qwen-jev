"""Basic pattern masking only; not a guarantee of anonymization."""

import re

PATTERNS = (
    ("url", re.compile(r"https?://[^\s]+|www\.[^\s]+", re.I)),
    ("email", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("cn_id", re.compile(r"(?<![A-Za-z0-9])\d{17}[0-9Xx](?![A-Za-z0-9])")),
    ("mobile", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")),
    ("handle", re.compile(r"@[\w.\-]+")),
)


def mask_text(text):
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    counts = {}
    for name, pattern in PATTERNS:
        text, count = pattern.subn(f"[{name.upper()}_REDACTED]", text)
        if count:
            counts[name] = count
    return text, counts
