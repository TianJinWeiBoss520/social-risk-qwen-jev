import json
from importlib.resources import files

from .metrics import confusion_metrics


def load_results(split="test"):
    if split not in {"test", "validation"}:
        raise ValueError("Use test or validation; never mix them")
    document = json.loads(files("social_risk_jev").joinpath("resources", f"{split}_results.json").read_text(encoding="utf-8"))
    n = document["n"]
    if len({r["id"] for r in document["methods"]}) != len(document["methods"]):
        raise ValueError("Duplicate method IDs")
    for row in document["methods"]:
        row["metrics"] = confusion_metrics(row["confusion_matrix"])
        if row["metrics"]["n"] != n:
            raise ValueError("Incomparable sample count")
    return document


def render_table(split="test"):
    methods = sorted(load_results(split)["methods"], key=lambda r: (-r["metrics"]["macro_f1"], r["id"]))
    lines = ["| 排名 | 方法 | Macro-F1 | 准确率 | 有害精确率 | 有害召回率 | FP | FN |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(methods, 1):
        m = row["metrics"]
        scores = " | ".join(f"{100*m[key]:.2f}%" for key in ("macro_f1", "accuracy", "harmful_precision", "harmful_recall"))
        lines.append(f"| {rank} | {row['name']} | {scores} | {m['false_positives']} | {m['false_negatives']} |")
    return "\n".join(lines)
