"""Explicit launcher for untouched archival scripts, never runs a stage by default."""

import argparse
import runpy
import sys
from pathlib import Path

STAGES = {
    "download": "download_ltedi_images.py", "audit": "audit_ltedi_images.py",
    "prepare": "prepare_ltedi_splits.py", "cpu": "train_ltedi_cpu_baselines.py",
    "qwen-base": "evaluate_ltedi_qwen_base.py", "lora": "run_ltedi_language_lora_pilot.py",
    "jev-prepare20": "prepare_ltedi_jev_pilot.py", "evidence20": "probe_ltedi_lora_evidence20.py",
    "jev-cloud20": "run_ltedi_jev_cloud20.py", "evidence172": "expand_ltedi_evidence_val172.py",
    "jev-cloud152": "run_ltedi_jev_remaining152.py", "frozen-test": "run_ltedi_frozen_test170.py",
    "jev-ablation": "run_ltedi_jev_text_vs_multimodal172.py", "review23": "prepare_ltedi_chain_review23.py",
    "dataset-review": "review_ltedi_dataset.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["list", *STAGES])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.stage == "list":
        for name, script in STAGES.items():
            print(f"{name}: {script}")
        return 0
    root = Path(__file__).resolve().parents[1]
    directory = root / "experiments/v1/src"
    script = directory / STAGES[args.stage]
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not arguments:
        raise SystemExit("EXPLICIT_ARGUMENTS_REQUIRED: use -- --help or -- --plan-only with your own paths")
    sys.path.insert(0, str(directory))
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
