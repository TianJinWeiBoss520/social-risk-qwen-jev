"""All default commands are local and free; none performs inference/training/POST."""

import argparse
import json
from importlib.resources import files
from pathlib import Path

from .decision.policy import prepare_request
from .decision.rules import review_decision
from .evaluation.leaderboard import render_table
from .perception.schema import consistency_flags, parse_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    result = sub.add_parser("results", help="Print historical aggregate results, without recomputing model predictions")
    result.add_argument("--split", choices=("test", "validation"), default="test")
    sub.add_parser("demo", help="Offline SYNTHETIC fixtures; not real model/API inference")
    validate = sub.add_parser("validate-evidence", help="Read a local seven-field JSON without any upload")
    validate.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    if args.command == "results":
        print(render_table(args.split))
    elif args.command == "validate-evidence":
        value = parse_evidence(args.input.read_text(encoding="utf-8"))
        print(json.dumps({"schema_valid": True, "screening_flags_not_proven_errors": consistency_flags(value),
                          "api_calls": 0}, ensure_ascii=False, indent=2))
    else:
        fixtures = json.loads(files("social_risk_jev").joinpath("resources", "synthetic_demo.json").read_text(encoding="utf-8"))
        for item in fixtures:
            request, counts = prepare_request(item["transcription"], item["evidence"])
            print(json.dumps({"source": "SYNTHETIC_CACHE_NOT_MODEL_INFERENCE", "api_calls": 0,
                              "redactions": counts, "cloud_payload_keys": sorted(request["state"]),
                              "decision": review_decision(item["synthetic_score"], needs_review=item["evidence"]["needs_review"])},
                             ensure_ascii=False, indent=2))
    return 0
