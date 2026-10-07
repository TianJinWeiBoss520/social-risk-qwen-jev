"""Strict version-one seven-field evidence schema; no silent repair/default labels."""

import json

FIELDS = frozenset({"label", "gender_targeted", "attack_present", "stance", "image_role", "needs_review", "evidence"})
FEATURES = tuple(sorted(FIELDS - {"label"}))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Non-finite JSON constant")


def validate_evidence(value, *, modality="image-text"):
    if modality not in {"image-text", "text-only"}:
        raise ValueError("Unsupported modality")
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Exactly seven evidence fields are required")
    if type(value["label"]) is not int or value["label"] not in (0, 1):
        raise ValueError("Binary integer label required")
    for field in ("gender_targeted", "attack_present", "needs_review"):
        if type(value[field]) is not bool:
            raise ValueError("Boolean evidence flag required")
    if value["stance"] not in {"support", "oppose", "report", "unclear"}:
        raise ValueError("Invalid stance")
    roles = {"supports", "changes_context", "neutral"} if modality == "image-text" else {"unavailable"}
    if value["image_role"] not in roles:
        raise ValueError("Image role contradicts available modality")
    if not isinstance(value["evidence"], str) or not value["evidence"].strip() or len(value["evidence"]) > 120:
        raise ValueError("Nonempty evidence of at most 120 characters required")
    return dict(value)


def parse_evidence(raw, *, modality="image-text"):
    return validate_evidence(json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant), modality=modality)


def consistency_flags(value):
    """Screening flags, not factual-error findings and never automatic relabeling."""
    value = validate_evidence(value)
    flags = []
    if value["label"] == 1 and not value["gender_targeted"]:
        flags.append("positive_label_without_explicit_gender_target")
    if value["label"] == 1 and not value["attack_present"]:
        flags.append("positive_label_without_attack_evidence")
    if value["label"] == 1 and value["stance"] in {"oppose", "report"}:
        flags.append("positive_label_with_counter_speech_or_report_stance")
    return flags
