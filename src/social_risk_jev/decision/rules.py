"""Scores support human review; these bands are not calibrated severity labels."""

import math


def probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Finite probability in [0, 1] required")
    return float(value)


def risk_band(value):
    value = probability(value)
    return "LOW" if value < .2 else "MEDIUM" if value < .8 else "HIGH"


def review_decision(value, *, threshold=.4, text_label=None, needs_review=False):
    value, threshold = probability(value), probability(threshold)
    if text_label is not None and (type(text_label) is not int or text_label not in (0, 1)):
        raise ValueError("Text label must be a binary integer")
    if type(needs_review) is not bool:
        raise ValueError("needs_review must be Boolean")
    flagged = value >= threshold or text_label == 1
    return {"flagged": flagged, "risk_band": risk_band(value), "score": value,
            "threshold": threshold, "needs_human_review": needs_review or flagged,
            "action": "queue_review" if needs_review or flagged else "no_flag_under_this_policy",
            "calibrated": False, "automatic_content_deletion": False}
