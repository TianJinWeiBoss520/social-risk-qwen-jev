"""Recompute published metrics from confusion counts, never from private records."""


def confusion_metrics(matrix):
    if not isinstance(matrix, list) or len(matrix) != 2 or any(not isinstance(row, list) or len(row) != 2 for row in matrix):
        raise ValueError("Expected 2x2 confusion matrix")
    values = [v for row in matrix for v in row]
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("Nonnegative integer counts required")
    tn, fp, fn, tp = values
    n = sum(values)
    if not n or not tn + fp or not tp + fn:
        raise ValueError("Both reference classes must be represented")
    def divide(a, b):
        return a / b if b else 0.0
    pos_f1 = divide(2 * tp, 2 * tp + fp + fn)
    neg_f1 = divide(2 * tn, 2 * tn + fp + fn)
    return {"n": n, "accuracy": (tn + tp) / n, "macro_f1": (pos_f1 + neg_f1) / 2,
            "harmful_precision": divide(tp, tp + fp), "harmful_recall": tp / (tp + fn),
            "harmful_f1": pos_f1, "balanced_accuracy": ((tp / (tp + fn)) + (tn / (tn + fp))) / 2,
            "false_positives": fp, "false_negatives": fn}
