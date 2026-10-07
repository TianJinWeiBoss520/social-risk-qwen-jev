#!/usr/bin/env python3
"""CPU text ablations on frozen LT-EDI train/validation data; test is NEVER read.

Dependencies: scikit-learn, joblib, and (unless skipped) xgboost-cpu.
TF-IDF vocabulary/IDF are learned from train only. Hyperparameters and threshold
are selected using validation Macro-F1, NOT reported as held-out test results.
Images remain in the dataset but are deliberately not used by these text controls.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATA = "/root/autodl-tmp/social_risk_jev/data/processed/ltedi_grouped_v1"
DEFAULT_OUTPUT = "/root/autodl-tmp/social_risk_jev/outputs/cpu_text_baselines_v1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def load_rows(path):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty split: {path.name}")
    ids = set()
    for row in rows:
        if not {"id", "image", "text", "label", "group_id"} <= set(row):
            raise ValueError("Unexpected manifest schema")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("Repeated/invalid ID")
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Expected original binary label, not a generated risk band")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Empty/invalid text")
        if not isinstance(row["group_id"], str) or not row["group_id"]:
            raise ValueError("Invalid group ID")
        image = Path(row["image"])
        if not image.is_absolute() or not image.is_file():
            raise ValueError(f"Source image missing: {row['id']}")
        ids.add(row["id"])
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Both labels must occur in this split")
    return rows


def load_train_validation(directory):
    train = load_rows(directory / "train.jsonl")
    val = load_rows(directory / "val.jsonl")
    for field in ("id", "group_id"):
        if {row[field] for row in train} & {row[field] for row in val}:
            raise ValueError(f"Train/validation overlap: {field}")
    if {normalized(row["text"]) for row in train} & {normalized(row["text"]) for row in val}:
        raise ValueError("Train/validation identical normalized text overlap")
    if any(not row["id"].startswith("train:") for row in train + val):
        raise ValueError("Train/validation must come from original train, never original dev")
    return train, val  # Deliberately no opening, hashing, or label access of test.jsonl.


def metrics(truth, scores, threshold=.5):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        brier_score_loss, confusion_matrix, f1_score, log_loss,
        precision_score, recall_score, roc_auc_score,
    )
    y, p = np.asarray(truth, dtype=int), np.asarray(scores, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid positive-class scores")
    predictions = (p >= threshold).astype(int)
    return {
        "n": len(y), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", labels=[0, 1], zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "harmful_precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "harmful_recall": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "harmful_f1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)), "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack((1 - p, p)), labels=[0, 1])),
        "confusion_matrix_true_rows_predicted_columns_0_1": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def select_threshold(truth, scores):
    # Small, predeclared validation-only search; equal scores prefer threshold .5.
    candidates = [metrics(truth, scores, threshold) for threshold in (.30, .35, .40, .45, .50, .55, .60, .65, .70)]
    return max(candidates, key=lambda item: (item["macro_f1"], -abs(item["threshold"] - .5), item["harmful_recall"]))


def positive_scores(classifier, features):
    import numpy as np
    classes = list(classifier.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Unexpected classifier classes")
    return np.asarray(classifier.predict_proba(features)[:, classes.index(1)], dtype=float)


def benchmark(vectorizer, classifier, texts):
    import numpy as np
    # Same-process CPU text latency: includes tokenization/TF-IDF + prediction,
    # excludes file/network/JSON parsing. Not equivalent to GPU service latency.
    positive_scores(classifier, vectorizer.transform(texts[:1]))
    durations = []
    for text in texts[:64]:
        begin = time.perf_counter()
        positive_scores(classifier, vectorizer.transform([text]))
        durations.append((time.perf_counter() - begin) * 1000)
    return {"unit": "milliseconds", "samples": len(durations), "batch_size": 1,
            "p50": float(np.percentile(durations, 50)), "p95": float(np.percentile(durations, 95)),
            "mean": float(np.mean(durations)), "includes": "TF-IDF transform + classifier.predict_proba",
            "excludes": "disk/network I/O and request/response JSON serialization"}


def make_candidates(y_train, seed, skip_xgboost):
    from sklearn.linear_model import LogisticRegression
    candidates = []
    for c in (1.0, 4.0):
        candidates.append((f"tfidf_logreg_c{int(c)}",
                           LogisticRegression(C=c, solver="liblinear", class_weight="balanced",
                                              max_iter=2000, random_state=seed),
                           {"algorithm": "LogisticRegression", "C": c, "solver": "liblinear",
                            "class_weight": "balanced", "max_iter": 2000}))
    if not skip_xgboost:
        from xgboost import XGBClassifier
        counts = Counter(int(label) for label in y_train)
        for depth in (2, 4):
            params = {"n_estimators": 200, "max_depth": depth, "learning_rate": .05,
                      "tree_method": "hist", "device": "cpu", "n_jobs": 4,
                      "min_child_weight": 3, "subsample": 1.0, "colsample_bytree": .8,
                      "reg_lambda": 5.0, "scale_pos_weight": counts[0] / counts[1],
                      "objective": "binary:logistic", "eval_metric": "logloss", "random_state": seed}
            candidates.append((f"tfidf_xgboost_d{depth}", XGBClassifier(**params), params))
    return candidates


def train_baselines(data_dir, output_dir, seed=2060, skip_xgboost=False, plan_only=False):
    data_dir, output_dir = data_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"OUTPUT_EXISTS_STOP: {output_dir}; do not overwrite or tune on test")
    if output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):
        raise ValueError("Output must be separate from the prepared dataset")
    train, val = load_train_validation(data_dir)
    print(f"TRAIN={len(train)} VAL={len(val)} INPUT_MODALITY=text_only_ablation TEST_READ=False", flush=True)
    if plan_only:
        print("PLAN_ONLY; no dependencies imported, models fitted or files written", flush=True)
        return None

    import joblib
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from threadpoolctl import threadpool_limits
    # Fail on missing XGBoost BEFORE any model training or output creation.
    y_train = np.array([row["label"] for row in train])
    y_val = np.array([row["label"] for row in val])
    candidates = make_candidates(y_train, seed, skip_xgboost)
    train_texts, val_texts = [row["text"] for row in train], [row["text"] for row in val]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2,
                                 max_features=30000, sublinear_tf=True, dtype=np.float32)
    overall_start = time.perf_counter()
    with threadpool_limits(limits=4):
        features_train = vectorizer.fit_transform(train_texts)
        features_val = vectorizer.transform(val_texts)
        print(f"TFIDF_FIT_ON_TRAIN_ONLY VOCABULARY={len(vectorizer.vocabulary_)}", flush=True)
        training_prevalence = float(np.mean(y_train))
        majority = metrics(y_val, np.full(len(val), training_prevalence))
        majority["score_definition"] = "Constant training positive-class prevalence; threshold 0.5 gives majority class"
        majority["training_positive_prevalence"] = training_prevalence
        print("MAJORITY_VAL=" + json.dumps(majority, ensure_ascii=False), flush=True)
        results, fitted = {}, {}
        for name, classifier, params in candidates:
            start = time.perf_counter()
            classifier.fit(features_train, y_train)
            fit_seconds = time.perf_counter() - start
            scores = positive_scores(classifier, features_val)
            fixed = metrics(y_val, scores, .5)
            tuned = select_threshold(y_val, scores)
            latency = benchmark(vectorizer, classifier, val_texts)
            results[name] = {"hyperparameters": params, "fit_seconds": fit_seconds,
                             "validation_fixed_threshold_0_5": fixed,
                             "validation_selected_threshold": tuned, "cpu_text_latency": latency}
            fitted[name] = (classifier, scores)
            print(f"VAL {name} MACRO_F1={tuned['macro_f1']:.4f} HARM_RECALL={tuned['harmful_recall']:.4f} "
                  f"HARM_PRECISION={tuned['harmful_precision']:.4f} PR_AUC={tuned['pr_auc_average_precision']:.4f} "
                  f"THRESHOLD={tuned['threshold']:.2f} FIT_SEC={fit_seconds:.2f}", flush=True)

    selected = max(results, key=lambda name: (
        results[name]["validation_selected_threshold"]["macro_f1"],
        results[name]["validation_selected_threshold"]["pr_auc_average_precision"],
    ))
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    for distribution in ("numpy", "scipy", "joblib", "xgboost-cpu", "xgboost"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "validation_model_selection_not_final_test", "input_modality": "text_only_ablation",
        "seed": seed, "versions": versions, "source_split_sha256": {
            "train": digest(data_dir / "train.jsonl"), "val": digest(data_dir / "val.jsonl")},
        "train_n": len(train), "validation_n": len(val), "test_read": False,
        "train_class_counts": dict(Counter(int(label) for label in y_train)),
        "validation_class_counts": dict(Counter(int(label) for label in y_val)),
        "label_meaning": {"0": "Not-Misogyny", "1": "Misogyny"},
        "feature_settings": {"analyzer": "char", "ngram_range": [1, 3], "min_df": 2,
                             "max_features": 30000, "sublinear_tf": True, "fit_split": "train_only"},
        "vocabulary_size": len(vectorizer.vocabulary_), "majority_validation": majority,
        "models": results, "selected_text_baseline": selected,
        "selection_rule": "validation Macro-F1, then validation average precision; no test access",
        "elapsed_seconds": time.perf_counter() - overall_start,
        "cautions": [
            "Validation is used for selection; its best scores are not unbiased test estimates.",
            "These baselines ignore images. Multimodal classical controls are a separate subsequent experiment.",
            "Class-weighted classifier probabilities are not calibrated operational risk probabilities.",
            "No three-level severity labels exist in the original binary dataset.",
            "Never load joblib/pickle artifacts from untrusted sources.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, (classifier, scores) in fitted.items():
        threshold = results[name]["validation_selected_threshold"]["threshold"]
        bundle = {"schema_version": 1, "vectorizer": vectorizer, "classifier": classifier,
                  "threshold": threshold, "input_modality": "text_only_ablation",
                  "source_split_sha256": report["source_split_sha256"], "seed": seed}
        artifact = output_dir / f"{name}.joblib"
        with artifact.open("xb") as stream:
            joblib.dump(bundle, stream, compress=3)
        results[name]["artifact_bytes"] = artifact.stat().st_size
        with (output_dir / f"{name}_validation_predictions.jsonl").open("x", encoding="utf-8") as stream:
            for row, score in zip(val, scores):
                stream.write(json.dumps({"id": row["id"], "score": float(score),
                                         "predicted_label": int(score >= threshold)}) + "\n")
    with (output_dir / "validation_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"SELECTED_TEXT_BASELINE={selected}\nCPU_BASELINES_COMPLETE TEST_UNTOUCHED=True", flush=True)
    print(f"REPORT={output_dir / 'validation_report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument("--skip-xgboost", action="store_true", help="Explicitly run logistic regression only")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    train_baselines(args.data_dir, args.output_dir, args.seed, args.skip_xgboost, args.plan_only)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        print(f"DEPENDENCY_MISSING={exc.name}; install CPU dependencies in a separate environment", flush=True)
        raise SystemExit(2)
    except Exception as exc:
        print(f"CPU_BASELINE_STOPPED {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(2)
