# Social Risk · Qwen3-VL + Jev

[Private dataset and badcase export](docs/DATA_EXPORT.md): public `data/train`, `data/val`, `data/test` and `badcase` contain documentation only. A standard-library offline utility exports existing images/transcripts, a two-column `image_name,label` CSV and the saved LoRA-evidence-plus-Jev mistakes into a separate **private** package. No new inference/API calls; no dataset redistribution. Validation/test mistakes are diagnostic, not training data.

A research project on Chinese multimodal misogyny detection: data quality checks, text baselines, Qwen3-VL-32B language-attention LoRA, structured evidence, and a hosted TypeSafe Jev policy decision layer.

**[Full Chinese README and ranked results](README.md)** · [Evaluation](docs/EVALUATION.md) · [Reproduction](docs/REPRODUCIBILITY.md)

## What the experiment actually found

On a frozen 170-example project test split, TF-IDF + Logistic Regression C=4 achieved the highest Macro-F1 (90.36%) and accuracy (92.94%). The text-OR-LoRA-Qwen combination achieved the highest positive-class recall (91.49%), at the cost of 12 false positives versus zero for the selected text baseline. LoRA improved the matched Qwen test Macro-F1 from 76.65% to 82.36%.

The predeclared primary text-OR-multimodal-Jev system achieved 87.25% test Macro-F1. It did **not** outperform the selected text baseline. Adding Qwen multimodal evidence to text-only Jev improved validation Macro-F1 from 69.42% to 84.25% at t=0.40. A subsequently completed **supplementary test comparator** on the same 170 test examples showed 63.04% → 79.74%. The latter was added after the original test results were revealed, with unchanged model, policy and thresholds; it is not a pre-registered original method or an image-only causal comparison.

## Pure Jev — supplementary 170-example test

Both thresholds use the same 170 pure-text scores, not separate sets of API calls. The Chinese main leaderboard now contains all 11 original methods plus these two supplementary controls.

| Method | Macro-F1 | Accuracy | Positive precision | Positive recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| LoRA Qwen multimodal evidence + Jev, t=0.40 | 79.74% | 84.12% | 72.73% | 68.09% | 12 | 15 |
| LoRA Qwen multimodal evidence + Jev, t=0.50 | 76.65% | 82.35% | 71.79% | 59.57% | 11 | 19 |
| Pure text-only Jev, t=0.50, supplementary control | 63.57% | 78.82% | 92.31% | 25.53% | 1 | 35 |
| Pure text-only Jev, t=0.40, supplementary comparator | 63.04% | 78.24% | 85.71% | 25.53% | 2 | 35 |

At matched t=0.40, Qwen evidence increases Macro-F1 by **16.70 percentage points**, with 20 fewer false negatives and 10 additional false positives. Pure Jev receives no images or Qwen evidence. This compares removal of the whole evidence bundle, including Qwen semantic processing, not the image alone. The unchanged policy still mentions absent `perception_hypotheses` in the text arm: this is not a separately optimized pure-Jev baseline. API calls occurred at different times. [Protocol, provenance and continuation audit](docs/PURE_JEV_TEST_SUPPLEMENT.md).

## Pure Jev vs multimodal evidence + Jev — validation only

The following comparison uses the same **172 validation examples (123 negative / 49 positive)**, Jev `jev-1.13.0`, and fixed decision policy. It is separate from the 170-example final test leaderboard. Rows are ordered by Macro-F1; compare methods at the same threshold.

| Method | Macro-F1 | Accuracy | Positive precision | Positive recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| LoRA Qwen multimodal evidence + Jev, t=0.40 | 84.25% | 88.37% | 91.43% | 65.31% | 3 | 17 |
| LoRA Qwen multimodal evidence + Jev, t=0.50 | 79.46% | 85.47% | 90.00% | 55.10% | 3 | 22 |
| Pure text-only Jev, t=0.40 | 69.42% | 80.81% | 94.44% | 34.69% | 1 | 32 |
| Pure text-only Jev, t=0.50 | 68.70% | 80.81% | 100.00% | 32.65% | 0 | 33 |

Pure Jev receives the transcript only. The multimodal system adds six Qwen evidence fields derived from the image and transcript, without the Qwen final label. At t=0.40, validation Macro-F1 increases by **14.83 percentage points**, with 15 fewer false negatives and two additional false positives. This adds Qwen semantic processing as well as visual information, so it is not an image-only causal gain. **The completed supplementary 170-example test is reported separately above; do not mix these validation scores into the test leaderboard.** This documentation update reuses existing results and makes no training, inference, or Jev API calls. [Full validation history](docs/VALIDATION_RESULTS.md).

## Safe offline entry points

```bash
python -m pip install -e .
social-risk results --split test
social-risk results --split validation
social-risk demo
```

The CLI performs no GPU inference or API requests. The demo uses synthetic content and preset scores, **not live model predictions**. Installation may access the package index.

The repository contains an installable lightweight interface package, untouched historical experimental scripts, synthetic/mock regression tests, and aggregate confusion matrices. Raw dataset contents, row-level predictions, private review records, keys and trained weights are intentionally excluded. Full experiment replay requires separately obtained data/weights, compatible GPU dependencies and explicit cloud consent.

Risk bands are operational labels, not calibrated probabilities. Outputs support human review, never automatic deletion. The test split is the original source dev set reserved for this project, not a hidden official benchmark test set. No DPO, evidence-supervised SFT or visual LoRA was completed in this project.

Original code and documentation are released under the [MIT License](LICENSE). Third-party dataset, model and service terms are separate. Author: **TianJinWeiBoss520**.
