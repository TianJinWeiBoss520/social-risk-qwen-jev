# Social Risk · Qwen3-VL + Jev

A research project on Chinese multimodal misogyny detection: data quality checks, text baselines, Qwen3-VL-32B language-attention LoRA, structured evidence, and a hosted TypeSafe Jev policy decision layer.

**[Full Chinese README and ranked results](README.md)** · [Evaluation](docs/EVALUATION.md) · [Reproduction](docs/REPRODUCIBILITY.md)

## What the experiment actually found

On a frozen 170-example project test split, TF-IDF + Logistic Regression C=4 achieved the highest Macro-F1 (90.36%) and accuracy (92.94%). The text-OR-LoRA-Qwen combination achieved the highest positive-class recall (91.49%), at the cost of 12 false positives versus zero for the selected text baseline. LoRA improved the matched Qwen test Macro-F1 from 76.65% to 82.36%.

The predeclared primary text-OR-multimodal-Jev system achieved 87.25% test Macro-F1. It did **not** outperform the selected text baseline. A separate validation ablation showed a benefit from adding Qwen multimodal evidence to text-only Jev (69.42% → 84.25% Macro-F1 at t=0.40); it is not an image-only causal comparison or an independent final test result.

## Pure Jev vs multimodal evidence + Jev — validation only

The following comparison uses the same **172 validation examples (123 negative / 49 positive)**, Jev `jev-1.13.0`, and fixed decision policy. It is separate from the 170-example final test leaderboard. Rows are ordered by Macro-F1; compare methods at the same threshold.

| Method | Macro-F1 | Accuracy | Positive precision | Positive recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| LoRA Qwen multimodal evidence + Jev, t=0.40 | 84.25% | 88.37% | 91.43% | 65.31% | 3 | 17 |
| LoRA Qwen multimodal evidence + Jev, t=0.50 | 79.46% | 85.47% | 90.00% | 55.10% | 3 | 22 |
| Pure text-only Jev, t=0.40 | 69.42% | 80.81% | 94.44% | 34.69% | 1 | 32 |
| Pure text-only Jev, t=0.50 | 68.70% | 80.81% | 100.00% | 32.65% | 0 | 33 |

Pure Jev receives the transcript only. The multimodal system adds six Qwen evidence fields derived from the image and transcript, without the Qwen final label. At t=0.40, validation Macro-F1 increases by **14.83 percentage points**, with 15 fewer false negatives and two additional false positives. This adds Qwen semantic processing as well as visual information, so it is not an image-only causal gain. **Pure Jev was not evaluated on the final 170-example test split.** This documentation update reuses existing results and makes no training, inference, or Jev API calls. [Full validation history](docs/VALIDATION_RESULTS.md).

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
