# Social Risk · Qwen3-VL + Jev

A research project on Chinese multimodal misogyny detection: data quality checks, text baselines, Qwen3-VL-32B language-attention LoRA, structured evidence, and a hosted TypeSafe Jev policy decision layer.

**[Full Chinese README and ranked results](README.md)** · [Evaluation](docs/EVALUATION.md) · [Reproduction](docs/REPRODUCIBILITY.md)

## What the experiment actually found

On a frozen 170-example project test split, TF-IDF + Logistic Regression C=4 achieved the highest Macro-F1 (90.36%) and accuracy (92.94%). The text-OR-LoRA-Qwen combination achieved the highest positive-class recall (91.49%), at the cost of 12 false positives versus zero for the selected text baseline. LoRA improved the matched Qwen test Macro-F1 from 76.65% to 82.36%.

The predeclared primary text-OR-multimodal-Jev system achieved 87.25% test Macro-F1. It did **not** outperform the selected text baseline. A separate validation ablation showed a benefit from adding Qwen multimodal evidence to text-only Jev (69.42% → 84.25% Macro-F1 at t=0.40); it is not an image-only causal comparison or an independent final test result.

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
