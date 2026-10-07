# 复现指南

## 先区分三种复现

1. **汇总指标复现：** 仓库包含混淆矩阵，CPU 即可重算排行榜；不是重新对数据推理。
2. **流程/安全回归：** 使用合成数据和 mock 模型/API 验证解析、划分、预算、缓存与失败路径；不证明真实模型精度。
3. **原始实验重跑：** 需要合法取得数据、真实基座与适配器、兼容 GPU 环境和可用 Jev 权限。这里不提供图片、转录、私有缓存或权重，不能声称开箱即得相同逐条结果。

## 1. 轻量本地运行

Python 3.12+，项目根目录：

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell：.\.venv\Scripts\Activate.ps1
python -m pip install -e .
social-risk results --split test
social-risk results --split validation
social-risk demo
```

安装工具可访问软件源；三个 CLI 命令自身不调用网络、GPU 或 API。不要安装 GPU 依赖来运行这些命令。

## 2. 运行回归测试

```bash
python -m pip install -e ".[cpu]"
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python scripts/run_tests.py
python scripts/check_publication.py
```

CPU extra 包括图像审计与传统基线依赖。完整历史回归还需要CPU PyTorch进行标签掩码的张量检查；它不加载Qwen、不要求GPU，不需要transformers/peft。CI固定CPU PyTorch 2.7.0，这不是原AutoDL GPU训练版本。根据 [PyTorch官方安装指南](https://pytorch.org/get-started/locally/) 选择CPU构建，避免额外下载CUDA库。测试中云端调用与GPU模型均模拟；不要设置密钥。历史测试之间复用了部分测试类，运行总数可能含重复执行的用例，不应宣传为同等数量的独立功能。

## 3. 历史实验入口

```bash
python scripts/run_experiment.py list
python scripts/run_experiment.py cpu -- --help
python scripts/run_experiment.py lora -- --help
python scripts/run_experiment.py frozen-test -- --help
```

启动器必须显式指定阶段与参数，不默认训练或计费。**历史源码保留 AutoDL 路径及来源哈希约束。** 公共包没有重写它们；不能只把根目录改名就认为全部阶段可无缝迁移。`configs/` 是已观察的配置快照，不会自动驱动这些脚本。

原始工作目录为 `/root/autodl-tmp/social_risk_jev`。在原始 AutoDL 环境继续使用原有 `src/` 和私有产物最稳妥；公开的 `experiments/v1/src/` 是独立备份。若在新机器复跑，要显式提供全部路径、阅读各阶段 `--help`，并把新数据/适配器/代码哈希记录为新实验版本，不要绕过旧缓存的防串用校验。

## 4. 数据与传统基线

数据来源与固定 commit 见 [数据卡](DATA_CARD.md)。自行确认访问和研究使用权限，不在本仓库重新分发。下载后应校验图像字节、可读性、重复候选和跨 split 防泄漏。原版 dev 被保留作项目 test；不从 test 中挖训练 badcase。

准备阶段在确认用户抽查限制后，用 seed 2060、val-fraction 0.15 生成分组清单。`--accept-user-spot-check` 只是记录用户抽查，不是政治筛查合格证。

具备本地合法数据清单后，CPU 基线可先零训练预检，再执行新输出目录：

```bash
python scripts/run_experiment.py cpu -- \
  --data-dir /absolute/path/to/ltedi_grouped_v1 \
  --output-dir /absolute/path/to/new_cpu_run --seed 2060 --plan-only

python scripts/run_experiment.py cpu -- \
  --data-dir /absolute/path/to/ltedi_grouped_v1 \
  --output-dir /absolute/path/to/new_cpu_run --seed 2060
```

TF-IDF 仅 fit train；模型/阈值选择仅看 val。不要反序列化不可信 joblib/pickle 文件。默认输出目录已存在时，不删除以“重新运行”。

## 5. GPU 微调

历史环境记录见 `requirements/gpu-observed.txt`：它记录实际机器软件版本，不保证任意 CUDA/平台可直接安装。不要盲目升级旧实验环境；新建独立 venv，并确认 PyTorch/CUDA/bitsandbytes 与 GPU 匹配。

基座为本地 Qwen3-VL-32B-Instruct 4-bit。历史语言 LoRA 只监督短 JSON 标签；不能称七字段解释也被人工标注训练。小样本 smoke 先确认视觉冻结、512 个 LoRA 张量、约 1992 万训练参数、权重重载及格式计数。

```bash
python scripts/run_experiment.py lora -- \
  --data-dir /absolute/path/to/ltedi_grouped_v1 \
  --model-path /absolute/path/to/qwen3_vl_32b_4bit \
  --cpu-baseline-dir /absolute/path/to/cpu_baselines \
  --output-dir /absolute/path/to/new_lora_run \
  --train-limit 16 --val-limit 5 --plan-only
```

真正执行前删除参数中的 `--plan-only`，而不是删除已有产物。全量实验去掉两个 limit，采用 [模型卡](MODEL_CARD.md) 的配置。**不同种子/软件/数据/提示词可导致差异，不承诺位级一致。** 适配器 checkpoint 不含完整优化器/RNG 状态，不能把适配器续训叫精确恢复训练。

## 6. 证据与 Jev

公共 `prepare_request` 只在本地构建状态；没有隐藏 API 自动发送。真实云端 runner 见历史归档，但它们强绑定已准备好的 v1 工件、源文件哈希与调用账本，因此不能对任意新数据直接调用。

云端前明确审批：样本范围、发送字段、模型版本、最大 POST 数、零重试、预估费用、是否复用缓存。不上传图片、gold label、Qwen 最终标签、路径或样本 ID；文本/证据仍可能含个人信息，基础脱敏不是完整匿名化。真实 key 只能在本地安全输入，不能提交 `.env` 或粘贴到 issue。

遇到 POST 超时且无法确认是否计费，不自动重发；先检查账本和服务账单。软件预算不是 provider 硬限额。API 定价与可用性见 [官方文档](https://docs.typesafe.ai/models)，必须在真正调用前再次确认。

## 7. 冻结评估与暂停

主方案固定为 `TEXT_OR_JEV_GE_0.40`，对照 t=0.50，传统基线选择 C=4。不要观察 test 后重新挑阈值，或删掉低分方法。纯 Jev 消融与私人 review23 属于开发验证记录，不扩写成未做过的 test 结果。

当前暂停点是代码与汇总结果归档，不再启动训练或云端调用。如果继续研究，使用新的独立外部/时间切分测试集，将新结果与本次冻结结果分开。
