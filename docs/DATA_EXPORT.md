# 私有数据与 badcase 导出

用户已选择：公开工程目录说明与工具，真实样本和逐条错误记录暂不公开。数据再分发权限尚未明确；原创代码 MIT 许可不授予数据、模型或服务的再分发权。

## 输出内容与边界

复用 AutoDL 现有原图、转录、978/172/170 条 manifest 和已完成 Jev 缓存。不重新下载、不重新切分、不改标签，不调用模型或 API，不读取密钥。仅需 Python 3.12+ 标准库，可在无卡模式执行。

```text
outputs/private_dataset_badcases_v1/  # 新建的私有导出包，不覆盖旧输出
├── data/
│   ├── train/                       # 978原图 + 同名txt
│   ├── val/                         # 172原图 + 同名txt
│   ├── test/                        # 170原图 + 同名txt
│   ├── labels.csv                   # 仅image_name,label两列，1320行
│   └── manifest_PRIVATE.jsonl
├── badcase/                         # val20 + test27；FP/FN分别存放
├── README_PRIVATE.md
└── completion_PRIVATE.json          # 私有源哈希、完成状态与数量
outputs/private_dataset_badcases_v1_PRIVATE.zip
```

第一列是带划分前缀的相对图片名称，如 `val/123.jpg`；第二列是整数 0/1。同名 `.txt` 保存原转录，不拼接模型解释。原始排除的 40 条不加入已用划分，原文件不删除。导出包保存真实数据，**不要提交该包、CSV、案例 JSON 或 ZIP 到 GitHub**。

## 1. 将导出脚本送到 AutoDL

如果本地已经取得当前仓库源码，在 Windows CMD 中执行一次，目标脚本若已存在先核对，不重复覆盖：

```bat
scp -P YOUR_SSH_PORT "E:\path\to\social-risk-qwen-jev\scripts\export_dataset_badcases.py" root@YOUR_SSH_HOST:/root/autodl-tmp/social_risk_jev/src/export_dataset_badcases.py
```

请使用自己的实际 SSH 地址与端口；不把密码/API key 发给助手或写进命令。公开仓库也包含此脚本，但必须在已推送新版本后才能从 GitHub 获取。

## 2. 零写入预检，然后导出

在 VS Code 的 AutoDL 远程终端执行：

```bash
cd /root/autodl-tmp/social_risk_jev || exit 1

/root/autodl-tmp/venvs/social_risk_cpu/bin/python \
  src/export_dataset_badcases.py --archive --plan-only
```

应显示 `TRAIN=978 VAL=172 TEST=170 CSV_ROWS=1320 BADCASE_VAL=20 BADCASE_TEST=27`、`API_CALLS=0`、`KEY_READ=False`。预检不写文件；如果退出码不是 0，请先排查，不直接跑下一步。

```bash
/root/autodl-tmp/venvs/social_risk_cpu/bin/python -u \
  src/export_dataset_badcases.py --archive
```

导出前必须不存在目标目录与 ZIP；已有输出会停止，绝不删除或覆盖。处理中断保留部分目录，不把它当完成包。如确需重试，用不同的 **新目录名** 并重新预检，不覆盖部分输出。正常完成应显示 `EXPORT_COMPLETE CSV_ROWS=1320 BADCASES=47 SOURCE_FILES_UNCHANGED=True`。

工具校验本次消费的文件哈希与关键历史来源绑定，复制图片后校验 SHA-256，并在结束前复核源文件未变。它不重新执行政治审查或全量图像可读性/近重复审核；既有数据治理限制仍保留。

## 3. 下载私有 ZIP（Windows CMD）

先在 Windows 创建一个未用于 GitHub 的私有存放目录，再下载；同名 ZIP 已存在时不覆盖：

```bat
if not exist "E:\path\to\social_risk_private" mkdir "E:\path\to\social_risk_private"
if exist "E:\path\to\social_risk_private\ltedi_data_badcases_PRIVATE.zip" (
  echo FILE_ALREADY_EXISTS_STOP
) else (
  scp -P YOUR_SSH_PORT root@YOUR_SSH_HOST:/root/autodl-tmp/social_risk_jev/outputs/private_dataset_badcases_v1_PRIVATE.zip "E:\path\to\social_risk_private\ltedi_data_badcases_PRIVATE.zip"
)
```

将 ZIP 解压到一个新的私有子目录，即得到 `data/train,val,test`、两列 CSV 与 `badcase`。不要对父工作区执行 `git add .`，也不要把私有副本复制进其他不具备忽略规则的仓库。

## 来源与结果核对

- `data/processed/ltedi_grouped_v1/{train,val,test}.jsonl` 与 `preparation_report.json`。
- 验证 Qwen：`outputs/ltedi_evidence_val172_v1/lora_evidence/predictions.json`。
- 验证 Jev：`outputs/jev_lora_remaining152_cloud_v1/predictions_LOCAL_ONLY.json`。
- 测试 Qwen：`outputs/ltedi_frozen_test170_v1/lora_evidence.json`。
- 测试 Jev：`outputs/ltedi_frozen_test170_v1/cloud/predictions_LOCAL_ONLY.json`。
- 两个阶段各自保存的请求体、控制映射、完成报告与来源绑定。

缺失/未完成/被更改的缓存会停止；不以错误样本子集冒充全量，不用当前 Qwen 重新生成旧 Jev 输入，也不自动补发请求。验证与测试案例都只用于诊断，不构造 DPO 对、不调阈值、不重标旧真值。

## 公开发布防线

`data` 和 `badcase` 仅允许明确列出的 README 路径被公开跟踪，其他文件默认忽略。`scripts/check_publication.py` 即使遇到强行跟踪的真实 CSV、图片或逐条 JSON 也会阻止发布；说明文件仍接受密钥扫描。发布前仍须人工检查 Git staged 文件，脚本不是完整隐私检测保证。
