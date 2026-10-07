# Data：公开目录说明，真实样本私有

此目录公开说明文件，不再分发第三方数据。原图、转录、标签 CSV 和逐条 manifest 只保存在所有者的 AutoDL / 本地私有副本中。项目 MIT 许可不覆盖数据；上游可访问不等于取得再分发许可。

沿用已完成实验的 manifest，不重新切分：

| 目录 | 条数 | 标签 0 | 标签 1 | 来源与用途 |
|---|---:|---:|---:|---|
| [train](train/README.md) | 978 | 700 | 278 | 原始 train 中的训练样本 |
| [val](val/README.md) | 172 | 123 | 49 | 原始 train 中保留的验证样本 |
| [test](test/README.md) | 170 | 123 | 47 | 原始 dev 保留的项目测试样本 |

私有导出包的 `data/labels.csv` **恰好两列**：第一列 `image_name`，第二列 `label`。第一列使用 `train/123.jpg` 这样的相对图片名称，避免跨目录同名歧义；每条对应一个真实图片文件。`0 = Not-Misogyny`，`1 = Misogyny`，不是风险程度分档。CSV 为 UTF-8 BOM，可在 Excel 中直接打开。

每个划分目录保存原始 JPG 字节与同名 UTF-8 `.txt` 转录。`manifest_PRIVATE.jsonl` 记录对应关系及来源哈希，不公开。导出是本地文件复制，不是重新下载、不改源标签、不删除原始排除样本。

源码：[离线私有导出工具](../scripts/export_dataset_badcases.py)。操作见 [私有数据与 badcase 导出指南](../docs/DATA_EXPORT.md)。真实数据生成后仍会被 `.gitignore` 忽略，发布检查也会阻止误提交。

数据来源：[Mysogyny-Meme-Detection](https://github.com/AJFaisal002/Mysogyny-Meme-Detection)，固定 commit `0179687f197a0b2babddcb4eb32547a07cba0933`。政治内容状态仍为用户抽查，未全量验证；详见 [数据卡](../docs/DATA_CARD.md)。
