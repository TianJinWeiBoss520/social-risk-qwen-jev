# Badcase：LoRA 图文感知＋Jev 的私有错误案例

只导出已有缓存中 **LoRA Qwen 图像＋文本证据 → Jev** 的误报与漏报，固定 `jev-1.13.0`、原政策、分类阈值 `t=0.40`。不是单独 Qwen 的错误、纯 Jev 的错误，也不是“文本 OR Jev”组合规则的错误。高/中/低风险区间与二分类阈值是不同概念。

| 划分 | 误报 FP | 漏报 FN | 总错误 |
|---|---:|---:|---:|
| Validation（172 条） | 3 | 17 | 20 |
| Test（170 条） | 12 | 15 | 27 |

这些是已记录结果对应的预期数量，不代表真实案例已下载到公开仓库。导出工具会核对完整缓存、请求—样本映射、请求哈希、来源绑定与混淆矩阵；不符时停止，不猜测、不重新调用 API。

私有导出结构：

```text
badcase/
├── val/{false_positive,false_negative}/<image_stem>/
├── test/{false_positive,false_negative}/<image_stem>/
│   ├── <original_name>.jpg   # 原始图片字节
│   ├── text.txt              # 原始中文转录
│   ├── qwen_perception.json  # 七字段感知；label 仅供本地诊断
│   ├── jev_input.json        # 已保存的实际请求体，非重新生成的提示词
│   ├── jev_output.json       # 已保存的分数、模型、usage、耗时字段
│   └── result.json           # 真值、预测、0.40阈值、FP/FN、请求哈希
└── index_PRIVATE.json        # 分划分的私有案例索引
```

`jev_input.json` 只有既有的基础脱敏转录、六个证据字段、模型与政策；不含 Qwen 最终标签或真实标签。`jev_output.json` 是既有结果的解析字段，不伪装成完整 HTTP 原始响应。风险分数未被证明是校准概率，Qwen 解释不是事实标注，也不自动断言失效原因。

真实图片/文本/输入/输出均私有。公开 GitHub 不存逐条案例，也不添加冒充真实案例的占位数据。导出文件标记 `eligible_for_training=false`；后续 SFT / DPO 应另用训练集内的样本和独立评估。见 [导出指南](../docs/DATA_EXPORT.md)。
