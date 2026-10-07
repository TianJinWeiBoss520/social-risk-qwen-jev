# 贡献指南

1. 先阅读 [评测协议](docs/EVALUATION.md) 和 [安全边界](SECURITY.md)。
2. 安装 CPU 开发依赖，执行 `python scripts/run_tests.py`。
3. 修改 policy、prompt、阈值或训练参数必须新建实验版本，记录来源和数据划分。
4. 新增结果需提供样本数、混淆矩阵、Macro-F1、精确率、召回率和可比口径。不得用验证集挑参数后把同一成绩称为最终测试。
5. 不提交图片、逐条预测、API账本、用户审核记录、模型权重、密钥或第三方数据。
6. 提交前执行 `python scripts/check_publication.py --require-license`。

历史 `experiments/v1/src` 为来源保全代码，不原地重写；改进放入新版本并附回归测试。
