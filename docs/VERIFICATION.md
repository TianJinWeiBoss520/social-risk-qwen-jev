# 本次工程整理验收

日期：2026-10-07；本地 Windows / Python 3.12.3。原 AutoDL GPU 实验和 Jev 结果没有重新执行。

- 公共接口测试、历史合成/mock回归：最新完整运行 **141 次测试执行，全部通过，无 skip**，包含新增纯 Jev 补充测试的完整性、协议范围和指标重算检查。历史测试复用类，因此不是宣称141个独立功能。
- CPU测试中的XGBoost使用原本地测试依赖；依赖没有拷入公开仓库。真实GPU推理、真实训练和云端调用不属于这些回归测试。
- 首次全量中一个本地HTTP测试出现WinError 10053；单独复测及之后完整复测通过。没有修改原始归档源码或静默跳过此测试；仍需留意Windows本地连接的潜在不稳定性。
- 13项测试结果（11项原冻结方法＋2项纯 Jev 事后补充固定阈值对照）、38项验证结果的指标由公开混淆矩阵重算；README与验证全表通过内容一致性检查。新增结果来自用户提供的完整170条日志；没有重新调用 API，也没有把补充方法描述为预注册实验。
- 历史30个Python源码/测试文件与源副本逐字节校验；仓库manifest保存SHA-256，测试验证一致性。LF属性避免跨平台checkout改变来源哈希。
- 轻量wheel离线构建成功；`demo`与排行榜入口验证成功，API调用数为0。合成demo不是实测模型预测。
- 私有工件/疑似密钥检查：0项发现。不证明完全不存在PII或任意形式密钥；仍需人工审核staged文件。
- GitHub Actions已在公开仓库完成验证。[提交83daec5的CI运行](https://github.com/TianJinWeiBoss520/social-risk-qwen-jev/actions/runs/37609046148)全部通过：安装、CPU张量依赖、发布检查、完整回归、demo和排行榜。首次运行的历史回归因未声明CPU PyTorch而失败，已补充CPU-only安装步骤；不改历史源码、不跳过失败测试。
- 作者已于2026-10-07确认MIT，根目录LICENSE与包元数据已补齐。发布检查使用`--require-license`。
- 已发布公开仓库 [TianJinWeiBoss520/social-risk-qwen-jev](https://github.com/TianJinWeiBoss520/social-risk-qwen-jev)，GitHub识别MIT；初次发布76个公开文件，纯 Jev 补充说明新增1个文档。未改写用户个人主页README；可自行在主页置顶项目。
- Git HTTPS发布修复时遇到网络连接重置，改用GitHub官方Git对象API；校验树/提交SHA与本地一致，非强制快进更新，不覆盖历史或其他项目。

原始工具目录、私人审核数据、远端训练产物未修改或删除。公开候选不包含图片、转录、逐条预测、私有HTML、权重或真实密钥。
