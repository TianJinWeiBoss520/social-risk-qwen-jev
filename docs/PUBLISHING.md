# 发布到 GitHub

目标账号：`TianJinWeiBoss520`；公开仓库名称 `social-risk-qwen-jev`。作者已确认MIT许可证，本机Git已核对登录账号；不需要安装GitHub插件。是否完成发布应以实际远端文件与推送结果为准，不以本文的发布计划为准。

## 发布前

1. 原创代码采用作者已确认的MIT许可；确保LICENSE、README与包元数据一致。
2. 执行离线测试；检查首屏测试排行榜与验证集表分别标注。
3. 执行 `python scripts/check_publication.py --require-license`。
4. 仅在本仓库根目录初始化 Git，不对父目录 `E:\smart_car_searching` 做 `git add .`。父目录包含其他项目与私有配置。
5. 在提交前审查 `git diff --cached --stat` 和 staged 文件清单；数据、日志、权重、私人 HTML/JSON、密钥不在其中。

## 账号连接后的正常发布流程

使用浏览器或经授权的 GitHub API 创建新的公开仓库，再通过已登录的 Git 推送；不覆盖同名已有仓库，不 force-push。如果仓库已存在，先只读检查其内容与默认分支，避免混入用户其他项目。凭据只交给 Git/官方服务，不打印或提交到项目。

网页上传方式也应只上传审核后的 public 文件；不要上传整个旧工具目录。上传 ZIP 到网页不会自动产生可浏览的源码目录。

## 让个人主页显眼

发布完成后在仓库 Description 写：

> Chinese multimodal moderation: Qwen3-VL LoRA, Jev decisions, reproducible baselines and privacy-aware evaluation.

Topics 可选 `multimodal`, `qwen3-vl`, `lora`, `content-moderation`, `risk-detection`, `model-evaluation`。随后在个人主页选择 **Customize your pins**，将仓库置顶；GitHub 最多显示 6 个 pinned 项目。[官方说明](https://docs.github.com/en/account-and-profile/how-tos/profile-customization/pinning-items-to-your-profile)

若已有个人主页 README，不自动重写它；只添加一段项目介绍并保留原文。需要单独确认要更新个人主页 README，而不是只创建项目仓库。

## 发布后的验收

- 浏览项目首页，确认 README 首屏测试表渲染。
- 查看 Actions 实际结果；不能把本地通过宣称为 GitHub CI 已运行。
- 确认代码、文档与轻量 CLI 可安装，文件链接有效。
- 确认没有 secret、图片、用户审核记录或逐条预测。
- 在个人主页确认置顶；记录实际仓库 URL 后才对外分享。
