# 安全说明

- key 只从环境变量或仓库根目录的 `.env` 读（`.env` 已在 `.gitignore` 里），代码只读不写；所有存下来或打印出来的接口返回都会把 key 替换成 `[key]`。任务状态里记的是任务号，不记 key。
- 上传的视频和图片会传到 RunningHub 和 APIMart，按它们自己的保存规则存放（APIMart 上传的图保留 72 小时）。别传你没有权利处理的素材。
- 发现安全问题（比如 key 出现在日志、报告或任务目录里），请在 GitHub 上用 [Security Advisory](https://github.com/garycaogan/mv-swap/security/advisories/new) 私下报告，不要开公开 issue。
