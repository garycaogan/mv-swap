# 参与贡献

欢迎提 issue 和 PR。改代码前先跑离线测试（不联网、不要 key、不花钱）：

```bash
python -m unittest discover -s tests -v
```

覆盖帧数补齐、切镜头（含漏切口和闪帧）、生成尺寸、单人/双人/第二遍补换的提示词、节点图的输入和节点变更检查；`tests/test_resume.py` 用假的 RunningHub 和 APIMart 查断点续跑（付费结果落盘不重调、已提交任务只收不重交、没过审时停在超分前）；`tests/test_references.py` 和 `tests/test_framing.py` 查参考图（首帧改人的输入顺序、验图不过只重画一次、生图任务和费用落盘不重交、接口拒绝原文保留）、逐秒审片（任一秒景别、机位、姿势不对就不过，缺秒也不过）和 `--max-segment-seconds` 切段不丢帧。其中两项会真调本机 ffmpeg 抽帧、拼诊断片，没装 ffmpeg 就跳过；想拿别的 ffmpeg 测，把 `MVSWAP_TEST_FFMPEG` 指向它（同目录要有 ffprobe）。CI 在 Python 3.10、3.12、3.13 上跑同一套测试。

动了生成、审片或拼片这些要花钱的环节，PR 里写一下真跑的结果：用一条 10 秒左右的高清单人舞蹈片段端到端跑一遍（约 ¥1、15 分钟），看 `report.md`、`final/compare.mp4`，成片帧数要等于原片帧数。

不要把 key、任务目录、生成的视频和别人的素材提交进来（`docs/media/` 里的演示素材除外，它们不进打包）；工作流节点图运行时从 RunningHub 现取，也不要提交进来。
