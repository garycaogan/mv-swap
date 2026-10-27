---
name: mv-swap
description: 一键 AI 换人翻拍（MV、舞蹈、影视片段）：给一条高清原片和「原片里的人=角色图」，自动切镜头、认人、在 RunningHub 上用 MiniMax H3 换人生成、审片、重跑、超分、拼片，出成片、对照片和成本报告。Use for 换人翻拍、把 MV 或舞蹈里的人换成自己的角色、动作迁移成片、character-swap remakes with MiniMax H3 on RunningHub. Not for original scripted shorts (use h3-ref-video).
---

# mv-swap

工具就在这个目录里：`mvswap.py` 是入口，`README.md` 讲原理、成本和每一步为什么这么做。跑之前确认 `.env`（或环境变量）里有 `RUNNINGHUB_API_KEY` 和 `APIMART_API_KEY`，机器上有 Python 3.10+、ffmpeg，装过 `requirements.txt`。

1. **先要高清原片**：720p 以上，最好 1080p 的本地文件。给的是链接就用 yt-dlp 下（工具会自己调），但很多网站游客只给低清，低于 720p 工具会停下；这时找别的高清来源或者问用户要文件，用户明确说可以用低清才加 `--allow-lowres`。默认选真人 MV、舞蹈；动画原片能跑但效果一般，用户点名要才做，而且角色图要先按原片画风重画。
2. **定选角**：`--cast "原片里的人（中文描述）=角色图路径"`，一人一条。左边写整条片都不变的特征（发色、演出服），别写发型长短、扎没扎头发这种会变的。角色图要全身、穿着成片里的那套衣服、背景干净。同性别替换最稳。
3. **跑**：
   ```bash
   python mvswap.py run jobs/<任务名> --source <原片> --start <秒> --end <秒> --cast "…=…" --cast "…=…"
   ```
   一条 1 分半的 MV 大约半小时、¥10 上下（钱包够让所有镜头一起跑时）；会员 key 一次只跑 5 个，会慢一些。断了就原样再跑 `run jobs/<任务名>`，不会重复花钱。跑之前用 `status` 或 RunningHub 控制台看一眼余额。
4. **收尾**：读 `report.md`，看 `final/compare.mp4`，把 `final/sheet.jpg` 过一遍，确认没有原来的人露脸。报告里「三次都没过审」的镜头要人看：审片模型偏严，多数能用。真不能用的手工修，改完 `state.json` 再跑一次 `run` 会补超分、重拼：很短的镜头用原片帧把露脸的几帧换掉；多人小人镜头三次都站错位置，就用生图按原帧重画、做成缓慢推拉，记成 graphic；认错人就只给这个镜头改写原片描述（`src_desc`）重跑。
5. 这个工具不发布、不做封面。成片里是真人肖像时，提醒用户自己确认原片和角色图的版权、肖像权。
