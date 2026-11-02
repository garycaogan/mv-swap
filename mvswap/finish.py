"""Graphic shots, the final cut, the side-by-side and the report."""
import json
import time
from pathlib import Path

from . import ai, config, framing, media


def final_size(width: int, height: int) -> tuple[int, int]:
    s = config.FINAL_LONG_SIDE / max(width, height)
    return round(width * s / 2) * 2, round(height * s / 2) * 2


def _aspect_word(w: int, h: int) -> str:
    ar = w / h
    return min({"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4}.items(), key=lambda kv: abs(kv[1] - ar))[0]


def graphic_shot(shot: dict, cast: list[dict], prep: dict, job: Path) -> dict:
    """Flat graphics (cut-out heads, drawings) can't go through the video model: redraw the middle frame with our
    people in the same graphic style, then hold it with a slow push-in for the shot's length."""
    mv = Path(prep["mv"])
    d = job / "graphic"
    d.mkdir(exist_ok=True)
    key = media.keyframes(mv, [shot["f0"] + shot["frames"] // 2], d, shot["id"], width=prep["width"])[0]
    by = {c["id"]: c for c in cast}
    who = [by[p] for p in shot["people"]]
    swaps = " ".join(f"Replace {c['src_desc']} with the person from image {i} ({c['desc']}), keeping the size, angle and expression of the one replaced."
                     for i, c in enumerate(who, start=2))
    still = d / f"{shot['id']}.png"
    paid = d / f"{shot['id']}.cny"
    if still.exists() and paid.exists():                 # a resume after a crash: the redraw is already paid for
        cny = float(paid.read_text())
    else:
        cny = ai.image(still, "Recreate image 1 exactly: the same layout, graphic style, colours, background, decorations and any words. "
                        + swaps + " Keep the flat graphic look of image 1 (cut-out, sticker or drawing), sharp and clean.",
                        [key] + [Path(c["picture"]) for c in who], size=_aspect_word(prep["width"], prep["height"]))
        paid.write_text(str(cny))
    from PIL import Image                                 # this machine's ffmpeg has no PNG decoder
    jpg = d / f"{shot['id']}-still.jpg"
    Image.open(still).convert("RGB").save(jpg, quality=95)
    w, h = final_size(prep["width"], prep["height"])
    n = shot["frames"]
    out = d / f"{shot['id']}.mp4"
    media.ff("-loop", "1", "-i", str(jpg), "-vf",
             f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
             f"zoompan=z='1+0.05*on/{max(n - 1, 1)}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={w}x{h}:fps={config.FPS},format=yuv420p",
             "-frames:v", str(n), "-c:v", "libx264", "-crf", "14", str(out))
    return {"file": str(out), "cny": cny}


def assemble(state: dict, job: Path) -> dict:
    prep = state["prepared"]
    w, h = final_size(prep["width"], prep["height"])
    scale = f"scale={w}:{h}:flags=lanczos,unsharp=5:5:0.4:5:5:0.0,setsar=1,format=yuv420p"
    work = job / "work"
    work.mkdir(exist_ok=True)
    enc = ["-r", str(config.FPS), "-an", "-c:v", "libx264", "-crf", "14", "-preset", "slow"]
    parts = []
    for s in state["shots"]:
        part = work / f"{s['id']}.mp4"
        chosen = s.get("chosen") and next((a for a in s.get("attempts", []) if a["n"] == s["chosen"]), None)
        up = s.get("upscaled") or {}
        if s["kind"] == "swap" and chosen and up.get("file") and up.get("n") == chosen["n"] and Path(up["file"]).exists():
            gr, orr, gg, og, gb, ob = up["fix"]
            lut = f"lutrgb=r='clip(val*{gr}+({orr}),0,255)':g='clip(val*{gg}+({og}),0,255)':b='clip(val*{gb}+({ob}),0,255)'"
            media.ff("-i", up["file"], "-vf", f"trim=end_frame={s['frames']},setpts=N/{config.FPS}/TB,format=rgb24,{lut},{scale}", *enc, str(part))
        elif s["kind"] == "swap" and chosen:
            media.ff("-i", chosen["file"], "-vf", f"trim=end_frame={s['frames']},setpts=N/{config.FPS}/TB,{scale}", *enc, str(part))
        elif s["kind"] == "graphic" and s.get("graphic"):
            media.ff("-i", s["graphic"]["file"], "-vf", f"trim=end_frame={s['frames']},setpts=N/{config.FPS}/TB,{scale}", *enc, str(part))
        else:                                                   # keep shots, and anything that never got a usable take
            a = s["f0"]
            media.ff("-i", prep["mv"], "-vf", f"select='between(n,{a},{a + s['frames'] - 1})',setpts=N/{config.FPS}/TB,{scale}", *enc, str(part))
        parts.append(part)
    (work / "list.txt").write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    media.ff("-f", "concat", "-safe", "0", "-i", str(work / "list.txt"), "-c", "copy", str(work / "video.mp4"))
    out = job / "final"
    out.mkdir(exist_ok=True)
    film = out / "film.mp4"
    dur = prep["frames"] / config.FPS
    if prep.get("audio"):
        media.ff("-i", str(work / "video.mp4"), "-i", prep["audio"], "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                 "-af", f"atrim=0:{dur:.3f},afade=t=out:st={max(dur - 0.5, 0):.3f}:d=0.5", "-c:a", "aac", "-b:a", "192k",
                 "-t", f"{dur:.3f}", "-movflags", "+faststart", str(film))
    else:
        media.ff("-i", str(work / "video.mp4"), "-c", "copy", "-movflags", "+faststart", str(film))
    side = out / "compare.mp4"
    landscape = w >= h
    half = (960, round(960 * h / w / 2) * 2) if landscape else (round(960 * w / h / 2) * 2, 960)
    stack = "vstack" if landscape else "hstack"
    args = ["-i", prep["mv"], "-i", str(film), "-filter_complex",
            f"[0:v]scale={half[0]}:{half[1]},setsar=1[a];[1:v]scale={half[0]}:{half[1]},setsar=1[b];[a][b]{stack},format=yuv420p[v]",
            "-map", "[v]"]
    args += ["-map", "1:a", "-c:a", "copy"] if prep.get("audio") else []
    media.ff(*args, "-c:v", "libx264", "-crf", "22", str(side))
    sheet = out / "sheet.jpg"
    secs = max(1, round(prep["frames"] / config.FPS))
    cols, rows = min(10, secs), min(10, -(-secs // 10))          # a 9-second clip gets one row, not a 10x10 canvas
    media.ff("-i", str(film), "-vf", f"fps=1,scale=192:-2,tile={cols}x{rows}", "-frames:v", "1", str(sheet))
    panels, _ = framing.pairs(Path(prep["mv"]), film, {"f0": 0, "frames": prep["frames"]}, out / "seconds", "final", width=256)
    seconds_sheet = out / "compare-seconds.jpg"
    framing.contact_sheet(panels, seconds_sheet)
    return {"film": str(film), "compare": str(side), "sheet": str(sheet), "compare_seconds": str(seconds_sheet),
            "frames": media.frame_count(film), "want": prep["frames"]}


def failed_shots(state: dict) -> list[dict]:
    return [s for s in state.get("shots", []) if s["kind"] == "swap"
            and not any((a.get("review") or {}).get("pass") for a in s.get("attempts", []))]


def _failure_reason(a: dict) -> str:
    rv = a.get("review") or {}
    if rv.get("issue"):
        return str(rv["issue"]).replace("\n", " ")
    reasons = []
    for key, label in (("originals_left", "原人物仍在"), ("extra_people", "多出人物"), ("added_overlays", "新增贴字或贴纸"), ("broken", "画面崩坏")):
        if rv.get(key):
            reasons.append(label)
    if rv.get("right_people") is False or rv.get("missing"):
        reasons.append("没有换对全部人物")
    if rv.get("style_kept") is False:
        reasons.append("画风改变")
    if rv.get("framing") is not None and int(rv["framing"] or 0) < 1:
        reasons.append("景别不符")
    if rv.get("score") is not None and float(rv["score"] or 0) < 6:
        reasons.append("评分低于 6")
    if rv.get("backdrop_lost_frames"):
        reasons.append(f"背景偏离 {rv['backdrop_lost_frames']} 帧")
    return "；".join(reasons) or str(a.get("error") or "未取得可用的过审结果").replace("\n", " ")


def report(state: dict, job: Path) -> Path:
    shots = state["shots"]
    rh_cny = sum(a.get("cny", 0) for s in shots for a in s.get("attempts", []))
    used = sum(next((a.get("cny", 0) for a in s.get("attempts", []) if a["n"] == s.get("chosen")), 0) for s in shots)
    gpu = sum(a.get("run_s", 0) for s in shots for a in s.get("attempts", [])) / 60
    graphic_cny = sum((s.get("graphic") or {}).get("cny", 0) for s in shots)
    reference_cny = sum(a.get("cny", 0) for s in shots for history in s.get("reference_attempts", {}).values() for a in history)
    up_cny = sum((s.get("upscaled") or {}).get("cny", 0) for s in shots)
    gpu += sum((s.get("upscaled") or {}).get("run_s", 0) for s in shots) / 60
    wall = (state.get("t_done") or state.get("t_reference_done") or time.time()) - state.get("t_start", time.time())
    kinds = {k: sum(1 for s in shots if s["kind"] == k) for k in ("swap", "keep", "graphic")}
    flagged = failed_shots(state)
    preparing_refs = not state.get("final") and (state.get("references_ready") or state.get("reference_blocked"))
    if preparing_refs:
        flagged = []
    layout = "上原片下成片" if state["prepared"]["width"] >= state["prepared"]["height"] else "左原片右成片"
    lines = [
        f"# 换人成片报告：{job.name}", "",
        ("参考图准备阶段，尚未调用 H3、没有成片。" if preparing_refs else
         f"{'诊断用成片' if state.get('review_blocked') else '成片'} `final/film.mp4`，对照 `final/compare.mp4`（{layout}），逐秒缩略图 `final/sheet.jpg`。"),
        f"原片 {state['prepared']['width']}×{state['prepared']['height']}，{state['prepared']['frames'] / config.FPS:.1f} 秒，"
        f"{len(shots)} 个镜头：换人 {kinds['swap']}、用原片 {kinds['keep']}、图形 {kinds['graphic']}。", "",
        f"花费：RunningHub 生成 ¥{rh_cny:.2f}（成片用到的 ¥{used:.2f}）、超分 ¥{up_cny:.2f}，图形镜头生图 ¥{graphic_cny:.2f}，"
        f"参考图编辑 ¥{reference_cny:.2f}，视觉模型约 ¥{state.get('vision_cny', 0):.2f}；"
        f"合计约 ¥{rh_cny + up_cny + graphic_cny + reference_cny + state.get('vision_cny', 0):.2f}；"
        f"GPU {gpu:.0f} 分钟；{'参考图准备' if preparing_refs else '从开始到成片'} {wall / 60:.0f} 分钟。", "",
    ]
    warn = [c for c in state["cast"] if c.get("src_gender") and c.get("gender") and c["src_gender"] != c["gender"]]
    if warn:
        lines += ["性别对不上的替换（男换女、女换男不稳，留意这些人的镜头）：" + "、".join(f"{c['hint']}→{c['name']}" for c in warn), ""]
    if state.get("reference_blocked"):
        lines += ["参考图失败或被接口拒绝，已停止，没有改写提示词绕过。原始错误：", "", "```text", state["reference_blocked"], "```", ""]
    if flagged:
        lines += [f"**{len(flagged)} 个镜头所有版本都没过审**，需要人看：" + "、".join(s["id"] for s in flagged),
                  "片里用的是已完成版本中分数最高的那次；没有可用版本的镜头暂用原片。", ""]
        if state.get("review_blocked"):
            lines += ["已在付费超分前停下，只在本地免费拼了诊断片和对照片。",
                      "看过对照片觉得能用，再加 `--accept-failed` 重跑会接着超分：",
                      f'`python mvswap.py run "{job}" --accept-failed`', ""]
        elif state.get("accept_failed"):
            lines += ["本次已通过 `--accept-failed` 接受未过审镜头，继续超分；审片结果仍保留供核对。", ""]
        for s in flagged:
            lines.append(f"- {s['id']}（{len(s.get('attempts', []))} 个版本）：")
            for a in s.get("attempts", []):
                lines.append(f"  - 版本 {a['n']}：{_failure_reason(a)}")
            if not s.get("attempts"):
                lines.append("  - 没有生成结果，片里暂用原片。")
        lines.append("")
    lines += ["| 镜头 | 秒 | 类型 | 谁 | 次数 | 采用 | 分 | 问题 |", "|---|---|---|---|---|---|---|---|"]
    for s in shots:
        ch = next((a for a in s.get("attempts", []) if a["n"] == s.get("chosen")), {})
        rv = ch.get("review") or {}
        lines.append(f"| {s['id']} | {s['f0'] / config.FPS:.1f}–{(s['f0'] + s['frames']) / config.FPS:.1f} | {s['kind']} | "
                     f"{','.join(s.get('people') or [])} | {len(s.get('attempts', []))} | {s.get('chosen') or '-'} | {rv.get('score', '')} | "
                     f"{(rv.get('issue') or s.get('note') or '').replace('|', '/')[:60]} |")
    lines += ["", "## 每镜参考图", "", "| 镜头 | 人物 | 原片景别 | H3 参考图 | 来源 / 方法 |", "|---|---|---|---|---|"]
    for s in shots:
        for person, ref in (s.get("references") or {}).items():
            lines.append(f"| {s['id']} | {person} | {ref['framing']} | `{ref['picture']}` | `{ref.get('source', '')}` / {ref['method']} {ref.get('crop_box', '')} |")
    lines += ["", f"参考图模式：`{state.get('args', {}).get('ref_mode', 'crop')}`。neutral 只输入角色主图生成中性姿势；keyframe 输入原片首帧和角色主图；crop 仅按景别裁图。参考图姿势可能被 H3 照搬。", ""]
    for s in shots:
        for person, history in s.get("reference_attempts", {}).items():
            for a in history:
                review = a.get("review") or {}
                lines.append(f"- {s['id']} / {person} 参考图 {a['n']}：`{a['file']}`，任务 `{a.get('task', '-')}`，费用 ¥{a.get('cny',0):.4f}，"
                             f"{'通过' if review.get('pass') and a.get('manual_pass') is not False else '未通过/未审'}；{review.get('issue') or a.get('error') or ''}")
                lines.append("  输入：" + "、".join(f"`{p}`" for p in a.get("pictures", [])))
    lines += ["", "## 逐秒审片", "", "逐秒及每段首尾同帧对照：景别改变一档、机位/姿势不符、回到参考图姿势、原片没露出的下半身出现，任一项直接不过；缺项也不过，总分不覆盖规则。", ""]
    for s in shots:
        for a in s.get("attempts", []):
            rv = a.get("review") or {}
            lines.append(f"- {s['id']} 版本 {a['n']}：{'通过' if rv.get('pass') else '不过'}，{rv.get('score', '-')} 分；{rv.get('issue') or '无问题'}；证据 `review/{s['id']}-{a['n']}/`。")
    cap = state.get("args", {}).get("max_segment_seconds", 0)
    lines += ["", f"每段最长秒数：{cap or '关闭'}；开启后保留原片切镜，超长镜头在动作变化较小处切开。当前 H3 工作流没有独立首帧约束输入，未伪造末帧衔接支持；接缝须看逐秒对照和整片。", ""]
    p = job / "report.md"
    p.write_text("\n".join(lines) + "\n")
    return p
