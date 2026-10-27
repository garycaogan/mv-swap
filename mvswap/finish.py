"""Graphic shots, the final cut, the side-by-side and the report."""
import json
import time
from pathlib import Path

from . import ai, config, media


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
    return {"film": str(film), "compare": str(side), "sheet": str(sheet), "frames": media.frame_count(film), "want": prep["frames"]}


def report(state: dict, job: Path) -> Path:
    shots = state["shots"]
    rh_cny = sum(a.get("cny", 0) for s in shots for a in s.get("attempts", []))
    used = sum(next((a.get("cny", 0) for a in s.get("attempts", []) if a["n"] == s.get("chosen")), 0) for s in shots)
    gpu = sum(a.get("run_s", 0) for s in shots for a in s.get("attempts", [])) / 60
    graphic_cny = sum((s.get("graphic") or {}).get("cny", 0) for s in shots)
    up_cny = sum((s.get("upscaled") or {}).get("cny", 0) for s in shots)
    gpu += sum((s.get("upscaled") or {}).get("run_s", 0) for s in shots) / 60
    wall = (state.get("t_done") or time.time()) - state.get("t_start", time.time())
    kinds = {k: sum(1 for s in shots if s["kind"] == k) for k in ("swap", "keep", "graphic")}
    flagged = [s for s in shots if s["kind"] == "swap" and not any(a.get("review", {}).get("pass") for a in s.get("attempts", []))]
    lines = [
        f"# 换人成片报告：{job.name}", "",
        f"成片 `final/film.mp4`，对照 `final/compare.mp4`（上原片下成片），逐秒缩略图 `final/sheet.jpg`。",
        f"原片 {state['prepared']['width']}×{state['prepared']['height']}，{state['prepared']['frames'] / config.FPS:.1f} 秒，"
        f"{len(shots)} 个镜头：换人 {kinds['swap']}、用原片 {kinds['keep']}、图形 {kinds['graphic']}。", "",
        f"花费：RunningHub 生成 ¥{rh_cny:.2f}（成片用到的 ¥{used:.2f}）、超分 ¥{up_cny:.2f}，图形镜头生图 ¥{graphic_cny:.2f}，"
        f"视觉模型约 ¥{state.get('vision_cny', 0):.2f}；"
        f"GPU {gpu:.0f} 分钟；从开始到成片 {wall / 60:.0f} 分钟。", "",
    ]
    warn = [c for c in state["cast"] if c.get("src_gender") and c.get("gender") and c["src_gender"] != c["gender"]]
    if warn:
        lines += ["性别对不上的替换（男换女、女换男不稳，留意这些人的镜头）：" + "、".join(f"{c['hint']}→{c['name']}" for c in warn), ""]
    if flagged:
        lines += [f"**{len(flagged)} 个镜头三次都没过审**，成片里用的是分数最高的那次，需要人看：" + "、".join(s["id"] for s in flagged), ""]
    lines += ["| 镜头 | 秒 | 类型 | 谁 | 次数 | 采用 | 分 | 问题 |", "|---|---|---|---|---|---|---|---|"]
    for s in shots:
        ch = next((a for a in s.get("attempts", []) if a["n"] == s.get("chosen")), {})
        rv = ch.get("review") or {}
        lines.append(f"| {s['id']} | {s['f0'] / config.FPS:.1f}–{(s['f0'] + s['frames']) / config.FPS:.1f} | {s['kind']} | "
                     f"{','.join(s.get('people') or [])} | {len(s.get('attempts', []))} | {s.get('chosen') or '-'} | {rv.get('score', '')} | "
                     f"{(rv.get('issue') or s.get('note') or '').replace('|', '/')[:60]} |")
    p = job / "report.md"
    p.write_text("\n".join(lines) + "\n")
    return p
