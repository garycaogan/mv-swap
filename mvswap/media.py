"""Source video in, shots out: download, trim to 24 fps, find cuts, cut padded clips, pull keyframes."""
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from . import config


def ff(*args, quiet=True):
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", *args] if quiet else ["ffmpeg", "-y", *args], check=True)


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    d = json.loads(out)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    num, den = (v.get("r_frame_rate") or "0/1").split("/")
    return {"width": v.get("width") or 0, "height": v.get("height") or 0, "codec": v.get("codec_name"),
            "fps": float(num) / float(den or 1), "duration": float(d.get("format", {}).get("duration") or 0),
            "audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


def frame_count(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    return int(out or 0)


# ------------------------------------------------------------------ source

def fetch_source(src: str, workdir: Path) -> Path:
    """A local file is copied as is; a link goes through yt-dlp (if it is installed), best quality up to 1080p in H.264.
    Many sites only give guests a low-res copy: the 720p check after this will say so."""
    out = workdir / "source.mp4"
    if Path(src).expanduser().exists():
        shutil.copyfile(Path(src).expanduser(), out)
        return out
    if not re.match(r"https?://", src):
        raise SystemExit(f"找不到原片：{src}")
    if not shutil.which("yt-dlp"):
        raise SystemExit("链接要用 yt-dlp 下载（pip install yt-dlp），或者自己下好高清原片，--source 写本地文件")
    subprocess.run(["yt-dlp", "-f", "bv*[height<=1080][vcodec^=avc1]+ba/bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
                    "-o", str(out), src], check=True)
    if probe(out)["codec"] != "h264":                      # AV1 / VP9 / ByteVC: re-encode once so every later step can read it
        tmp = workdir / "source-h264.mp4"
        ff("-i", str(out), "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-c:a", "aac", str(tmp))
        tmp.replace(out)
    return out


def detect_bars(source: Path, start: float, end: float, w: int, h: int) -> tuple[int, int, int, int] | None:
    """Letterbox / pillarbox: rows and columns that stay dark in every sampled frame. (ffmpeg's cropdetect missed the
    film clip's bars: a channel watermark sits in them.) Returns (w, h, x, y) of the picture, or None."""
    import numpy as np
    sw, sh = 192, round(192 * h / w)
    frames = []
    for k in range(12):
        t = start + (end - start) * (k + 0.5) / 12
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(source), "-frames:v", "1", "-vf", f"scale={sw}:{sh}",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
        if len(raw) == sw * sh:
            frames.append(np.frombuffer(raw, np.uint8).reshape(sh, sw))
    if not frames:
        return None
    m = np.max(np.stack(frames), axis=0)
    rows = np.where(np.percentile(m, 90, axis=1) > 28)[0]
    cols = np.where(np.percentile(m, 90, axis=0) > 28)[0]
    if not len(rows) or not len(cols):
        return None
    y0, y1 = rows[0] * h / sh, (rows[-1] + 1) * h / sh
    x0, x1 = cols[0] * w / sw, (cols[-1] + 1) * w / sw
    if (y1 - y0) > 0.97 * h and (x1 - x0) > 0.97 * w:
        return None
    y0, x0 = int(y0) // 2 * 2 + 2, int(x0) // 2 * 2 + (2 if x0 > 0 else 0)      # a couple of pixels inside the edge
    ch = (int(y1) - y0 - 2) // 2 * 2
    cw = (int(x1) - x0 - (2 if x1 < w else 0)) // 2 * 2
    return cw, ch, x0, y0


def prepare(source: Path, start: float, end: float | None, workdir: Path, crop: str | None = "auto") -> dict:
    """mv24.mp4 = the chosen range at exactly 24 fps (CFR, black bars cropped off); audio.m4a = the same range's sound.
    crop: "auto", "none" or "w:h:x:y"."""
    info = probe(source)
    end = min(end or info["duration"], info["duration"])
    box = None
    if crop == "auto":
        box = detect_bars(source, start, end, info["width"], info["height"])
    elif crop and crop != "none":
        box = tuple(int(v) for v in crop.split(":"))
    vf = (f"crop={box[0]}:{box[1]}:{box[2]}:{box[3]}," if box else "") + f"fps={config.FPS}"
    mv = workdir / "mv24.mp4"
    ff("-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}", "-vf", vf, "-an",
       "-c:v", "libx264", "-crf", "12", "-preset", "fast", str(mv))
    if box:
        info["width"], info["height"] = box[0], box[1]
    audio = workdir / "audio.m4a"
    if info["audio"]:
        ff("-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}", "-vn", "-c:a", "aac", "-b:a", "192k", str(audio))
    frames = frame_count(mv)
    return {"mv": str(mv), "audio": str(audio) if info["audio"] else None, "frames": frames, "width": info["width"],
            "height": info["height"], "start": start, "end": end, "crop": list(box) if box else None}


# ------------------------------------------------------------------ shots

def scene_scores(mv: Path) -> list[float]:
    """Scene-change score of every frame (index = frame number)."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(mv), "-vf", "select='gte(scene,0)',metadata=print:key=lavfi.scene_score",
                          "-an", "-f", "null", "-"], capture_output=True, text=True).stderr
    return [float(x) for x in re.findall(r"lavfi\.scene_score=([\d.]+)", err)]


def structure(mv: Path) -> list[float]:
    """Brightness-normalised correlation of each frame with the one before (index = frame number). A flash or a strobe
    changes brightness but keeps the layout (high); a real cut changes the layout (low)."""
    import numpy as np
    w, h = 64, 36
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mv), "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float64)
    fr = fr - fr.mean(axis=(1, 2), keepdims=True)
    norm = np.sqrt((fr * fr).sum(axis=(1, 2)))
    out = [1.0]
    for i in range(1, len(fr)):
        d = norm[i - 1] * norm[i]
        out.append(float((fr[i - 1] * fr[i]).sum() / d) if d > 1e-6 else 1.0)
    return out


def find_shots(scores: list[float], frames: int, max_frames: int, sims: list[float] | None = None) -> list[dict]:
    """Cuts at score > CUT_SCORE, plus quieter cuts (> HIDDEN_CUT_SCORE); with `sims` (from structure()) a candidate
    only counts when the layout really changed (< 0.5), so concert flashes, pushes and fast moves don't split a shot.
    Cuts closer than 6 frames to the previous one are dropped. Shots longer than `max_frames` are split evenly (H3 can't
    make them in one take)."""
    def real(i):
        return sims is None or i >= len(sims) or sims[i] < 0.5
    cands = sorted(i for i, s in enumerate(scores) if s > config.HIDDEN_CUT_SCORE and 0 < i < frames and real(i))
    strong = [i for i in cands if scores[i] > config.CUT_SCORE]
    cuts = list(strong)
    for i in cands:
        if i not in strong and all(abs(i - c) >= 6 for c in cuts):
            cuts.append(i)
    kept = []
    for i in sorted(cuts):
        if not kept or i - kept[-1] >= 6:
            kept.append(i)
    cuts = kept
    edges = [0] + sorted(cuts) + [frames]
    edges = [e for k, e in enumerate(edges) if k == 0 or e - edges[k - 1] >= 3]   # drop flash frames
    if edges[-1] != frames:
        edges.append(frames)
    shots = []
    for a, b in zip(edges, edges[1:]):
        n = math.ceil((b - a) / max_frames)
        for k in range(n):
            f0 = a + (b - a) * k // n
            f1 = a + (b - a) * (k + 1) // n
            shots.append({"f0": f0, "frames": f1 - f0, "joined": k > 0})
    for i, s in enumerate(shots):
        s["id"] = f"s{i + 1:03d}"
        s["gen_frames"] = gen_frames(s["frames"])
    return shots


def gen_frames(frames: int) -> int:
    """Frames to ask H3 for: at least PAD_MIN more than the shot, rounded up to 17k+5 (what the workflow can make)."""
    n = frames + config.PAD_MIN
    return n + (5 - n % 17) % 17


def target_size(width: int, height: int) -> tuple[int, int]:
    """About 0.8 MP in the source's aspect, sides multiples of 32 (16:9 -> 1184x672, 9:16 -> 672x1184)."""
    ar = width / height
    w = round(math.sqrt(0.8e6 * ar) / 32) * 32
    h = round(w / ar / 32) * 32
    return w, h


def cut_clip(mv: Path, audio: Path | None, shot: dict, size: tuple[int, int], out: Path):
    """The shot's frames, held on its last frame up to gen_frames, scaled to the generation size, with its sound."""
    a, F, pad = shot["f0"], shot["frames"], shot["gen_frames"] - shot["frames"]
    vf = (f"select='between(n,{a},{a + F - 1})',setpts=N/{config.FPS}/TB,tpad=stop_mode=clone:stop={pad},"
          f"scale={size[0]}:{size[1]}:flags=lanczos,setsar=1")
    args = ["-i", str(mv)]
    if audio:
        args += ["-ss", f"{a / config.FPS:.4f}", "-i", str(audio)]
    args += ["-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]"]
    args += ["-map", "1:a", "-af", "apad", "-c:a", "aac"] if audio else []
    args += ["-r", str(config.FPS), "-t", f"{shot['gen_frames'] / config.FPS:.4f}", "-c:v", "libx264", "-crf", "12", "-preset", "fast", str(out)]
    ff(*args)


def keyframes(video: Path, idxs: list[int], out_dir: Path, tag: str, width: int = 512) -> list[Path]:
    out = []
    for i in idxs:
        p = out_dir / f"{tag}-{i}.jpg"
        ff("-i", str(video), "-vf", f"select=eq(n\\,{i}),scale={width}:-2", "-vsync", "0", "-frames:v", "1", "-q:v", "3", str(p))
        if p.exists():
            out.append(p)
    return out


def raw_frames(video: Path, n: int, w: int = 96, h: int = 54, start: int = 0):
    """Small RGB frames as a numpy array (n, h, w, 3) for colour checks."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='gte(n,{start})',scale={w}:{h}", "-vsync", "0",
                          "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
