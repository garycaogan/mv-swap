"""Shot-sized identity references and timestamped, synchronized review evidence."""
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from . import config, media

SIZES = {"close-up": 0, "medium": 1, "three-quarter": 2, "full": 3}


def visible_description(description: str, target: str) -> str:
    """Keep identity details (hair/eyes) while removing off-screen clothing clauses."""
    if target == "full":
        return description
    hidden = r"\b(?:shoes?|boots?|loafers?|heels?|footwear)\b"
    if target in ("close-up", "medium"):
        hidden = r"\b(?:mini\s*skirt|mini\s*skirts|skirts?|stockings?|socks?|trousers?|pants?|shorts?|leggings?|shoes?|boots?|loafers?|heels?|footwear)\b"
    clauses = re.split(r",|;|\band\b", description, flags=re.I)
    return ", ".join(s.strip(" .") for s in clauses if s.strip(" .") and not re.search(hidden, s, re.I))


def shot_size(value: str) -> str:
    value = str(value).strip().lower()
    return {"特写": "close-up", "半身": "medium", "七分身": "three-quarter", "全身": "full",
            "wide": "full", "waist-up": "medium", "full-body": "full"}.get(value, value)


def reference(cast: dict, geometry: dict, out: Path) -> dict:
    """Select a compatible existing crop, else crop the primary without rescaling pixels."""
    target = shot_size(geometry.get("framing", ""))
    if target not in SIZES:
        raise ValueError(f"无法判断 {cast['id']} 的景别，不能提交 H3")
    refs = cast.get("references") or []
    suitable = [r for r in refs if r.get("compatible") is True and shot_size(r.get("framing", "")) == target]
    if suitable:
        r = suitable[0]
        return {**r, "framing": target, "method": "existing", "desc": visible_description(r.get("desc") or cast["desc"], target)}
    if not refs:
        raise ValueError("参考图没有人物框和腰线分析，不能盲裁")
    r = refs[0]
    original = shot_size(r.get("framing", ""))
    if original not in SIZES or SIZES[original] < SIZES[target]:
        raise ValueError("主参考图的取景比原片更窄，不能凭空补全身体")
    box = r.get("bbox")
    landmark = r.get({"close-up": "chin_y", "medium": "waist_y", "three-quarter": "knee_y"}.get(target, ""))
    if not box or len(box) != 4 or landmark is None:
        raise ValueError("参考图缺少人物框或裁切部位坐标")
    with Image.open(r["picture"]) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        x0, y0, x1, y1 = map(float, box)
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and y0 < float(landmark) <= y1):
            raise ValueError("参考图人物框或裁切部位坐标无效")
        # A little headroom and side room; never include knees/stockings in a waist crop.
        top = max(0, math.floor((y0 - .04 * (y1 - y0)) * h))
        bottom = min(h, math.ceil(float(landmark) * h))
        if target == "close-up":
            bottom = min(h, math.ceil((float(landmark) + .08 * (y1 - y0)) * h))
        left, right = max(0, math.floor((x0 - .03) * w)), min(w, math.ceil((x1 + .03) * w))
        if bottom <= top or right <= left:
            raise ValueError("参考图裁切区域为空")
        out.parent.mkdir(parents=True, exist_ok=True)
        im.crop((left, top, right, bottom)).save(out, quality=95)
    # The full-body description would invite the model to reveal off-screen garments again.
    desc = visible_description(r.get("desc") or cast["desc"], target)
    desc = desc or "the same face, hair and visible upper-body clothing as the cropped reference"
    return {"picture": str(out), "source": r.get("source", r["picture"]), "framing": target,
            "method": "crop", "crop_box": [left, top, right, bottom], "pixels": [right-left, bottom-top], "desc": desc}


def sample_indices(shot: dict) -> list[int]:
    """Every global integer second plus both segment edges; offsets are exact 24fps frame indices."""
    start, n = shot["f0"], shot["frames"]
    return sorted({0, n - 1, *[i - start for i in range(start, start + n) if i % config.FPS == 0]})


def pairs(source: Path, take: Path, shot: dict, directory: Path, tag: str, width: int = 384) -> tuple[list[Path], list[float]]:
    directory.mkdir(parents=True, exist_ok=True)
    idx = sample_indices(shot)
    src = media.keyframes(source, [shot["f0"] + i for i in idx], directory, tag + "-src", width)
    dst = media.keyframes(take, idx, directory, tag + "-out", width)
    if len(src) != len(idx) or len(dst) != len(idx):
        raise ValueError("逐秒对照缺帧，不能通过审片")
    paths, seconds = [], []
    for i, a, b in zip(idx, src, dst):
        sec = (shot["f0"] + i) / config.FPS
        with Image.open(a) as left, Image.open(b) as right:
            right = ImageOps.pad(right, left.size)
            panel = Image.new("RGB", (left.width * 2, left.height + 24), "white")
            panel.paste(left, (0, 24)); panel.paste(right, (left.width, 24))
            ImageDraw.Draw(panel).text((4, 5), f"{sec:.3f}s | LEFT SOURCE | RIGHT REMAKE", fill="black")
            path = directory / f"{tag}-pair-{i}.jpg"
            panel.save(path, quality=90)
        paths.append(path); seconds.append(round(sec, 3))
    return paths, seconds


def contact_sheet(paths: list[Path], out: Path, cols: int = 5):
    thumbs = []
    for path in paths:
        with Image.open(path) as im:
            thumbs.append(ImageOps.contain(im, (384, 380)).copy())
    w, h = max(im.width for im in thumbs), max(im.height for im in thumbs)
    cols = min(cols, len(thumbs))
    sheet = Image.new("RGB", (cols * w, math.ceil(len(thumbs) / cols) * h), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out, quality=92)
