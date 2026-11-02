"""Who replaces whom: `--cast "原片里的人=图片路径"`, one picture of each new person (full body, the outfit they wear in the film)."""
import shutil
from pathlib import Path

from . import ai


def parse(specs: list[str], refs_dir: Path) -> list[dict]:
    refs_dir.mkdir(parents=True, exist_ok=True)
    cast = []
    for i, spec in enumerate(specs, start=1):
        if "=" not in spec:
            raise SystemExit(f"--cast 要写成「原片里的人=图片路径」，这条是：{spec}")
        hint, who = (s.strip() for s in spec.split("=", 1))
        pic = Path(who).expanduser()
        if not pic.is_file():
            raise SystemExit(f"找不到角色图「{who}」：--cast 右边写一张图片的路径（全身、穿着成片里的衣服）")
        dst = refs_dir / f"P{i}{pic.suffix.lower() or '.jpg'}"
        shutil.copyfile(pic, dst)
        cast.append({"id": f"P{i}", "hint": hint, "name": pic.stem, "picture": str(dst), "references": [{"picture": str(dst), "source": str(pic)}]})
    return cast


def describe(cast: list[dict]):
    """English look and gender of each replacement, read from its own picture (prompts work best in English)."""
    for c in cast:
        if c.get("desc"):
            continue
        ans = ai.ask_json(
            "Describe the person in this picture for a video-generation prompt. Return JSON "
            '{"gender": "woman" or "man", "desc": "one sentence: hair (colour, length, style, accessories), eye colour, '
            'every visible garment with its colour, shoes; no background, no pose, no mood", '
            '"references": [{"index": 0, "compatible": true, "framing": "close-up|medium|three-quarter|full", '
            '"bbox": [0.1,0.02,0.9,0.98], "chin_y": 0.2, "waist_y": 0.5, "knee_y": 0.8, '
            '"desc": "only visible hair, face and garments"}]}. '
            'Image 0 is the approved primary identity/outfit. Analyze EVERY image in order (zero-based index). '
            'compatible is true ONLY for the same face identity AND same outfit as image 0. '
            'framing means close-up=head/shoulders, medium=head to waist, three-quarter=head to knees, full=feet visible. '
            'bbox is the visible person including hair in normalized [left,top,right,bottom]; landmarks are normalized image y coordinates. '
            'Use null for a landmark outside the image. Include generous hair/head space in bbox.',
            [Path(r["picture"]) for r in c.get("references", [{"picture": c["picture"]}])])
        c["gender"] = ans.get("gender", "woman")
        c["desc"] = ans.get("desc", "").strip().rstrip(".")
        for ref, analysis in zip(c.get("references", []), ans.get("references") or []):
            if analysis.get("index") == c["references"].index(ref):
                ref.update({k: analysis[k] for k in ("compatible", "framing", "bbox", "chin_y", "waist_y", "knee_y", "desc") if k in analysis})
