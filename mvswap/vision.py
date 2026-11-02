"""The vision model's three jobs: describe the original performers, label every shot, review every take."""
import json
from pathlib import Path

from . import ai
from .framing import SIZES, shot_size


def performers(cast: list[dict], frames: list[Path]):
    """Turn the user's Chinese hints (「金发女生」) into how each original performer looks, in English."""
    hints = "\n".join(f'{c["id"]}: {c["hint"]}' for c in cast)
    ans = ai.ask_json(
        "These are frames from a music video. The user named the people to be replaced:\n" + hints + "\n\n"
        "For each id, describe how that person looks in these frames so another model can find them in any shot: "
        "gender, hair, the most distinctive clothes or accessories. Return JSON "
        '{"P1": {"gender": "woman" or "man", "desc": "the blonde woman with short wavy hair in a black leather jacket"}, ..., '
        '"style": {"kind": "live" or "animation", "desc": "e.g. a live-action music video, or a stylized 3D animated film with '
        'smooth shading and glossy hair"}}. Each desc is a short noun phrase starting with \'the\' (style desc starts with \'a\').', frames)
    style = ans.get("style") or {}
    for c in cast:
        a = ans.get(c["id"]) or {}
        c["src_gender"] = a.get("gender", "")
        c["src_desc"] = (a.get("desc") or c["hint"]).strip().rstrip(".")
    return {"kind": "animation" if style.get("kind") == "animation" else "live", "desc": (style.get("desc") or "").strip().rstrip(".")}


def label_shot(cast: list[dict], frames: list[Path], style: dict | None = None) -> dict:
    """kind swap / keep / graphic, who is visible, shot size, words on screen — from three frames of one shot. The
    video's own look counts as normal footage: in an animated film the animated characters are "swap", not "graphic"
    (without this line, all 40 character shots of an animated MV were labelled graphic)."""
    who = "\n".join(f'{c["id"]}: {c["src_desc"]}' for c in cast)
    look = (style or {}).get("desc") or "a live-action music video"
    ans = ai.ask_json(
        f"These are three frames (start, middle, end) of ONE shot from {look}. The people to be replaced are:\n" + who + "\n\n"
        "The video's own look is the normal footage: characters shown in that same look (live actors in live action, the "
        "animated characters in an animated film) count as filmed.\n\n"
        "Answer in JSON:\n"
        '{"people": ["P1", ...] (ids of those people who appear and could be recognised by face, hair or outfit in any of the frames; [] if none),\n'
        ' "kind": "swap" if any listed person appears in the video\'s own look and is recognisable; "graphic" only if they appear as a '
        'different, flat kind of picture laid into the video (cut-out photo stickers, flat 2D drawings over the scene, a photo on a card); '
        '"keep" if nobody listed is recognisable (empty set, objects, title cards, '
        'silhouettes, an extreme close-up of only lips or an eye),\n'
        ' "framing": "close-up" | "medium" | "three-quarter" | "full",\n'
        ' "geometry": {"P1": {"framing": "medium", "bbox": [0.1,0.05,0.9,1.0], "head_top_y": 0.18}},\n'
        ' "text_on_screen": true/false (words or handwriting drawn over the picture),\n'
        ' "others": true/false (people not in the list are clearly visible),\n'
        ' "backdrop": "the setting behind the people and its lighting in a few words, e.g. a dark palace hall lit by a golden beam from above",\n'
        ' "note": "a few words on what happens"}. '
        'For EACH listed person, geometry describes their visible crop, not their pose: '
        'close-up=head/shoulders, medium=head to waist, three-quarter=head to knees, full=feet visible. '
        'bbox is normalized [left,top,right,bottom] including the hair. head_top_y is the normalized top of the HAIR, '
        'not raised hands (median across these frames). If framing varies, use the widest visible crop.', frames)
    people = [p for p in ans.get("people") or [] if any(c["id"] == p for c in cast)]
    kind = ans.get("kind") if ans.get("kind") in ("swap", "keep", "graphic") else "swap"
    if kind == "swap" and not people:
        kind = "keep"
    return {"kind": kind, "people": people, "framing": ans.get("framing", ""), "text": bool(ans.get("text_on_screen")),
            "geometry": ans.get("geometry") or {},
            "others": bool(ans.get("others")), "backdrop": (ans.get("backdrop") or "").strip().rstrip("."), "note": ans.get("note", "")}


def backdrop(frames: list[Path]) -> str:
    """Just the setting and lighting of a shot, for shots labelled before labels carried it."""
    ans = ai.ask_json('Three frames of one shot. Return JSON {"backdrop": "the setting behind the people and its lighting in a few words, '
                       'e.g. a dark palace hall lit by a golden beam from above"}.', frames)
    return (ans.get("backdrop") or "").strip().rstrip(".")


def review_take(cast: list[dict], people: list[str], src_frames: list[Path], out_frames: list[Path], seconds: list[float] | None = None) -> dict:
    """Compare the take with the original at the same moments; a take passes only when every check is clean."""
    by = {c["id"]: c for c in cast}
    mapping = "\n".join(f'- {p}: {by[p]["src_desc"]} must become: {by[p]["desc"]}' for p in people)
    n = len(src_frames)
    paired = seconds is not None
    seconds = seconds if paired else list(range(n))
    evidence = (f"Images 1-{n} are synchronized pairs, LEFT=original, RIGHT=remake; timestamps (seconds from the prepared source): {seconds}. "
                if paired else f"Images 1-{n} are originals; images {n+1}-{2*n} are remakes at the same moments. ")
    ans = ai.ask_json(
        evidence + f"In the remake:\n{mapping}\nEverything else (set, camera, framing, lighting, colours, movement) should match the original.\n\n"
        f"Return exactly {len(seconds)} per_second rows with unique zero-based sample IDs 0 through {len(seconds)-1}, "
        f"in image order; copy these timestamps exactly (never round): {seconds}. "
        "HARD CHECK at EVERY timestamp, for EVERY listed person: classify source_size and remake_size into exactly "
        "close-up (head/shoulders), medium (head to waist), three-quarter (head to knees), full (feet visible). "
        "A one-category change in EITHER direction fails, even briefly. camera_match checks angle, subject scale and position; "
        "pose_match checks head/body direction, torso lean, arm/hand/leg positions at that SAME timestamp, not just similar dance style. "
        "reference_pose=true if the performer freezes into a reference-like standing pose instead of following the original action. "
        "newly_visible_lower_body=true if skirt hem, thighs, stockings or shoes appear where the original crop excluded those body regions. "
        "Judge body regions, not differences in garment design. ANY false camera_match/pose_match, size mismatch, reference_pose or "
        "newly_visible_lower_body is a FAIL irrespective of total score. Never excuse it as slight widening or a high quality image. "
        "Use unknown when not assessable; never omit a timestamp.\n"
        "Check the remake against the original and return JSON:\n"
        '{"per_second": [{"sample": 0, "second": 0, "source_size": "medium", "remake_size": "medium", '
        '"source_pose": "head/torso direction and left/right arm/hand positions", "remake_pose": "describe independently", "camera_match": true, '
        '"pose_match": true, "reference_pose": false, "newly_visible_lower_body": false, "reason": "brief observation"}],\n'
        ' "framing": 0-2 (2 = same shot size and composition, 1 = close, 0 = different shot),\n'
        ' "right_people": true/false (each original person became the right new person),\n'
        ' "missing": [ids from the list above whose person was NOT turned into their new person in the remake],\n'
        ' "originals_left": true/false (an original person is still visible unchanged),\n'
        ' "extra_people": true/false (someone appears who is not in the original frame),\n'
        ' "added_overlays": true/false (text, stickers, sparkles or graphics that the original does not have),\n'
        ' "style_kept": true/false (the remake keeps the original\'s look: live action stays live action, animation stays the same animation style),\n'
        ' "broken": true/false (melted faces, wrong number of limbs, a still reference photo instead of the scene),\n'
        ' "score": 0-10 (how good a stand-in for the original this is),\n'
        ' "issue": "one short sentence on the worst problem, empty if none"}', src_frames + out_frames, max_tokens=7000)
    ok = (ans.get("right_people") is True and not ans.get("originals_left") and not ans.get("extra_people")
          and not ans.get("added_overlays") and not ans.get("broken") and ans.get("style_kept") is not False
          and int(ans.get("framing") or 0) >= 1 and float(ans.get("score") or 0) >= 6)
    missing = [p for p in ans.get("missing") or [] if p in people]
    checks = ans.get("per_second") or []
    indexed = any(isinstance(c, dict) and "sample" in c for c in checks)
    drift, incomplete = [], []
    for sample, second in enumerate(seconds):
        matches = [c for c in checks if isinstance(c, dict) and
                   (type(c.get("sample")) is int and c["sample"] == sample if indexed else c.get("second") == second)]
        if len(matches) != 1:
            incomplete.append(second)
            continue
        c = matches[0]
        c["second"] = second  # Image/sample identity is authoritative, never the model's rounded timestamp.
        source_size, remake_size = shot_size(c.get("source_size", "")), shot_size(c.get("remake_size", ""))
        if source_size not in SIZES or remake_size not in SIZES or any(type(c.get(k)) is not bool for k in
                ("camera_match", "pose_match", "reference_pose", "newly_visible_lower_body")):
            incomplete.append(second)
        elif source_size != remake_size or c["camera_match"] is False or c["pose_match"] is False or c["reference_pose"] or c["newly_visible_lower_body"]:
            drift.append(second)
    issue = ans.get("issue") or ""
    if drift:
        issue = "取景/姿势漂到参考图（第 " + "、".join(f"{s:g}" for s in drift) + " 秒）" + ("；" + issue if issue else "")
    if incomplete:
        issue = "逐秒取景/姿势核对不完整（第 " + "、".join(f"{s:g}" for s in incomplete) + " 秒）；" + issue
    return {**{k: ans.get(k) for k in ("framing", "right_people", "originals_left", "extra_people", "added_overlays", "broken", "style_kept", "score", "issue")},
            "issue": issue, "per_second": checks, "drift_seconds": drift, "unchecked_seconds": incomplete,
            "rule": "per-second-v1", "missing": missing, "pass": ok and not missing and not drift and not incomplete}


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)
