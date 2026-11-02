"""Paid, checkpointed reference edits. Neutral edits never receive the source actor's photo."""
from pathlib import Path

from . import ai, framing, media

MODES = ("crop", "neutral", "keyframe")


def edit_prompt(mode: str, character: dict, shot: dict, geometry: dict) -> str:
    size = framing.shot_size(geometry.get("framing", shot.get("framing", "medium")))
    crop = {"close-up": "head and shoulders", "medium": "head to waistline, waist-up only",
            "three-quarter": "head to knees", "full": "head to feet"}.get(size)
    if not crop:
        raise ValueError("参考图取景不明确，不能生图")
    appearance = framing.visible_description(character["desc"], size)
    if mode == "neutral":
        top = geometry.get("head_top_y", geometry.get("bbox", [.1, .18, .9, 1])[1])
        return ("Edit the sole input image. Repaint EXACTLY THE SAME PERSON, preserving the same facial identity, face shape, "
                "hair color and hairstyle, all hair ornaments, eye color, and the same outfit's visible upper garments. "
                f"Identity details: {appearance}. Frame {crop}. Leave about {float(top)*100:.0f}% of the canvas above the hair, "
                "with natural body proportions. Front-facing, looking directly at camera. Both shoulders level. "
                "BOTH ARMS hang naturally straight DOWN at the sides of the torso, relaxed and separate from the torso. "
                "Hands naturally hang below the waist; when the framing ends at the waist, hands remain OUTSIDE the image. "
                "NO hand touching hair, NO crossed arms, NO hands in front of chest or waist, NO gestures, NO bent-leg pose. "
                "Keep exactly two anatomical arms; if fingers are visible, exactly five normal fingers per hand. "
                "Wear the same outfit naturally. Simple unobtrusive plain background with soft even light, "
                "in the same rendering style as the input image. Show only garments visible within this framing, no skirt hem, stockings "
                "or shoes in a waist-up portrait. Do not copy the input pose or its wider framing.")
    if mode == "keyframe":
        return ("Image 1 is the exact first frame of the source shot. Image 2 is the approved replacement character. "
                f"Edit image 1: replace ONLY {character['src_desc']}; every other person stays unchanged. Use the EXACT facial identity, hairstyle, hair ornaments, eye color "
                f"and visible clothing from image 2 ({appearance}). Keep image 1's exact framing ({crop}), subject scale, "
                "headroom, camera angle, background, lighting, body pose, head/torso direction and every arm/hand position. "
                "Replace the source facial identity completely. Take POSE from image 1, IDENTITY from image 2. "
                "Do not import image 2's hair-touching hand, crossed arms, bent-leg pose or background. "
                "Exactly two anatomical arms, no additional hands or fingers, no text or collage.")
    raise ValueError(mode)


def inspect(mode: str, primary: Path, edited: Path, target: str, first: Path | None = None, geometry: dict | None = None) -> dict:
    images = [primary, edited] if mode == "neutral" else [first, primary, edited]
    required = ["same_identity", "hair_kept", "ornaments_kept", "eyes_kept", "outfit_kept", "framing_matches",
                "headroom_matches", "anatomy_valid"]
    required += ["front_facing", "arms_at_sides"] if mode == "neutral" else ["pose_matches_first", "camera_matches_first", "background_matches_first"]
    prompt = (f"Inspect the LAST image against the approved character ({'first' if mode == 'neutral' else 'second'} image). "
              f"Required framing: {target}. Check the same facial identity, hairstyle and hair ornaments, eye color, "
              "same visible outfit, normal anatomy with exactly two arms and no extra hands/fingers. "
              "For waist-up framing hands may legitimately be below/outside the image; that is anatomy_valid. "
              + (f"Headroom target is {(geometry or {}).get('head_top_y', .18)} of image height above the hair; "
                 "do not copy the primary image's wider framing or headroom. It must face forward, both arms hanging DOWN "
                 "at the sides, no hand touching hair, no crossing arms, no gestures. "
                 if mode == "neutral" else "The FIRST image supplies the exact pose, headroom, camera and background; these must match. ")
              + "Return JSON with each of these keys as a strict true/false: " + ", ".join(required)
              + ', plus "issue": "specific problem or empty", "desc": "the edited person identity and visible clothing only". '
              "A different expression is fine if the identity is preserved. Mark uncertain checks false.")
    previews = [ai.shrink(Path(p), edited.with_name(edited.stem + f"-qa{i}.jpg")) for i, p in enumerate(images)]
    reply = ai.ask_json(prompt, previews)
    return {**reply, "pass": all(reply.get(k) is True for k in required), "required": required}


def prepare(job, shot: dict, character: dict, geometry: dict, mode: str) -> dict:
    (job.dir / "refs").mkdir(exist_ok=True)
    primary = Path(character["picture"])
    target = framing.shot_size(geometry.get("framing", shot.get("framing", "medium")))
    history = shot.setdefault("reference_attempts", {}).setdefault(character["id"], [])
    for attempt in history:
        if attempt.get("review", {}).get("pass") and attempt.get("manual_pass") is not False:
            return _result(attempt, character, target, mode)
    while True:
        pending = next((a for a in history if a.get("status") != "done" or "review" not in a), None)
        if not pending:
            if len(history) >= 2:
                raise RuntimeError("参考图两次都不合格，停止生成 H3")
            pending = {"n": len(history)+1, "mode": mode, "status": "pending"}
            pending["file"] = str(job.dir / "refs" / f"{shot['id']}-{character['id']}-{mode}-{pending['n']}.png")
            pending["prompt"] = edit_prompt(mode, character, shot, geometry)
            if history:
                issue = history[-1].get("review", {}).get("issue") or history[-1].get("manual_issue")
                if issue:
                    pending["prompt"] += f" The previous edit failed inspection: {issue}. Correct this explicitly while preserving all identity and pose requirements."
            pending["pictures"] = [str(primary)]
            if mode == "keyframe":
                first = media.keyframes(Path(job.state["prepared"]["mv"]), [shot["f0"]], job.dir / "refs",
                                        shot["id"] + "-first", width=job.state["prepared"]["width"])[0]
                pending["first"] = str(first)
                pending["pictures"] = [str(first), str(primary)]
            history.append(pending)
            job.save()
        if pending.get("status") == "failed" or pending.get("error"):
            raise RuntimeError(pending.get("error") or "参考图编辑失败；保留原文，停止，不改写提示词绕过")
        try:
            if pending.get("status") != "done":
                w, h = job.state["prepared"]["width"], job.state["prepared"]["height"]
                aspects = {"9:16":9/16, "16:9":16/9, "1:1":1, "3:4":3/4, "4:3":4/3}
                aspect = min(aspects, key=lambda k: abs(aspects[k]-w/h))
                cny = ai.image(Path(pending["file"]), pending["prompt"], [Path(p) for p in pending["pictures"]],
                               size=aspect, state=pending, checkpoint=job.save)
                pending.update(status="done", cny=cny)
                job.save()
            if "review" not in pending:
                pending["review"] = inspect(mode, primary, Path(pending["file"]), target,
                                            Path(pending["first"]) if pending.get("first") else None, geometry)
                job.save()
            if pending["review"]["pass"]:
                return _result(pending, character, target, mode)
        finally:
            job.save()


def _result(attempt: dict, character: dict, target: str, mode: str) -> dict:
    return {"picture": attempt["file"], "source": character["picture"], "framing": target, "method": mode,
            "edit_attempt": attempt["n"], "desc": framing.visible_description(attempt["review"].get("desc") or character["desc"], target)}
