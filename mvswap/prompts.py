"""Prompts for Nick's swap workflow, as worked out on a 97-second MV remake.

What mattered: say only who becomes whom, that framing follows the original frame by frame, and that the set, lighting
and colours come from the original video (else the references' plain backgrounds leak in). Listing props or graphics
made the model add stickers and widen close-ups; asking for "every cut" made it invent cuts. A shot with one person gets
only that person's picture, or the other one tends to walk in. Picture numbers count images only: <Picture 1>, <Picture 2>.
"""

FRAMING = ("The camera, framing and shot size come from <Video 1>, frame by frame: a close-up stays the same close-up, a face that "
           "fills the frame still fills the frame, a medium shot stays a medium shot, a wide shot stays the same wide shot.")
SET = "The backdrop, set, lighting and colours are the same as in <Video 1>."
TEXT = " Any words already written on screen in <Video 1> stay exactly as they are."
KEEP = ("Maintain the original video's scene, camera movement, composition, lighting, timing, movements, posture, eye direction, "
        "facial expressions and overall performance unchanged.")


def _noun(c):
    return "young man" if c.get("gender") == "man" else "young woman"


def _pron(c):
    return ("he", "him", "his") if c.get("gender") == "man" else ("she", "her", "her")


OTHERS = "Everyone else in <Video 1> stays exactly as they are, in the same place."


def _style(style: dict | None) -> tuple[str, str]:
    """(sentence to add, how the result was made) — animation keeps its own look and the new characters are drawn in it."""
    if style and style.get("kind") == "animation":
        desc = style.get("desc") or "an animated film"
        return (f" <Video 1> is {desc}; the result keeps exactly this animation style, rendering, lighting and colours, and the new "
                "characters are drawn in that same style, as their pictures show them.", "animated")
    return "", "filmed"


def build(cast: list[dict], people: list[str], text_on_screen: bool, second_pass: bool = False, others: bool = False,
          style: dict | None = None, backdrop: str = "") -> tuple[str, list[str]]:
    """(prompt, picture paths in <Picture N> order) for one shot. `second_pass`: the video is our own half-done take,
    so the people already swapped must stay; `others`: people we don't replace (fans, a villain) share the shot.
    Either way there is no "only person in this shot"."""
    by = {c["id"]: c for c in cast}
    who = [by[p] for p in people]
    text = TEXT if text_on_screen else ""
    style_line, made = _style(style)
    bg = (f"The backdrop is {backdrop}, exactly as in <Video 1>, with the same lighting and colours; only the people come from the pictures, "
          "never the plain background behind them." if backdrop else SET)
    keep_rest = second_pass or others
    if len(who) == 1:
        c = who[0]
        he, him, his = _pron(c)
        prompt = "\n\n".join([
            f"Replace {c['src_desc']} in <Video 1> with the {_noun(c)} in <Picture 1>. "
            + (OTHERS if keep_rest else f"{he.capitalize()} is the only person in this shot."),
            f"Use <Picture 1> as the identity and appearance reference: {c['desc']}. Keep {his} face, hairstyle, outfit "
            f"and body proportions consistent with the picture.",
            FRAMING + " " + bg + text + style_line,
            f"{KEEP} {he.capitalize()} naturally performs exactly the same movements as the person {he} replaces, in the same place in the frame. "
            f"The result looks as if the original video had been {made} with {him}.",
        ])
        return prompt, [c["picture"]]
    maps = " ".join(f"{c['src_desc'][0].upper() + c['src_desc'][1:]} in <Video 1> becomes the {_noun(c)} in <Picture {i}>."
                    for i, c in enumerate(who, start=1))
    refs = " ".join(f"Use <Picture {i}> as the identity and appearance reference for the {_noun(c)}: {c['desc']}."
                    for i, c in enumerate(who, start=1))
    prompt = "\n\n".join([
        f"Replace the people in <Video 1> with the people in the reference pictures. {maps}",
        "Only these people come from the pictures. " + FRAMING + " "
        + (OTHERS if keep_rest else "Whoever is in the frame in <Video 1> is the only one in the frame.") + " " + bg + text + style_line,
        refs + " Keep their faces, hairstyles, outfits and body proportions consistent with the pictures.",
        f"{KEEP} Each replacement naturally performs exactly the same movements as the person they replace, in the same place in the frame. "
        f"The result looks as if the original video had been {made} with these people.",
    ])
    return prompt, [c["picture"] for c in who]
