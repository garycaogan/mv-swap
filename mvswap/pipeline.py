"""One command from source video to finished swap film; every stage is saved to state.json and resumes from there."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import cast as cast_mod
from . import ai, config, finish, media, prompts, rh, vision


class Job:
    def __init__(self, path: Path):
        self.dir = path.resolve()                               # ffmpeg concat lists resolve relative paths against the list file
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self._vision_base = self.state.get("vision_cny", 0.0)   # earlier runs of this job; this run's spend is added on save

    def save(self):
        self.state["vision_cny"] = round(self._vision_base + ai.vision_spent(), 3)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1))
        tmp.replace(self.state_file)

    def log(self, event: str, **f):
        line = {"at": time.strftime("%H:%M:%S"), "event": event, **f}
        with (self.dir / "ledger.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(line["at"], event, " ".join(f"{k}={v}" for k, v in f.items()), flush=True)


# ------------------------------------------------------------------ stages

def stage_prepare(job: Job, source: str, start: float, end: float | None, allow_lowres: bool, crop: str = "auto"):
    if "prepared" in job.state:
        return
    src_dir = job.dir / "src"
    src_dir.mkdir(exist_ok=True)
    src = media.fetch_source(source, src_dir)
    info = media.probe(src)
    short = min(info["width"], info["height"])
    job.log("source", width=info["width"], height=info["height"], duration=round(info["duration"], 1))
    if short < config.MIN_SOURCE_SHORT_SIDE and not allow_lowres:
        raise SystemExit(f"原片只有 {info['width']}×{info['height']}，低于 {config.MIN_SOURCE_SHORT_SIDE}p；换高清源，"
                         "或者确认就用这个画质时加 --allow-lowres")
    job.state["prepared"] = media.prepare(src, start, end, src_dir, crop)
    if job.state["prepared"].get("crop"):
        job.log("crop", box=":".join(map(str, job.state["prepared"]["crop"])))
    job.state["t_start"] = time.time()
    job.save()


def stage_shots(job: Job):
    if "shots" in job.state:
        return
    p = job.state["prepared"]
    scores = media.scene_scores(Path(p["mv"]))
    sims = media.structure(Path(p["mv"]))
    max_frames = config.MAX_GEN_FRAMES - config.PAD_MIN - 16       # so gen_frames never passes MAX_GEN_FRAMES
    job.state["shots"] = media.find_shots(scores, p["frames"], max_frames, sims)
    job.log("shots", count=len(job.state["shots"]))
    job.save()


def _keys(job: Job, shot: dict, video: Path, offset: int, tag: str) -> list[Path]:
    F = shot["frames"]
    idx = sorted({min(F - 1, max(0, round(F * x))) for x in (0.15, 0.5, 0.85)})
    d = job.dir / "frames"
    d.mkdir(exist_ok=True)
    return media.keyframes(video, [offset + i for i in idx], d, f"{shot['id']}-{tag}")


def stage_cast(job: Job, specs: list[str]):
    if job.state.get("cast") and all(c.get("src_desc") for c in job.state["cast"]):
        return
    cast = cast_mod.parse(specs, job.dir / "refs")
    cast_mod.describe(cast)
    shots = job.state["shots"]
    pick = [shots[int(i * (len(shots) - 1) / 7)] for i in range(min(8, len(shots)))]
    frames = []
    for s in pick:
        ks = _keys(job, s, Path(job.state["prepared"]["mv"]), s["f0"], "src")
        frames.append(ks[len(ks) // 2])
    job.state["style"] = vision.performers(cast, frames)
    job.log("style", kind=job.state["style"]["kind"], desc=job.state["style"]["desc"])
    job.state["cast"] = cast
    for c in cast:
        job.log("cast", id=c["id"], source=c["src_desc"], becomes=c["name"], gender=f"{c.get('src_gender')}->{c.get('gender')}")
    job.save()


def stage_label(job: Job):
    todo = [s for s in job.state["shots"] if "kind" not in s]
    if not todo:
        return
    mv = Path(job.state["prepared"]["mv"])

    def one(s):
        return s, vision.label_shot(job.state["cast"], _keys(job, s, mv, s["f0"], "src"), job.state.get("style"))
    with ThreadPoolExecutor(10) as pool:
        for s, lab in pool.map(one, todo):
            s.update(lab)
            job.log("label", shot=s["id"], kind=s["kind"], people=",".join(s["people"]), framing=s["framing"])
            job.save()


def stage_backdrops(job: Job):
    """Shots labelled before labels named the setting get it now (one small vision call each)."""
    mv = Path(job.state["prepared"]["mv"])
    todo = [s for s in job.state["shots"] if s.get("kind") == "swap" and "backdrop" not in s]
    with ThreadPoolExecutor(10) as pool:
        for s, bg in pool.map(lambda s: (s, vision.backdrop(_keys(job, s, mv, s["f0"], "src"))), todo):
            s["backdrop"] = bg
            job.save()


def stage_graphic(job: Job, allow_many: bool = False):
    shots = job.state["shots"]
    n = sum(1 for s in shots if s["kind"] == "graphic")
    if n > max(3, len(shots) // 5) and not allow_many:          # every graphic shot is a paid redraw; this many usually means mislabels
        raise SystemExit(f"{n}/{len(shots)} 个镜头被标成图形镜头（每个要单独生图，约 ¥0.5），多半是标错了。看一下 state.json 里这些镜头，"
                         "确认真是贴纸、平面插画这类画面再加 --allow-many-graphic 重跑")
    todo = [s for s in shots if s["kind"] == "graphic" and not s.get("graphic")]
    with ThreadPoolExecutor(4) as pool:                          # image redraws take ~40 s each; run them side by side
        for s, g in pool.map(lambda s: (s, finish.graphic_shot(s, job.state["cast"], job.state["prepared"], job.dir)), todo):
            s["graphic"] = g
            job.log("graphic", shot=s["id"], cny=g["cny"])
            job.save()


def _backdrop_lost(job: Job, shot: dict, take: Path) -> int:
    """Frames whose outer ring (mostly set and backdrop) drifts far from the original's: the references' plain
    backgrounds leaking in, or a still reference photo replacing the scene."""
    import numpy as np
    F = shot["frames"]
    src = media.raw_frames(Path(job.state["prepared"]["mv"]), F, start=shot["f0"]).astype(np.int16)
    out = media.raw_frames(take, F).astype(np.int16)
    n = min(len(src), len(out))
    if not n:
        return F
    ring = np.ones(src.shape[1:3], bool)
    ring[8:-8, 14:-14] = False
    # median over the ring: new clothes on part of the edge (a white hoodie where a black cardigan was) don't count,
    # a backdrop that changed all round does
    diff = np.median(np.abs(src[:n][:, ring] - out[:n][:, ring]).mean(axis=2), axis=1)
    return int((diff > 45).sum())


def stage_generate(job: Job, max_attempts: int):
    p = job.state["prepared"]
    size = media.target_size(p["width"], p["height"])
    mv, audio = Path(p["mv"]), Path(p["audio"]) if p.get("audio") else None
    (job.dir / "shots").mkdir(exist_ok=True)
    (job.dir / "takes").mkdir(exist_ok=True)
    sched = None
    stage_review(job)                                           # a resume may hold finished takes nobody reviewed yet
    for _ in range(max_attempts):
        todo = [s for s in job.state["shots"] if s["kind"] == "swap"
                and not any((a.get("review") or {}).get("pass") for a in s.get("attempts", []))
                and len(s.get("attempts", [])) < max_attempts]
        if not todo:
            break
        takes = []
        for s in todo:
            clip = job.dir / "shots" / f"{s['id']}.mp4"
            if not clip.exists():
                media.cut_clip(mv, audio, s, size, clip)
            s.setdefault("attempts", [])
            pending = next((a for a in s["attempts"] if a["status"] in ("pending", "submitted")), None)
            if pending:                                         # a resume: run the attempt that never finished
                a = pending
            else:
                n = len(s["attempts"]) + 1
                text = s.get("text", False) and not any((x.get("review") or {}).get("added_overlays") for x in s["attempts"])
                # half done: the best take so far turned some people but not others -> run that take again for the rest only
                done = [x for x in s["attempts"] if x["status"] == "done" and x.get("review")]
                best = max(done, key=lambda x: float(x["review"].get("score") or 0), default=None)
                missing = (best or {}).get("review", {}).get("missing") or []
                half = best and missing and len(missing) < len(s["people"]) and int(best["review"].get("framing") or 0) >= 1 \
                    and not best["review"].get("broken") and not best["review"].get("extra_people")
                who = missing if half else s["people"]
                prompt, pics = prompts.build(job.state["cast"], who, text, second_pass=bool(half), others=bool(s.get("others")),
                                             style=job.state.get("style"), backdrop=s.get("backdrop", ""))
                a = {"n": n, "seed": config.SEEDS[(n - 1) % len(config.SEEDS)], "status": "pending", "prompt": prompt, "pictures": pics,
                     "file": str(job.dir / "takes" / f"{s['id']}-{n}.mp4")}
                if half:
                    a.update(base=best["n"], video=best["file"], people=who)
                s["attempts"].append(a)
            takes.append({"id": f"{s['id']}#{a['n']}", "video": a.get("video") or str(clip), "pictures": a["pictures"], "prompt": a["prompt"],
                          "seconds": s["gen_frames"] / config.FPS, "width": size[0], "height": size[1], "seed": a["seed"], "out": a["file"],
                          **({"task": a["task"], "key": a["key"]} if a["status"] == "submitted" and a.get("task") else {})})
        job.save()
        job.log("round", takes=len(takes))
        by_take = {t["id"]: t for t in takes}

        def on_event(tid, event, info):
            sid, n = tid.split("#")
            s = next(x for x in job.state["shots"] if x["id"] == sid)
            a = next(x for x in s["attempts"] if x["n"] == int(n))
            a["status"] = {"submitted": "submitted", "done": "done", "failed": "failed"}[event]
            a.update({k: v for k, v in info.items() if k in ("key", "task", "run_s", "cny", "error")})
            job.log(event, take=tid, **{k: v for k, v in info.items() if k in ("key", "run_s", "cny", "error")})
            job.save()

        sched = sched or rh.Scheduler(job.log)
        sched.run(list(by_take.values()), on_event)
        stage_review(job)
    for s in job.state["shots"]:
        if s["kind"] != "swap":
            continue
        done = [a for a in s.get("attempts", []) if a["status"] == "done" and a.get("review")]
        best = sorted(done, key=lambda a: (bool(a["review"].get("pass")), float(a["review"].get("score") or 0)), reverse=True)
        s["chosen"] = best[0]["n"] if best else None
    job.save()


def _colour_fix(before: Path, after: Path) -> list[float]:
    """Per-channel gain and offset that bring SeedVR2's output back to the take's colours (it drifts towards magenta)."""
    import numpy as np
    a = media.raw_frames(before, 24).reshape(-1, 3).astype(np.float64)
    b = media.raw_frames(after, 24).reshape(-1, 3).astype(np.float64)
    if not len(a) or not len(b):
        return [1, 0, 1, 0, 1, 0]
    fix = []
    for ch in range(3):
        g = (a[:, ch].std() + 1e-6) / (b[:, ch].std() + 1e-6)
        fix += [round(float(g), 4), round(float(a[:, ch].mean() - g * b[:, ch].mean()), 2)]
    return fix


def stage_upscale(job: Job):
    """SeedVR2 on each chosen take (trimmed to the shot), short side = the final cut's; keep and graphic shots are
    already at full size. Per shot, so no batch straddles a cut."""
    p = job.state["prepared"]
    fw, fh = finish.final_size(p["width"], p["height"])
    d = job.dir / "upscale"
    d.mkdir(exist_ok=True)
    takes = []
    for s in job.state["shots"]:
        a = s["kind"] == "swap" and s.get("chosen") and next((x for x in s["attempts"] if x["n"] == s["chosen"]), None)
        if not a or (s.get("upscaled") or {}).get("n") == a["n"]:
            continue
        src = d / f"{s['id']}-in.mp4"
        media.ff("-i", a["file"], "-vf", f"trim=end_frame={s['frames']},setpts=N/{config.FPS}/TB", "-an", "-r", str(config.FPS),
                 "-c:v", "libx264", "-crf", "12", "-preset", "fast", str(src))
        takes.append({"id": f"{s['id']}#{a['n']}", "video": str(src), "pictures": [], "upscale": min(fw, fh), "seconds": s["frames"] / config.FPS,
                      "out": str(d / f"{s['id']}-{a['n']}-up.mp4")})
    if not takes:
        return
    job.log("upscale", shots=len(takes))

    def on_event(tid, event, info):
        sid, n = tid.split("#")
        s = next(x for x in job.state["shots"] if x["id"] == sid)
        if event == "done":
            src, out = d / f"{sid}-in.mp4", d / f"{sid}-{n}-up.mp4"
            got = media.frame_count(out)
            if got < s["frames"]:                               # a short output would pull the picture off the music
                s["upscaled"] = {"n": int(n), "failed": f"超分只出了 {got}/{s['frames']} 帧，用没超分的版本", "cny": info.get("cny", 0)}
            else:
                s["upscaled"] = {"n": int(n), "file": str(out), "cny": info.get("cny", 0), "run_s": info.get("run_s", 0),
                                 "fix": _colour_fix(src, out)}
        elif event == "failed":
            s["upscaled"] = {"n": int(n), "failed": str(info.get("error"))[:200]}
        job.log(f"upscale_{event}", take=tid, **{k: v for k, v in info.items() if k in ("run_s", "cny", "error")})
        job.save()
    rh.Scheduler(job.log).run(takes, on_event)


def stage_review(job: Job):
    mv = Path(job.state["prepared"]["mv"])
    todo = [(s, a) for s in job.state["shots"] if s["kind"] == "swap" for a in s.get("attempts", []) if a["status"] == "done" and "review" not in a]

    def one(sa):
        s, a = sa
        take = Path(a["file"])
        lost = _backdrop_lost(job, s, take)
        rv = vision.review_take(job.state["cast"], s["people"], _keys(job, s, mv, s["f0"], "src"), _keys(job, s, take, 0, f"t{a['n']}"))
        rv["backdrop_lost_frames"] = lost
        if lost > max(2, s["frames"] // 10):
            rv["pass"] = False
            rv["issue"] = (rv.get("issue") or "") + f"（{lost} 帧背景和原片差太多）"
        return s, a, rv
    with ThreadPoolExecutor(6) as pool:
        for s, a, rv in pool.map(one, todo):
            a["review"] = rv
            job.log("review", take=f"{s['id']}#{a['n']}", ok=rv["pass"], score=rv.get("score"), issue=rv.get("issue"))
            job.save()


def run(out: Path, source: str | None, cast_specs: list[str], start: float = 0, end: float | None = None, allow_lowres: bool = False,
        max_attempts: int = config.MAX_ATTEMPTS, crop: str = "auto", upscale: bool = True, allow_many_graphic: bool = False) -> dict:
    job = Job(out)
    if not job.state.get("args"):
        if not source or not cast_specs:
            raise SystemExit("新任务要给 --source 和至少一个 --cast")
        job.state["args"] = {"source": source, "cast": cast_specs, "start": start, "end": end, "crop": crop, "upscale": upscale}
        job.save()
    a = job.state["args"]
    stage_prepare(job, a["source"], a["start"], a["end"], allow_lowres, a.get("crop", "auto"))
    stage_shots(job)
    stage_cast(job, a["cast"])
    stage_label(job)
    stage_backdrops(job)
    stage_graphic(job, allow_many_graphic)
    stage_generate(job, max_attempts)
    if a.get("upscale", True):
        stage_upscale(job)
    res = finish.assemble(job.state, job.dir)
    job.state["t_done"] = time.time()
    job.state["final"] = res
    job.save()
    rep = finish.report(job.state, job.dir)
    job.log("done", film=res["film"], frames=res["frames"], want=res["want"], report=str(rep))
    return res
