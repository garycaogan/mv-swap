"""One command from source video to finished swap film; every stage is saved to state.json and resumes from there."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock

from . import cast as cast_mod
from . import ai, config, finish, framing, media, prompts, references, rh, vision


class Job:
    def __init__(self, path: Path):
        self.dir = path.resolve()                               # ffmpeg concat lists resolve relative paths against the list file
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self._vision_base = self.state.get("vision_cny", 0.0)   # earlier runs of this job; this run's spend is added on save
        self._vision_start = ai.vision_spent()
        self._lock = RLock()

    def save(self):
        with self._lock:
            self.state["vision_cny"] = round(self._vision_base + ai.vision_spent() - self._vision_start, 3)
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
    cap = job.state.get("args", {}).get("max_segment_seconds", 0)
    if cap:
        import numpy as np
        raw = media.raw_frames(Path(p["mv"]), p["frames"]).astype(np.float32)
        motion = [0.0] + np.abs(np.diff(raw, axis=0)).mean(axis=(1, 2, 3)).tolist()
        job.state["shots"] = media.split_shots(job.state["shots"], min(max_frames, int(cap * config.FPS)), motion)
    job.log("shots", count=len(job.state["shots"]))
    job.save()


def _keys(job: Job, shot: dict, video: Path, offset: int, tag: str) -> list[Path]:
    F = shot["frames"]
    idx = sorted({min(F - 1, max(0, round(F * x))) for x in (0.15, 0.5, 0.85)})
    d = job.dir / "frames"
    d.mkdir(exist_ok=True)
    return media.keyframes(video, [offset + i for i in idx], d, f"{shot['id']}-{tag}")


def stage_cast(job: Job, specs: list[str]):
    try:
        if not job.state.get("cast"):
            job.state["cast"] = cast_mod.parse(specs, job.dir / "refs")
            job.save()
        cast = job.state["cast"]
        for c in cast:
            if not c.get("desc"):
                cast_mod.describe([c])
                job.save()                                     # each paid description survives the next call failing
        missing = [c for c in cast if not c.get("src_desc")]
        if missing:
            shots = job.state["shots"]
            pick = [shots[int(i * (len(shots) - 1) / 7)] for i in range(min(8, len(shots)))]
            frames = []
            for s in pick:
                ks = _keys(job, s, Path(job.state["prepared"]["mv"]), s["f0"], "src")
                frames.append(ks[len(ks) // 2])
            job.state["style"] = vision.performers(missing, frames)
            job.save()
            job.log("style", kind=job.state["style"]["kind"], desc=job.state["style"]["desc"])
            for c in cast:
                job.log("cast", id=c["id"], source=c["src_desc"], becomes=c["name"], gender=f"{c.get('src_gender')}->{c.get('gender')}")
    finally:
        job.save()                                             # also record billed calls whose replies could not be parsed


def _vision_batch(job: Job, todo: list, call, record, workers: int):
    """Checkpoint each finished call in its worker; another slow/failed call must not hide its result."""
    def one(item):
        try:
            result = call(item)
            with job._lock:
                record(item, result)
                job.save()
        finally:
            job.save()

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))


def stage_label(job: Job):
    todo = [s for s in job.state["shots"] if "kind" not in s]
    if not todo:
        return
    mv = Path(job.state["prepared"]["mv"])

    def one(s):
        return vision.label_shot(job.state["cast"], _keys(job, s, mv, s["f0"], "src"), job.state.get("style"))

    def record(s, lab):
        s.update(lab)
        job.log("label", shot=s["id"], kind=s["kind"], people=",".join(s["people"]), framing=s["framing"])

    _vision_batch(job, todo, one, record, 10)


def stage_backdrops(job: Job):
    """Shots labelled before labels named the setting get it now (one small vision call each)."""
    mv = Path(job.state["prepared"]["mv"])
    todo = [s for s in job.state["shots"] if s.get("kind") == "swap" and "backdrop" not in s]
    _vision_batch(job, todo, lambda s: vision.backdrop(_keys(job, s, mv, s["f0"], "src")),
                  lambda s, bg: s.update(backdrop=bg), 10)


def stage_references(job: Job):
    """Reuse source geometry; edit modes have checkpointed generation and image QA."""
    mode = job.state.get("args", {}).get("ref_mode", "crop")
    by = {c["id"]: c for c in job.state.get("cast", [])}
    for s in job.state.get("shots", []):
        if s.get("kind") != "swap" or s.get("references"):
            continue
        if any(not by[p].get("references") for p in s["people"]):
            # Legacy jobs retain their paid inputs; new jobs always carry analyzed reference candidates.
            continue
        refs = {}
        for person in s["people"]:
            geometry = (s.get("geometry") or {}).get(person) or {"framing": s.get("framing")}
            if mode == "crop":
                refs[person] = framing.reference(by[person], geometry, job.dir / "refs" / f"{s['id']}-{person}-{framing.shot_size(geometry.get('framing'))}.jpg")
            else:
                refs[person] = references.prepare(job, s, by[person], geometry, mode)
        s["references"] = refs
        job.save()
        job.log("references", shot=s["id"], pictures=",".join(r["picture"] for r in refs.values()))


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

    def unfinished(a):
        return a["status"] in ("pending", "submitted")

    while True:
        # Collect paid remote tasks before considering any new submissions, even at/above the attempt limit.
        collecting = [s for s in job.state["shots"] if s["kind"] == "swap"
                      and any(unfinished(a) and a.get("task") for a in s.get("attempts", []))]
        todo = collecting or [s for s in job.state["shots"] if s["kind"] == "swap"
                and not any((a.get("review") or {}).get("pass") for a in s.get("attempts", []))
                and (len(s.get("attempts", [])) < max_attempts
                     or any(unfinished(a) and a["n"] <= max_attempts for a in s.get("attempts", [])))]
        if not todo:
            break
        takes = []
        for s in todo:
            clip = job.dir / "shots" / f"{s['id']}.mp4"
            s.setdefault("attempts", [])
            pending = next((a for a in s["attempts"] if unfinished(a) and a.get("task")), None)
            pending = pending or next((a for a in s["attempts"] if unfinished(a)), None)
            if not (pending and pending.get("task")) and not clip.exists():
                media.cut_clip(mv, audio, s, size, clip)
            if pending:                                         # a resume: run the attempt that never finished
                a = pending
            else:
                n = len(s["attempts"]) + 1
                text = s.get("text", False) and not any((x.get("review") or {}).get("added_overlays") for x in s["attempts"])
                # half done: the best take so far turned some people but not others -> run that take again for the rest only
                done = [x for x in s["attempts"] if x["status"] == "done" and x.get("review")]
                best = max(done, key=lambda x: float(x["review"].get("score") or 0), default=None)
                missing = (best or {}).get("review", {}).get("missing") or []
                half = best and missing and not best["review"].get("drift_seconds") and not best["review"].get("unchecked_seconds") \
                    and len(missing) < len(s["people"]) and int(best["review"].get("framing") or 0) >= 1 \
                    and not best["review"].get("broken") and not best["review"].get("extra_people")
                who = missing if half else s["people"]
                shot_cast = [{**c, **(s.get("references") or {}).get(c["id"], {})} for c in job.state["cast"]]
                prompt, pics = prompts.build(shot_cast, who, text, second_pass=bool(half), others=bool(s.get("others")),
                                             style=job.state.get("style"), backdrop=s.get("backdrop", ""))
                a = {"n": n, "seed": config.SEEDS[(n - 1) % len(config.SEEDS)], "status": "pending", "prompt": prompt, "pictures": pics,
                     "file": str(job.dir / "takes" / f"{s['id']}-{n}.mp4")}
                if half:
                    a.update(base=best["n"], video=best["file"], people=who)
                s["attempts"].append(a)
            takes.append({"id": f"{s['id']}#{a['n']}", "video": a.get("video") or str(clip), "pictures": a["pictures"], "prompt": a["prompt"],
                          "seconds": s["gen_frames"] / config.FPS, "width": size[0], "height": size[1], "seed": a["seed"], "out": a["file"],
                          **({"task": a["task"], "key": a["key"]} if a.get("task") else {})})
        job.save()
        job.log("round", takes=len(takes))
        by_take = {t["id"]: t for t in takes}

        def on_event(tid, event, info):
            sid, n = tid.split("#")
            s = next(x for x in job.state["shots"] if x["id"] == sid)
            a = next(x for x in s["attempts"] if x["n"] == int(n))
            a["status"] = {"submitted": "submitted", "done": "done", "failed": "failed"}[event]
            a.update({k: v for k, v in info.items() if k in ("key", "task", "run_s", "cny", "error")})
            job.save()
            job.log(event, take=tid, **{k: v for k, v in info.items() if k in ("key", "run_s", "cny", "error")})

        sched = sched or rh.Scheduler(job.log)
        _known_keys(sched, by_take.values())
        try:
            sched.run(list(by_take.values()), on_event)
        finally:
            job.save()
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

    def complete(s, up):
        src, out = d / f"{s['id']}-in.mp4", Path(up["out"])
        got = media.frame_count(out)
        if got < s["frames"]:                                   # a short output would pull the picture off the music
            up["failed"] = f"超分只出了 {got}/{s['frames']} 帧，用没超分的版本"
        else:
            up["fix"] = _colour_fix(src, out)
            up["file"] = str(out)
        up["status"] = "done"
        job.save()

    for s in job.state["shots"]:
        a = s["kind"] == "swap" and s.get("chosen") and next((x for x in s["attempts"] if x["n"] == s["chosen"]), None)
        if not a:
            continue
        up = s.get("upscaled") or {}
        if up.get("n") == a["n"]:
            if up.get("file") or up.get("failed") or up.get("status") in ("done", "failed"):
                continue
            if up.get("status") == "downloaded":
                complete(s, up)                                 # local post-processing can fail after the paid result arrives
                continue
        else:
            up = {"n": a["n"], "status": "pending", "out": str(d / f"{s['id']}-{a['n']}-up.mp4")}
            s["upscaled"] = up
        src = d / f"{s['id']}-in.mp4"
        if not up.get("task"):
            media.ff("-i", a["file"], "-vf", f"trim=end_frame={s['frames']},setpts=N/{config.FPS}/TB", "-an", "-r", str(config.FPS),
                     "-c:v", "libx264", "-crf", "12", "-preset", "fast", str(src))
        takes.append({"id": f"{s['id']}#{a['n']}", "video": str(src), "pictures": [], "upscale": min(fw, fh), "seconds": s["frames"] / config.FPS,
                      "out": up["out"], **({"task": up["task"], "key": up["key"]} if up.get("task") else {})})
    if not takes:
        return
    job.save()
    job.log("upscale", shots=len(takes))

    def on_event(tid, event, info):
        sid, n = tid.split("#")
        s = next(x for x in job.state["shots"] if x["id"] == sid)
        up = s["upscaled"]
        up.update({k: v for k, v in info.items() if k in ("key", "task", "run_s", "cny")})
        up["status"] = event
        if event == "failed":
            up["failed"] = str(info.get("error"))[:200]
        if event == "done":
            up["status"] = "downloaded"
        job.save()                                             # task/key and billing survive logging or colour-fix failures
        if event == "done":
            complete(s, up)
        job.log(f"upscale_{event}", take=tid, **{k: v for k, v in info.items() if k in ("run_s", "cny", "error")})
    sched = rh.Scheduler(job.log)
    _known_keys(sched, takes)
    try:
        sched.run(takes, on_event)
    finally:
        job.save()


def _known_keys(sched, takes):
    """A task submitted under a key that is no longer configured can't be polled; stop instead of resubmitting or spinning."""
    lost = sorted({t["key"] for t in takes if t.get("task") and t.get("key") not in sched.keys})
    if lost:
        raise SystemExit(f"有已提交的 RunningHub 任务记在 {'、'.join(lost)} 名下（k1 是 RUNNINGHUB_API_KEY 里的第一个 key，依此类推），"
                         "现在没有这个 key。把原来的 key 按原顺序配回去再跑，已经付过钱的任务会接着收")


def stage_review(job: Job):
    mv = Path(job.state["prepared"]["mv"])
    todo = [(s, a) for s in job.state["shots"] if s["kind"] == "swap" for a in s.get("attempts", []) if a["status"] == "done" and "review" not in a]

    def one(sa):
        s, a = sa
        take = Path(a["file"])
        lost = _backdrop_lost(job, s, take)
        panels, seconds = framing.pairs(mv, take, s, job.dir / "review" / f"{s['id']}-{a['n']}", "seconds")
        rv = vision.review_take(job.state["cast"], s["people"], panels, [], seconds=seconds)
        rv["evidence"] = [str(p) for p in panels]
        rv["backdrop_lost_frames"] = lost
        if lost > max(2, s["frames"] // 10):
            rv["pass"] = False
            rv["issue"] = (rv.get("issue") or "") + f"（{lost} 帧背景和原片差太多）"
        return rv

    def record(sa, rv):
        s, a = sa
        a["review"] = rv
        job.log("review", take=f"{s['id']}#{a['n']}", ok=rv["pass"], score=rv.get("score"), issue=rv.get("issue"))

    _vision_batch(job, todo, one, record, 6)


def run(out: Path, source: str | None, cast_specs: list[str], start: float = 0, end: float | None = None, allow_lowres: bool = False,
        max_attempts: int = config.MAX_ATTEMPTS, crop: str = "auto", upscale: bool = True, allow_many_graphic: bool = False,
        accept_failed: bool = False, max_segment_seconds: float = 0, ref_mode: str | None = None, refs_only: bool = False) -> dict:
    import math
    if not math.isfinite(max_segment_seconds) or (max_segment_seconds != 0 and max_segment_seconds < 1):
        raise SystemExit("--max-segment-seconds 要为 0（关闭）或至少 1 秒")
    if ref_mode is not None and ref_mode not in references.MODES:
        raise SystemExit("--ref-mode 要为 crop、neutral 或 keyframe")
    job = Job(out)
    if not job.state.get("args"):
        if not source or not cast_specs:
            raise SystemExit("新任务要给 --source 和至少一个 --cast")
        job.state["args"] = {"source": source, "cast": cast_specs, "start": start, "end": end, "crop": crop, "upscale": upscale,
                             "max_segment_seconds": max_segment_seconds, "ref_mode": ref_mode or config.DEFAULT_REF_MODE}
        job.save()
    try:
        return _run(job, allow_lowres, max_attempts, allow_many_graphic, accept_failed, refs_only)
    finally:
        job.save()


def _run(job: Job, allow_lowres: bool, max_attempts: int, allow_many_graphic: bool, accept_failed: bool, refs_only: bool = False) -> dict:
    a = job.state["args"]
    stage_prepare(job, a["source"], a["start"], a["end"], allow_lowres, a.get("crop", "auto"))
    stage_shots(job)
    stage_cast(job, a["cast"])
    stage_label(job)
    stage_backdrops(job)
    try:
        stage_references(job)
    except RuntimeError as exc:
        job.state["reference_blocked"] = str(exc)
        job.state["t_reference_done"] = time.time()
        job.save()
        finish.report(job.state, job.dir)
        raise SystemExit(str(exc)) from None
    job.state.pop("reference_blocked", None)
    if refs_only:
        job.state["references_ready"] = True
        job.state["t_reference_done"] = time.time()
        job.save()
        finish.report(job.state, job.dir)
        return {"references_ready": True}
    stage_graphic(job, allow_many_graphic)
    # Keep an accepted diagnostic cut on resume, including an interruption during its paid upscale.
    accepted = accept_failed or job.state.get("accept_failed", False)
    resume_accepted = accepted and (job.state.get("review_blocked") or job.state.get("accept_failed"))
    stage_generate(job, 0 if resume_accepted else max_attempts)
    upscale = a.get("upscale", True)
    flagged = finish.failed_shots(job.state)
    # Only the paid upscale waits for a person; without it the cut is already final.
    job.state["review_blocked"] = [s["id"] for s in flagged] if upscale and not accepted else []
    job.state["accept_failed"] = bool(flagged and accepted)
    job.save()
    if upscale and not job.state["review_blocked"]:
        stage_upscale(job)
    res = finish.assemble(job.state, job.dir)
    job.state["t_done"] = time.time()
    job.state["final"] = res
    job.save()
    rep = finish.report(job.state, job.dir)
    job.log("diagnostic" if job.state["review_blocked"] else "done", film=res["film"], frames=res["frames"], want=res["want"], report=str(rep))
    if job.state["review_blocked"]:
        print("有镜头所有版本都没过审，已跳过付费超分。看过对照片觉得能用，再加 --accept-failed 重跑会接着超分。", flush=True)
    return res
