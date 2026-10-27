#!/usr/bin/env python3
"""一键换人：给一条高清原片和「原片里的人=角色图」，自动切镜头、认人、生成、审片、重跑、拼片。

    python mvswap.py run jobs/<任务名> --source <文件或链接> \\
        --cast "金发女生=cast/a.jpg" --cast "戴黑帽墨镜的男生=cast/b.jpg" [--start 0 --end 97]
    python mvswap.py run jobs/<任务名>        # 断了接着跑（参数记在任务目录里）
    python mvswap.py status jobs/<任务名>     # 现在到哪一步、每个镜头的情况、花了多少
    python mvswap.py assemble jobs/<任务名>   # 只重新拼片和出报告

--cast 右边写一张角色图（全身、穿着成片里的衣服）。原片默认要 720p 以上，确认用低清就加 --allow-lowres。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mvswap import config, finish, pipeline  # noqa: E402


def status(job_dir: Path):
    st = json.loads((job_dir / "state.json").read_text())
    shots = st.get("shots", [])
    kinds = {}
    for s in shots:
        kinds[s.get("kind", "未标")] = kinds.get(s.get("kind", "未标"), 0) + 1
    swap = [s for s in shots if s.get("kind") == "swap"]
    passed = sum(1 for s in swap if any((a.get("review") or {}).get("pass") for a in s.get("attempts", [])))
    running = sum(1 for s in swap for a in s.get("attempts", []) if a["status"] in ("pending", "submitted"))
    cny = sum(a.get("cny", 0) for s in shots for a in s.get("attempts", []))
    print(f"镜头 {len(shots)}：{kinds}；换人镜头过审 {passed}/{len(swap)}，在跑 {running}；RunningHub ¥{cny:.2f}")
    if st.get("final"):
        print("成片：", st["final"]["film"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("job", type=Path)
    r.add_argument("--source")
    r.add_argument("--cast", action="append", default=[])
    r.add_argument("--start", type=float, default=0)
    r.add_argument("--end", type=float)
    r.add_argument("--allow-lowres", action="store_true")
    r.add_argument("--max-attempts", type=int, default=config.MAX_ATTEMPTS)
    r.add_argument("--crop", default="auto", help="auto（裁掉黑边）/ none / w:h:x:y")
    r.add_argument("--no-upscale", action="store_true", help="不做 SeedVR2 超分")
    r.add_argument("--allow-many-graphic", action="store_true", help="图形镜头超过两成也照做")
    s = sub.add_parser("status")
    s.add_argument("job", type=Path)
    a = sub.add_parser("assemble")
    a.add_argument("job", type=Path)
    args = ap.parse_args()
    if args.cmd == "run":
        pipeline.run(args.job, args.source, args.cast, args.start, args.end, args.allow_lowres, args.max_attempts, args.crop,
                     not args.no_upscale, args.allow_many_graphic)
    elif args.cmd == "status":
        status(args.job)
    else:
        job = pipeline.Job(args.job)
        job.state["final"] = finish.assemble(job.state, job.dir)
        job.save()
        print(finish.report(job.state, job.dir))


if __name__ == "__main__":
    main()
