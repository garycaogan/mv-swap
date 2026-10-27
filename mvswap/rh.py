"""RunningHub: fetch the two workflows' graphs, upload, submit, poll, download — spread over one or more API keys.

The graphs belong to their authors on RunningHub; they are fetched with your key (`getJsonApiFormat`) and cached, not
shipped here. A key billed from a wallet holds some balance per running task and refuses new ones (605
NOT_ENOUGH_BALANCE) when it runs short; a member key runs a fixed number at once (421 when full). Either way the
scheduler just waits for a slot. Keys are read from the environment and replaced by [key] in anything saved or printed.
"""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from . import config

# the inputs we fill in, and the node types they must be — checked on every fetched graph
SWAP_NODES = {"554": "VHS_LoadVideo", "200": "LoadImage", "385": "MiniMaxH3ReferenceToVideo", "557": "JjkText", "386": "Float",
              "456": "SelfLiftH3Sampler"}
SEEDVR_NODES = {"16": "VHS_LoadVideo", "65": "SeedVR2", "71": "Int"}


def redact(obj):
    s = json.dumps(obj, ensure_ascii=False)
    for k in config.rh_keys():
        s = s.replace(k, "[key]")
    return json.loads(s)


def call(key: str, path: str, body: dict, timeout: float = 60) -> dict:
    req = urllib.request.Request(config.RH_BASE + path, data=json.dumps({"apiKey": key, **body}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except Exception as exc:                        # creation errors come back as code -1, never retried blindly
        out = {"code": -1, "msg": f"network: {type(exc).__name__}: {exc}"}
    if isinstance(out.get("data"), dict):
        out["data"].pop("netWssUrl", None)
        out["data"].pop("clientId", None)
    return redact(out)


def account(key: str) -> dict:
    return call(key, "/uc/openapi/accountStatus", {"apikey": key}).get("data") or {}


def workflow(workflow_id: str, nodes: dict) -> dict:
    """The workflow's API graph, from the cache or fetched with the first key; refuses a graph whose nodes changed."""
    cache = config.CACHE / f"workflow-{workflow_id}.json"
    if cache.is_file():
        g = json.loads(cache.read_text())
    else:
        out = call(config.rh_keys()[0], "/api/openapi/getJsonApiFormat", {"workflowId": workflow_id})
        p = (out.get("data") or {}).get("prompt") if isinstance(out.get("data"), dict) else None
        if out.get("code") != 0 or not p:
            raise SystemExit(f"取不到 RunningHub 工作流 {workflow_id} 的节点图：{out.get('code')} {out.get('msg')}。"
                             f"先在网页上打开 https://www.runninghub.cn/post/{workflow_id} 确认它还在。")
        g = json.loads(p) if isinstance(p, str) else p
    bad = [n for n, t in nodes.items() if n not in g or (t and g[n].get("class_type") != t)]
    if bad:
        raise SystemExit(f"工作流 {workflow_id} 的节点变了（{', '.join(bad)} 对不上），作者可能改过它；删掉 {cache} 重取，还不行就要改 rh.py 里的节点号")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(g, ensure_ascii=False))
    return g


def upload(path: Path, key: str) -> str:
    last = None
    for attempt in range(5):                        # the upload endpoint drops connections now and then
        res = subprocess.run(["curl", "-s", "-X", "POST", config.RH_BASE + "/task/openapi/upload", "-H", "Authorization: Bearer " + key,
                              "-F", "apiKey=" + key, "-F", "fileType=input", "-F", f"file=@{path}"],
                             capture_output=True, text=True, timeout=300)
        try:
            out = redact(json.loads(res.stdout))
            if out.get("code") == 0:
                return out["data"]["fileName"]
            last = out
        except ValueError:
            last = res.stdout[:200]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"upload {path.name} failed: {last}")


def download(url: str, out: Path) -> bool:
    ok = subprocess.run(["curl", "-s", "-L", "--retry", "6", "--retry-all-errors", "--retry-delay", "5", "-o", str(out), url],
                        timeout=900).returncode == 0
    return ok and out.exists() and out.stat().st_size > 10000


def build_graph(base: dict, video_name: str, picture_names: list[str], prompt: str, seconds: float, width: int, height: int, seed: int) -> dict:
    """The swap graph with our inputs: 554 source video, 200 (+901…) pictures, 557 prompt, 386 seconds, 385 size, 456 seed."""
    g = json.loads(json.dumps(base))
    g["554"]["inputs"]["video"] = video_name
    g["200"]["inputs"]["image"] = picture_names[0]
    for i, name in enumerate(picture_names[1:], start=1):
        nid = str(900 + i)
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}, "_meta": {"title": f"picture {i + 1}"}}
        g["385"]["inputs"][f"ref_images.ref_image_{i}"] = [nid, 0]
    g["557"]["inputs"]["text"] = prompt
    g["386"]["inputs"]["value"] = float(seconds)
    g["385"]["inputs"]["width"], g["385"]["inputs"]["height"] = width, height
    g["456"]["inputs"]["seed"] = seed
    return g


def build_seedvr_graph(base: dict, video_name: str, short_side: int = 1080) -> dict:
    """SeedVR2 3B on our frames as they are: 65 reads the video directly (the workflow's own 0.3 MP pre-shrink, nodes
    68/69, is dropped) and 71 sets the output's short side (about 43 GPU-seconds per output second, linear)."""
    g = json.loads(json.dumps(base))
    g["16"]["inputs"]["video"] = video_name
    g["65"]["inputs"]["images"] = ["16", 0]
    g["65"]["inputs"]["seed"] = 42
    g["71"]["inputs"]["value"] = short_side
    g.pop("68", None)
    g.pop("69", None)
    return g


class Scheduler:
    """Runs a list of takes to completion across the keys.

    take: {"id", "video", "pictures", "prompt", "seconds", "width", "height", "seed", "out"} — `out` is where the mp4 goes.
    `on_event(take_id, event, info)` is called on submit / done / failed so the caller can persist state.
    Keys are referred to by their position ("k1", "k2"…) in saved state, never by value.
    """

    def __init__(self, log):
        self.log = log
        self.keys = {f"k{i + 1}": k for i, k in enumerate(config.rh_keys())}
        self.uploads: dict[str, str] = {}
        self.caps = {name: config.RH_CONCURRENCY for name in self.keys}
        self._swap = self._seedvr = None
        self.log("rh_capacity", **self.caps)

    def _name(self, path: str, key_name: str) -> str:
        """Uploads belong to the account of the key that made them."""
        if (path, key_name) not in self.uploads:
            self.uploads[(path, key_name)] = upload(Path(path), self.keys[key_name])
        return self.uploads[(path, key_name)]

    def _graph(self, take: dict, key_name: str) -> tuple[dict, str]:
        if take.get("upscale"):                                # a finished take going through SeedVR2
            self._seedvr = self._seedvr or workflow(config.RH_SEEDVR_ID, SEEDVR_NODES)
            return build_seedvr_graph(self._seedvr, self._name(take["video"], key_name), take["upscale"]), config.RH_SEEDVR_ID
        self._swap = self._swap or workflow(config.RH_WORKFLOW_ID, SWAP_NODES)
        return build_graph(self._swap, self._name(take["video"], key_name), [self._name(p, key_name) for p in take["pictures"]],
                           take["prompt"], take["seconds"], take["width"], take["height"], take["seed"]), config.RH_WORKFLOW_ID

    def run(self, takes: list[dict], on_event=lambda *a: None, poll: float = 20):
        live: dict[str, dict] = {t["id"]: {"take": t, "key": t["key"], "task": t["task"], "t": time.time()}
                                 for t in takes if t.get("task") and t.get("key") in self.keys}   # resumed: already running, just poll
        queue = sorted((t for t in takes if t["id"] not in live), key=lambda t: -t["seconds"])   # long takes first
        retries: dict[str, int] = {}
        while queue or live:
            for take in list(queue):
                key = next((k for k in self.keys if sum(1 for v in live.values() if v["key"] == k) < self.caps[k]), None)
                if not key:
                    break
                g, wid = self._graph(take, key)
                res = call(self.keys[key], "/task/openapi/create", {"workflowId": wid, "workflow": json.dumps(g, ensure_ascii=False), "addMetadata": False})
                if res.get("code") == 0:
                    queue.remove(take)
                    live[take["id"]] = {"take": take, "key": key, "task": str(res["data"]["taskId"]), "t": time.time()}
                    on_event(take["id"], "submitted", {"key": key, "task": live[take["id"]]["task"]})
                elif res.get("msg") == "NOT_ENOUGH_BALANCE" or res.get("code") in (605, 421, 415):
                    self.caps[key] = sum(1 for v in live.values() if v["key"] == key)   # this key is full for now
                    self.log("rh_key_full", key=key, live=self.caps[key], msg=res.get("msg"))
                    if not live and not any(self.caps.values()):
                        raise RuntimeError("RunningHub 提交不上：钱包余额不够或者 key 不可用（" + str(res.get("msg")) + "）")
                    break
                else:
                    queue.remove(take)
                    on_event(take["id"], "failed", {"error": res})
            for tid, v in list(live.items()):
                res = call(self.keys[v["key"]], "/task/openapi/outputs", {"taskId": v["task"]})
                code = res.get("code")
                if code == 0 and res.get("data"):
                    item = next((d for d in res["data"] if str(d.get("fileType", "")).lower() in ("mp4", "video")), res["data"][0])
                    if not download(item["fileUrl"], Path(v["take"]["out"])):
                        self.log("download_retry", take=tid)
                        continue
                    money = float(item.get("consumeMoney") or 0)
                    coins = float(item.get("consumeCoins") or 0)
                    on_event(tid, "done", {"key": v["key"], "run_s": int(item.get("taskCostTime") or 0), "wall_s": round(time.time() - v["t"]),
                                           "cny": round(money + coins * config.RH_COIN_CNY, 4), "money": money, "coins": coins})
                    del live[tid]
                    self.caps[v["key"]] = min(config.RH_CONCURRENCY, self.caps[v["key"]] + 1)   # a slot freed up
                elif code in (805, 807):
                    reason = str((res.get("data") or {}).get("failedReason", res.get("msg")))[:300]
                    del live[tid]
                    if retries.get(tid, 0) < 1 and ("memory" in reason.lower() or "oom" in reason.lower() or code == 805):
                        retries[tid] = retries.get(tid, 0) + 1                       # OOM and odd node errors usually pass on a rerun
                        queue.append(v["take"])
                        self.log("rh_retry", take=tid, reason=reason)
                    else:
                        on_event(tid, "failed", {"error": reason})
            if live or queue:
                time.sleep(poll)
