"""A vision model (any OpenAI-compatible chat endpoint) to label and review shots, and GPT Image on APIMart to redraw
graphic shots. Keys are read from the environment and never logged."""
import base64
import json
import mimetypes
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from . import config

_spent = {"vision": 0.0, "image": 0.0}


def _req(base: str, key: str, method: str, path: str, body: dict | None = None, timeout: float = 180) -> dict:
    r = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                               headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": config.UA})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            last = exc
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"{base}{path}: {type(last).__name__}: {str(last).replace(key, '[key]')[:200]}")


def _data_url(p: Path) -> str:
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def ask_json(prompt: str, images: list[Path], max_tokens: int = 4000):
    """One vision call that must answer with JSON; one retry when the reply doesn't parse."""
    parts = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": _data_url(Path(p))}} for p in images]
    text = ""
    for attempt in range(2):
        out = _req(config.VISION_BASE, config.vision_key(), "POST", "/chat/completions",
                   {"model": config.VISION_MODEL, "messages": [{"role": "user", "content": parts}], "max_tokens": max_tokens,
                    "stream": False})                                        # APIMart streams unless told not to
        usage = out.get("usage") or {}                 # billed by tokens; an estimate from the configured prices
        pin, pout = config.VISION_PRICE
        _spent["vision"] += (int(usage.get("prompt_tokens") or 0) * pin + int(usage.get("completion_tokens") or 0) * pout) / 1e6 * config.USD_CNY
        text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        m = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                pass
        parts[0]["text"] = prompt + "\n\nAnswer with the JSON only."
    raise RuntimeError("vision model gave no JSON: " + text[:300])


def shrink(src: Path, out: Path, longest: int = 1024) -> Path:
    """Image models bill reference pictures by their pixels; 1024 px on the long side looks the same and costs less."""
    from PIL import Image
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((longest, longest), Image.LANCZOS)
        im.save(out, "JPEG", quality=90)
    return out


def upload(p: Path) -> str:
    """A public link to a local picture (APIMart keeps it 72 h)."""
    key = config.apimart_key()
    out = subprocess.run(["curl", "-s", "-X", "POST", config.APIMART_BASE + "/uploads/images", "-H", "Authorization: Bearer " + key,
                          "-F", f"file=@{p}"], capture_output=True, text=True, timeout=180).stdout
    data = json.loads(out)
    url = data.get("url") or (data.get("data") or {}).get("url")
    if not url:
        raise RuntimeError("APIMart upload failed: " + out.replace(key, "[key]")[:200])
    return url


def image(out: Path, prompt: str, refs: list[Path], size: str = "16:9", resolution: str = "2k", quality: str = "high") -> float:
    """GPT Image with reference pictures; returns the cost in CNY."""
    import tempfile
    key = config.apimart_key()
    with tempfile.TemporaryDirectory() as tmp:
        urls = [upload(shrink(Path(r), Path(tmp) / f"ref{i}.jpg")) for i, r in enumerate(refs)]
    body = {"model": config.IMAGE_MODEL, "prompt": prompt, "size": size, "resolution": resolution, "quality": quality, "n": 1,
            "image_urls": urls}
    task = ((_req(config.APIMART_BASE, key, "POST", "/images/generations", body).get("data") or [{}])[0] or {}).get("task_id")
    if not task:
        raise RuntimeError("APIMart gave no task id")
    while True:
        st = _req(config.APIMART_BASE, key, "GET", f"/tasks/{task}")
        st = st.get("data") or st
        if st.get("status") == "completed":
            images = (st.get("result") or {}).get("images") or []
            urls = (images[0].get("url") if images else None) or []
            url = urls[0] if isinstance(urls, list) else urls
            req = urllib.request.Request(url, headers={"User-Agent": config.UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                out.write_bytes(r.read())
            cny = round(float(st.get("cost") or 0) * config.USD_CNY, 4)    # APIMart reports US dollars
            _spent["image"] += cny
            return cny
        if st.get("status") in ("failed", "cancelled"):
            raise RuntimeError("image failed: " + json.dumps(st.get("error"), ensure_ascii=False)[:300])
        time.sleep(5)


def vision_spent() -> float:
    """Vision-model spend in this process (estimated from tokens); image costs are recorded on the shots that paid them."""
    return round(_spent["vision"], 3)
