"""Keys, endpoints and the numbers the pipeline relies on. Keys come from the environment or a `.env` file at the repo
root (see `.env.example`); they are read, never written or logged."""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent


def _load_env_file(path: Path = ROOT / ".env"):
    """KEY=value lines; values already in the environment win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def need(name: str, what: str) -> str:
    v = env(name)
    if not v:
        raise SystemExit(f"缺少 {name}（{what}）：写进仓库根目录的 .env，或者 export {name}=…；格式见 .env.example")
    return v


RH_BASE = env("RUNNINGHUB_BASE", "https://www.runninghub.cn")
RH_CONCURRENCY = int(env("RUNNINGHUB_CONCURRENCY", "5"))         # a member key runs 5 tasks at once; enterprise keys more
RH_COIN_CNY = float(env("RUNNINGHUB_COIN_CNY", str(399 / 180000)))  # what one RH coin cost you (Max monthly: ¥399 for 180k)
RH_WORKFLOW_ID = "2109190174289780738"   # Nick「MinimaxH3 swap 角色迁移」on RunningHub.cn
RH_SEEDVR_ID = "1989896719183867905"     # C-KIRIN「视频高清放大+补帧：SeedVR2」on RunningHub.cn
CACHE = Path(env("MVSWAP_CACHE", str(Path.home() / ".cache/mv-swap")))

# vision (label shots, review takes) through any OpenAI-compatible endpoint; by default the same APIMart key as images
VISION_BASE = env("VISION_BASE_URL", "https://api.apimart.ai/v1").rstrip("/")
VISION_MODEL = env("VISION_MODEL", "gemini-3.8-flash")
VISION_PRICE = (float(env("VISION_PRICE_IN", "0.3")), float(env("VISION_PRICE_OUT", "2.5")))   # USD per million tokens, for the estimate
APIMART_BASE = env("APIMART_BASE_URL", "https://api.apimart.ai/v1").rstrip("/")
IMAGE_MODEL = env("IMAGE_MODEL", "gpt-image-2.5-sunburst")
USD_CNY = float(env("USD_CNY", "7.2"))
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36"

FPS = 24
PAD_MIN = 12                     # held frames added to every shot: H3 only makes 17k+5 frames, and a bad tail lands in the padding
MAX_GEN_FRAMES = 362             # 15.08 s, the longest take we have run on this workflow
FINAL_LONG_SIDE = 1920
MIN_SOURCE_SHORT_SIDE = 720      # low-res sources come out soft and smeared; --allow-lowres to go ahead anyway
CUT_SCORE = 0.28                 # scene score for an obvious cut
HIDDEN_CUT_SCORE = 0.12          # second pass inside shots: pink-on-pink cuts the first pass misses
MAX_ATTEMPTS = 3
SEEDS = [20261024, 777123, 4242]


def rh_keys() -> list[str]:
    """One or more RunningHub API keys (comma-separated). Several keys share the work."""
    keys = [k.strip() for k in need("RUNNINGHUB_API_KEY", "RunningHub 的 API key，控制台「API 调用」里拿").split(",") if k.strip()]
    return keys


def vision_key() -> str:
    return env("VISION_API_KEY") or need("APIMART_API_KEY", "APIMart 的 key，用来认人、审片；也可以单独配 VISION_API_KEY")


def apimart_key() -> str:
    return need("APIMART_API_KEY", "APIMart 的 key，图形镜头重画要用")
