"""
HuggingFace 环境设置 —— 必须在 import transformers / huggingface_hub **之前**调用 setup()。

为什么需要这个文件：
  1. **huggingface.co 在本机网络下不可达**（实测连接超时），必须走 hf-mirror.com 镜像。
  2. 默认缓存目录在 C 盘（`~/.cache/huggingface`），而 C 盘只剩 24GB，
     一个 Qwen 7B 就是 15GB —— 缓存必须挪到 E 盘。
  3. Xet 存储后端（`cas-bridge.xethub.hf.co`）**不经过镜像**，实测大文件会超时，
     需要 `HF_HUB_DISABLE_XET=1` 关掉。

用法：在脚本最顶部（第一行 import 之前）写

    from hf_env import setup   # 若脚本在子目录里，用 sys.path 或相对导入
    setup()

    import transformers        # 必须在 setup() 之后
"""

from __future__ import annotations

import os
from pathlib import Path

MIRROR = "https://hf-mirror.com"
# 本机缓存位置（E 盘空间充裕；云端机器不设，用默认路径）
LOCAL_CACHE = Path(r"E:\hf_cache")


def setup(verbose: bool = False) -> dict:
    """设置 HF_ENDPOINT / HF_HOME / HF_HUB_DISABLE_XET。已存在的值不会被覆盖。"""
    os.environ.setdefault("HF_ENDPOINT", MIRROR)

    # 关键：Xet 存储后端的 blob 走 cas-bridge.xethub.hf.co，**不经过 hf-mirror**。
    # 实测在首次下载时它在大文件上超时了（"The read operation timed out"），
    # 虽然会自动续传，但会让下载时间翻倍。关掉它，全部流量走镜像。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # 只在 Windows 且 E 盘存在时才改缓存位置；云端 Linux 机器保持默认
    if os.name == "nt" and LOCAL_CACHE.drive and Path(LOCAL_CACHE.drive + "\\").exists():
        LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(LOCAL_CACHE))

    # 关掉「每次联网检查是否有新版本」的提示
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    info = {
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
        "HF_HOME": os.environ.get("HF_HOME", "(默认 ~/.cache/huggingface)"),
        "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "(未设置)"),
    }
    if verbose:
        for k, v in info.items():
            print(f"  {k} = {v}")
    return info


def report() -> None:
    """打印当前生效的配置 + huggingface_hub 的实际缓存路径。"""
    info = setup()
    print("  环境变量：")
    for k, v in info.items():
        print(f"    {k:12s} = {v}")
    try:
        from huggingface_hub import constants  # setup() 之后再导入

        print(f"    实际缓存     = {constants.HF_HUB_CACHE}")
        print(f"    端点         = {constants.ENDPOINT}")
    except Exception as e:  # noqa: BLE001
        print(f"    (读取 huggingface_hub 失败: {type(e).__name__}: {e})")


if __name__ == "__main__":
    print("HuggingFace 配置：")
    report()
