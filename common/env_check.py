"""
环境自检脚本 —— 每次换机器或换环境后先跑这个。

用法：
    conda activate llm
    python check_env.py
"""

import sys
import platform
import time


def main():
    print("=" * 52)
    print("环境自检")
    print("=" * 52)

    print(f"操作系统   : {platform.system()} {platform.release()}")
    print(f"Python     : {sys.version.split()[0]}")
    print(f"解释器路径 : {sys.executable}")

    print("-" * 52)

    try:
        import torch
    except ImportError:
        print("PyTorch 未安装。请先执行： pip install torch")
        return 1

    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA 编译版本: {torch.version.cuda}")

    available = torch.cuda.is_available()
    print(f"CUDA 可用  : {available}")

    if available:
        print(f"显卡       : {torch.cuda.get_device_name(0)}")
        print(f"算力等级   : {torch.cuda.get_device_capability(0)}")
        print(f"显卡数量   : {torch.cuda.device_count()}")
        props = torch.cuda.get_device_properties(0)
        print(f"显存       : {props.total_memory / 1024**3:.1f} GB")
        device = "cuda"
    else:
        print("当前为 CPU 模式。")
        print("  如果本机没有 NVIDIA 显卡，这是正常的 —— 训练请用云端 GPU。")
        device = "cpu"

    print("-" * 52)

    x = torch.randn(3, 4)
    print(f"张量运算   : 正常，(3,4) @ (4,3) -> {tuple((x @ x.T).shape)}")

    n = 2048
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        c = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"矩阵乘基准 : {n}x{n} x10 用时 {elapsed:.2f}s  [{device.upper()}]")

    if not available:
        print()
        print("提示：CPU 模式下这个基准会偏慢，属正常。")
        print("      跑通小模型（10M 参数级）没问题，7B 级微调请上 GPU。")

    print("=" * 52)
    print("自检完成，环境可用。")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    sys.exit(main())
