"""
W1 · 空白版（第二次重写用）

用途：合上 attention.py，用这个文件从零重写一遍。
      不看提示、不看参考、只凭记忆。

与 attention.py 的区别：
    - 没有任何提示（没有"为什么"、没有伪代码、没有 API 暗示）
    - 只保留函数签名、形状契约、测试用例
    - 6 处填空

用法：
    conda activate llm
    cd E:\\llm-lab\\week01
    python attention_blank.py

在 VS Code 里按 Ctrl+F 搜索 **填空**，可以依次跳到 6 处。

================================================================
重写记录（每次写完记一行，看得见的进步才有动力）
================================================================
第 1 次：__2026____ 年 _9__ 月 __11_ 日    用时 ___8___ 分钟    结果 _9__ / 9
第 2 次：______ 年 ___ 月 ___ 日    用时 ______ 分钟    结果 ___ / 9
第 3 次：______ 年 ___ 月 ___ 日    用时 ______ 分钟    结果 ___ / 9

（目标：第 3 次能在 10 分钟内、不查任何资料、9/9 通过）
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _blank(where: str):
    """占位符。把它替换成你的代码，这一行就不会再被触发了。"""
    raise NotImplementedError(f"{where} 还没填")


# ==========================================================================
# 1 · 数值稳定的 softmax
# ==========================================================================
def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿 dim 维做 softmax，返回同形状张量，要求数值稳定。"""
    x_max = x.max(dim=dim,keepdim=True).values
    e = torch.exp(x-x_max)

    return e/e.sum(dim=dim,keepdim=True)


# ==========================================================================
# 2 · 缩放点积注意力
# ==========================================================================
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Q:    (..., L, d_k)
    K:    (..., S, d_k)
    V:    (..., S, d_v)
    mask: None 或可广播到 (..., L, S) 的 bool 张量，True 表示允许关注
    返回: (..., L, d_v)
    """
    d_k = Q.size(-1)

    # 填空 2/6
    scores = (Q@K.transpose(-1,-2))/math.sqrt(d_k)

    # 填空 3/6
    if mask is not None:
        scores = scores.masked_fill(~mask,float("-inf"))

    # 填空 4/6
    attn = stable_softmax(scores,-1)

    # 填空 5/6
    out = attn@V

    return out


# ==========================================================================
# 3 · 因果掩码
# ==========================================================================
def make_causal_mask(seq_len: int) -> torch.Tensor:
    """
    返回形状 (seq_len, seq_len) 的 bool 张量。
    语义：位置 i 只能看到 j <= i（不能看未来）。
    """
    # 填空 6/6
    row = torch.arange(seq_len).view(-1,1)
    col = torch.arange(seq_len).view(1,-1)
    return row>=col


# ==========================================================================
# 以下为测试脚手架，不需要改动
# ==========================================================================

def _reference_attention(Q, K, V, mask=None):
    """官方实现，用于对照。"""
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)


def test_softmax_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(4, 7) * 3
    mine = stable_softmax(x, dim=-1)
    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(mine, ref, atol=1e-6), f"与 torch.softmax 不一致，最大误差 {(mine - ref).abs().max():.2e}"


def test_softmax_sums_to_one():
    torch.manual_seed(0)
    x = torch.randn(5, 9)
    out = stable_softmax(x, dim=-1)
    assert torch.allclose(out.sum(dim=-1), torch.ones(5), atol=1e-6), "每一行之和应该等于 1"


def test_softmax_numerical_stability():
    x = torch.tensor([[1e4, 1e4 - 1.0, 0.0]])
    out = stable_softmax(x, dim=-1)
    assert torch.isfinite(out).all(), "出现了 inf/nan —— 没有减去最大值"
    assert torch.allclose(out.sum(dim=-1), torch.ones(1), atol=1e-6), "数值不稳定"


def test_attention_shapes():
    torch.manual_seed(0)
    Q = torch.randn(2, 5, 8)
    K = torch.randn(2, 7, 8)
    V = torch.randn(2, 7, 16)
    out = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 5, 16), f"形状应为 (2, 5, 16)，实际为 {tuple(out.shape)}"


def test_attention_matches_reference():
    torch.manual_seed(0)
    Q = torch.randn(2, 6, 8)
    K = torch.randn(2, 6, 8)
    V = torch.randn(2, 6, 8)
    mine = scaled_dot_product_attention(Q, K, V)
    ref = _reference_attention(Q, K, V)
    assert torch.allclose(mine, ref, atol=1e-5), f"与官方实现不一致，最大误差 {(mine - ref).abs().max():.2e}"


def test_attention_with_mask_matches_reference():
    torch.manual_seed(0)
    L, d = 6, 8
    Q = torch.randn(1, L, d)
    K = torch.randn(1, L, d)
    V = torch.randn(1, L, d)
    mask = make_causal_mask(L)
    mine = scaled_dot_product_attention(Q, K, V, mask)
    ref = _reference_attention(Q, K, V, mask)
    assert torch.allclose(mine, ref, atol=1e-5), f"带掩码时与官方实现不一致，最大误差 {(mine - ref).abs().max():.2e}"


def test_causal_mask_shape():
    m = make_causal_mask(4)
    assert m.shape == (4, 4), f"形状应为 (4, 4)，实际为 {tuple(m.shape)}"
    assert m.dtype == torch.bool, f"dtype 应为 bool，实际为 {m.dtype}"
    expected = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    assert torch.equal(m, expected), f"内容不对：\n{m}"


def test_causal_mask_blocks_future():
    torch.manual_seed(0)
    L, d = 6, 8
    Q = torch.randn(1, L, d)
    K = torch.randn(1, L, d)
    V = torch.randn(1, L, d)
    mask = make_causal_mask(L)

    out1 = scaled_dot_product_attention(Q, K, V, mask)

    V2 = V.clone()
    V2[:, L - 1] += 100.0
    out2 = scaled_dot_product_attention(Q, K, V2, mask)

    assert torch.allclose(out1[:, :L - 1], out2[:, :L - 1], atol=1e-5), \
        "未来位置的信息泄漏到了过去 —— 掩码方向可能反了"
    assert not torch.allclose(out1[:, L - 1], out2[:, L - 1]), \
        "最后一个位置的输出应该发生变化，但没有"


def test_attention_weights_sum_to_one():
    torch.manual_seed(0)
    Q = torch.randn(1, 5, 8)
    K = torch.randn(1, 5, 8)
    V = torch.randn(1, 5, 8)
    out = scaled_dot_product_attention(Q, K, V)
    lo, hi = V.min().item(), V.max().item()
    assert out.min().item() >= lo - 1e-5 and out.max().item() <= hi + 1e-5, \
        f"输出超出了 V 的取值范围 [{lo:.3f}, {hi:.3f}] —— 说明不是凸组合"


TESTS = [
    ("softmax 与 torch.softmax 一致", test_softmax_matches_torch),
    ("softmax 每行之和为 1", test_softmax_sums_to_one),
    ("softmax 数值稳定（1e4 量级不溢出）", test_softmax_numerical_stability),
    ("注意力输出形状正确", test_attention_shapes),
    ("注意力与官方实现一致", test_attention_matches_reference),
    ("带掩码时与官方实现一致", test_attention_with_mask_matches_reference),
    ("因果掩码形状与内容正确", test_causal_mask_shape),
    ("因果掩码不泄漏未来信息", test_causal_mask_blocks_future),
    ("输出是 V 的凸组合", test_attention_weights_sum_to_one),
]


def run_tests():
    print("=" * 62)
    print("测试")
    print("=" * 62)
    passed = failed = todo = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  [通过]  {name}")
            passed += 1
        except (NotImplementedError, TypeError) as e:
            print(f"  [待填]  {name}   ({type(e).__name__})")
            todo += 1
        except AssertionError as e:
            print(f"  [失败]  {name}\n          {e}")
            failed += 1
        except Exception as e:
            print(f"  [异常]  {name}\n          {type(e).__name__}: {e}")
            failed += 1
    print("-" * 62)
    print(f"通过 {passed} / 失败 {failed} / 待填 {todo}   (共 {len(TESTS)} 项)")
    return failed, todo


QUIZ = [
    "为什么是除以 sqrt(d_k)，而不是除以 d_k？",
    "Q、K、V 三个矩阵各自的语义是什么？它们是怎么来的？",
    "掩码为什么填 -inf 而不是 0？填 0 会发生什么？",
    "softmax 为什么要先减最大值？这一步是近似还是恒等变换？",
    "L 和 S 分别代表什么？什么情况下 L != S？",
    "输出的物理意义是什么？为什么它一定落在 V 的凸包内？",
    "如果一整行都被掩码了，softmax 会输出什么？怎么处理？",
    "缩放的作用是「让梯度更大」吗？用 attention.py 的实验数据反驳或支持这句话。",
]


def main():
    print()
    failed, todo = run_tests()
    print()
    print("=" * 62)
    print("自测问题（能答上来才算真懂）")
    print("=" * 62)
    for i, q in enumerate(QUIZ, 1):
        print(f"  {i}. {q}")
    print()
    print("=" * 62)
    if failed == 0 and todo == 0:
        print("全部通过。把用时和结果记到文件顶部的「重写记录」里。")
    else:
        print(f"还有 {todo} 处待填、{failed} 项失败。")
    print("=" * 62)


if __name__ == "__main__":
    main()
