"""
W1 · 手写缩放点积注意力（Scaled Dot-Product Attention）

目标：不查资料能从头写出来，并且能讲清每一步在算什么。

用法：
    conda activate llm
    python w1_attention.py

你需要填 6 处：TODO 1（softmax）、TODO 2a/2b/2c/2d（注意力四步）、TODO 3（因果掩码）。

================================================================
怎么填空？
================================================================
每处都有一行长这样（`_todo(...)` 是一个占位符函数）：

    scores = _todo("2a")

**它不是要保留的东西，只是一个「这里还没写」的标记。**
把这一行整行删掉，换成你自己的代码。

⚠️ 替换完之后，那一行 `_todo(...)` 就不该再存在了。
如果实现写好了但占位符还留着，会变成**死代码** ——
Python 里 `return` / 赋值之后的语句永远不会执行，不影响结果，
但读代码的人（包括三个月后的你）会困惑，所以要删干净。

然后运行 `python attention.py`，看到 [通过] 就是对了。
在 VS Code 里按 Ctrl+F 搜索 **填空**，可以依次跳到剩下的每一处。
（填空 1/6 是 softmax，你已经写完了，标记已不需要。）

填完运行脚本，9 项测试全绿就算过关。

判过关的标准不是"测试通过"，而是：关掉这个文件，第二天能不看着提示重写一遍。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _todo(tag: str):
    """占位符：把 TODO 实现好之后，这一行就不会被触发了。"""
    raise NotImplementedError(f"TODO {tag} 还没实现")


# ==========================================================================
# TODO 1 · 数值稳定的 softmax
# ==========================================================================
def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim = dim,keepdim = True).values
    e = torch.exp(x-x_max)
    return e/e.sum(dim = dim,keepdim = True)
   
    return _todo("1")


# ==========================================================================
# TODO 2 · 缩放点积注意力
# ==========================================================================
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    参数形状（... 表示任意数量的前置维度，比如 batch 和 head）：
        Q:    (..., L, d_k)      L 个 query
        K:    (..., S, d_k)      S 个 key
        V:    (..., S, d_v)      S 个 value
        mask: None，或可广播到 (..., L, S) 的 bool 张量，True = 允许关注

    返回：
        (..., L, d_v)

    整个过程只有四步：缩放点积 → 掩码 → softmax → 加权求和。
    """
    d_k = Q.size(-1)

    # --- TODO 2a ---------------------------------------------------------
    # 计算未归一化的注意力分数 scores，形状 (..., L, S)
    #
    # 关键词：Q 与 K 的转置做矩阵乘，再除以 sqrt(d_k)
    # 为什么要除以 sqrt(d_k)？
    #   Q·K 是 d_k 个乘积之和，每个乘积的方差约为 1，
    #   所以点积的方差约为 d_k，标准差约为 sqrt(d_k)。
    #   d_k 一大，分数就分布得很开，softmax 会被推向 one-hot（饱和），
    #   大部分位置权重趋近 0，梯度也就趋近 0。
    #   除以 sqrt(d_k) 把方差拉回 1，softmax 保持在"有区分度但不极端"的区间。
    #   —— 运行本文件末尾的 experiment_scaling() 可以直接看到这个现象。

    # 填空 2/6 —— 把下一行替换成你的代码
    scores = (Q @ K.transpose(-1,-2))/math.sqrt(d_k)

    # --- TODO 2b ---------------------------------------------------------
    # 掩码：把 mask 为 False 的位置填成一个极小的数
    #
    # 为什么用 -inf 而不是 0？
    #   0 在 softmax 里代表"权重约为 1"，语义正好相反。
    #   填 -inf 之后，exp(-inf) = 0，softmax 自动给出 0 权重。
    #
    # 提示：scores.masked_fill(~mask, float("-inf"))
    #       注意 ~ 是按位取反，把 True/False 反过来。
    #       另外 mask 需要能广播到 scores 的形状（causal mask 通常是 (L, S)）。
    if mask is not None:
        # 填空 3/6 —— 把下一行替换成你的代码
        scores = scores.masked_fill(~mask,float("-inf"))

    # --- TODO 2c ---------------------------------------------------------
    # 对最后一维做 softmax，得到注意力权重 attn，形状 (..., L, S)
    # 提示：用你自己写的 stable_softmax
    # 填空 4/6 —— 把下一行替换成你的代码
    attn = stable_softmax(scores,-1)

    # --- TODO 2d ---------------------------------------------------------
    # 用注意力权重对 V 做加权求和，得到 (..., L, d_v)
    # 填空 5/6 —— 把下一行替换成你的代码
    out = attn @ V

    return out


# ==========================================================================
# TODO 3 · 因果掩码
# ==========================================================================
def make_causal_mask(seq_len: int) -> torch.Tensor:
    """
    构造因果掩码（causal mask），返回形状 (seq_len, seq_len) 的 bool 张量。
    语义：位置 i 只能看到 j <= i（不能偷看未来）。

    期望的样子（seq_len = 4）：
        [[ T, F, F, F],
         [ T, T, F, F],
         [ T, T, T, F],
         [ T, T, T, T]]

    提示：
        row = torch.arange(seq_len).view(-1, 1)   # 行下标 → query 的位置 i
        col = torch.arange(seq_len).view(1, -1)   # 列下标 → key 的位置 j
        然后对两者做比较，返回一个 (seq_len, seq_len) 的 bool 张量。
    """
    # 填空 6/6 —— 把下一行替换成你的代码
    row = torch.arange(seq_len).view(-1,1)
    col = torch.arange(seq_len).view(1,-1)
    return row >= col


# ==========================================================================
# 以下都是脚手架，不需要改动
# ==========================================================================

def _reference_attention(Q, K, V, mask=None):
    """官方实现，用来对照你自己写的结果。"""
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
    """输入量级到 1e4 时，直接 exp 会溢出成 nan。"""
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
    """最关键的一条：改动未来的 V，不应影响过去位置的输出。"""
    torch.manual_seed(0)
    L, d = 6, 8
    Q = torch.randn(1, L, d)
    K = torch.randn(1, L, d)
    V = torch.randn(1, L, d)
    mask = make_causal_mask(L)

    out1 = scaled_dot_product_attention(Q, K, V, mask)

    V2 = V.clone()
    V2[:, L - 1] += 100.0                      # 只改最后一个位置
    out2 = scaled_dot_product_attention(Q, K, V2, mask)

    assert torch.allclose(out1[:, :L - 1], out2[:, :L - 1], atol=1e-5), \
        "未来位置的信息泄漏到了过去 —— 掩码方向可能反了（应该屏蔽 j > i）"
    assert not torch.allclose(out1[:, L - 1], out2[:, L - 1]), \
        "最后一个位置的输出应该发生变化，但没有 —— 掩码把该看的位置也挡住了"


def test_attention_weights_sum_to_one():
    """未掩码时，每一行注意力权重之和应为 1；输出必须是 V 的凸组合。"""
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


def run_tests() -> int:
    print("=" * 66)
    print("测试")
    print("=" * 66)
    passed = failed = todo = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  [通过]  {name}")
            passed += 1
        except NotImplementedError as e:
            print(f"  [待填]  {name}   ({e})")
            todo += 1
        except AssertionError as e:
            print(f"  [失败]  {name}\n          {e}")
            failed += 1
        except Exception as e:
            print(f"  [异常]  {name}\n          {type(e).__name__}: {e}")
            failed += 1
    print("-" * 66)
    print(f"通过 {passed} / 失败 {failed} / 待填 {todo}   (共 {len(TESTS)} 项)")
    return failed, todo


def experiment_scaling():
    """
    实验：为什么必须除以 sqrt(d_k)？

    记录方式：把下面的输出粘到你的实验记录里，写一句话结论。
    """
    print()
    print("=" * 66)
    print("实验：缩放对 softmax 饱和度与梯度的影响")
    print("=" * 66)
    print("Jacobian 范数 = ||diag(p) - p p^T||_F，衡量 softmax 的梯度通量；")
    print("越接近 0 说明越饱和，反向传播时梯度越难传回去。")
    print()

    torch.manual_seed(0)
    n = 64
    for d_k in [16, 64, 256, 1024]:
        Q = torch.randn(1, n, d_k)
        K = torch.randn(1, n, d_k)
        raw = Q @ K.transpose(-2, -1)
        scaled = raw / math.sqrt(d_k)

        print(f"d_k = {d_k}")
        for label, logits in (("未缩放", raw), ("已缩放", scaled)):
            p = torch.softmax(logits[0, 0], dim=-1)
            jac = (torch.diag(p) - torch.outer(p, p)).norm().item()
            print(f"    {label}   logits.std={logits.std().item():7.2f}   "
                  f"最大权重={p.max().item():.4f}   Jacobian 范数={jac:.4f}")
        print()

    print("结论提示：")
    print("  1. 未缩放时，logits 的标准差随 d_k 增大（约等于 sqrt(d_k)），")
    print("     softmax 迅速饱和成 one-hot：Jacobian 范数从 0.33 一路掉到 0.0001，")
    print("     注意力权重几乎不再随输入变化，梯度传不回去。")
    print("  2. 除以 sqrt(d_k) 后，logits.std 稳定在 1.0 附近，")
    print("     Jacobian 范数稳定在 0.17 附近，与 d_k 几乎无关。")
    print()
    print("  【注意一个反直觉的点】")
    print("  d_k = 16 时，缩放后的 Jacobian（0.17）反而比不缩放（0.33）更小。")
    print("  所以缩放的真正作用不是「让梯度更大」，而是「让行为不随 d_k 崩溃」。")
    print("  做实验时如果只看单个数据点，很容易得出相反的错误结论 ——")
    print("  这也是为什么实验设计里「跨条件对比」比「单点观测」重要。")


def quiz():
    print()
    print("=" * 66)
    print("自测问题（能答上来才算过关，答不上就回去重看代码）")
    print("=" * 66)
    for i, q in enumerate([
        "为什么是除以 sqrt(d_k)，而不是除以 d_k？",
        "Q、K、V 三个矩阵各自的语义是什么？它们是怎么来的？",
        "掩码为什么填 -inf 而不是 0？填 0 会发生什么？",
        "softmax 为什么要先减最大值？不减会怎样？",
        "L 和 S 分别代表什么？什么情况下 L != S？",
        "输出的物理意义是什么？为什么它一定落在 V 的凸包内？",
        "如果一整行都被掩码了，softmax 会输出什么？怎么处理？",
        "缩放的作用是「让梯度更大」吗？用上面的实验数据反驳或支持这句话。",
    ], 1):
        print(f"  {i}. {q}")
    print()
    print("提示：第 7 题是个真实的工程坑 —— 训练时如果有 padding，")
    print("      整行被 mask 掉会产生 nan，实际代码里通常会加一个极小的 epsilon。")


def main():
    print()
    failed, todo = run_tests()
    experiment_scaling()
    quiz()
    print()
    print("=" * 66)
    if failed == 0 and todo == 0:
        print("全部通过。下一步：合上提示，从空白文件重写一遍。")
    else:
        print(f"还有 {todo} 处待填、{failed} 项失败。改完再运行一次即可。")
    print("=" * 66)


if __name__ == "__main__":
    main()
