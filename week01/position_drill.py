"""
位置编码：三个可验证的事实
========================================================================

背景：你在论文摘要里写了「位置编码没有看懂」。那是我唯一看到你明确说不会的点，
      所以这个文件专门解决它。

    你要做的：填 1 个空（实现论文 §3.5 的公式），然后跑三个验证。

过关标准：
    三个验证全过，并且你能回答文件末尾的 5 个自测题。
========================================================================
"""

from __future__ import annotations

import math

import torch

# 复用你 W1 写的注意力实现（在同一个目录里）
from attention import make_causal_mask, scaled_dot_product_attention


def _todo(where: str):
    raise NotImplementedError(f"{where} 还没填")


# ==========================================================================
# 1 · 论文 §3.5 的位置编码
# ==========================================================================
def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """返回 (seq_len, d_model) 的位置编码矩阵。

    论文的公式（d_model 必须是偶数）：

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    对应的代码思路（位置 pos 是行，维度 2i / 2i+1 是列）：

        pos   = arange(seq_len).view(-1, 1)          # (seq_len, 1)
        i     = arange(0, d_model, 2)                # 0, 2, 4, ... 偶数维度下标
        omega = exp(-log(10000) * i / d_model)       # 每个维度对的频率 1/10000^(2i/d)
        pe[:, 0::2] = sin(pos * omega)
        pe[:, 1::2] = cos(pos * omega)
    """
    assert d_model % 2 == 0, "d_model 必须是偶数"
    # 填空 1/1 —— 把下面这行替换成你的实现
    pe = _todo("positional_encoding")
    return pe


# ==========================================================================
# 验证 1 · 没有位置编码时，注意力分不清词序
# ==========================================================================
def check_1_permutation():
    print()
    print("=" * 72)
    print("验证 1 · 没有位置编码时，注意力「分不清」词序")
    print("=" * 72)

    torch.manual_seed(0)
    L, d = 4, 8
    X = torch.randn(1, L, d)

    # 打乱词序：0,1,2,3 → 2,0,3,1
    perm = [2, 0, 3, 1]
    X_shuffled = X[:, perm, :]

    out = scaled_dot_product_attention(X, X, X)
    out_shuffled = scaled_dot_product_attention(X_shuffled, X_shuffled, X_shuffled)

    print(f"  原输入 X          : shape {tuple(X.shape)}")
    print(f"  打乱后 X_shuffled : 词序变成 {perm}")
    print()
    print(f"  out            = {[[round(v, 4) for v in row] for row in out[0, :, :3].tolist()]}")
    print(f"  out_shuffled   = {[[round(v, 4) for v in row] for row in out_shuffled[0, :, :3].tolist()]}")
    print()
    print("  关键：把 out 按 perm **正向重排**，能不能拿回 out_shuffled？")
    print("        （因为打乱后第 i 个位置，对应的是原序列的第 perm[i] 个位置）")

    gather = out[:, perm, :]
    diff = (gather - out_shuffled).abs().max().item()
    print(f"  重排后最大误差     : {diff:.2e}")
    print()
    if diff < 1e-6:
        print("  结论：误差是浮点量级 —— **两个结果本质上是同一个东西**。")
        print("        也就是说：纯粹的自注意力对词的顺序「不敏感」，")
        print("        '我打你' 和 '你打我' 在它眼里完全一样。")
        print("        → 这就是位置编码存在的理由：必须人为把位置信息注进去。")
    else:
        print("  结论：⚠️ 误差偏大，检查一下重排逻辑")


# ==========================================================================
# 验证 2 · PE(pos + k) 是 PE(pos) 的线性变换（这是 sin/cos 的精髓）
# ==========================================================================
def check_2_linear_shift():
    print()
    print("=" * 72)
    print("验证 2 · PE(pos+k) 可以由 PE(pos) 经过一个「固定的旋转」得到")
    print("=" * 72)

    d_model = 16
    seq_len = 32
    pe = positional_encoding(seq_len, d_model)

    k = 5          # 位置偏移
    # 取第 0 行和第 k 行
    pe_pos = pe[0]              # PE(0)
    pe_pos_k = pe[k]            # PE(k)

    # 逐个 2 维块验证：对第 i 组 (sin, cos)，偏移 k 相当于旋转 k*omega_i
    print(f"  d_model = {d_model}，偏移 k = {k}")
    print()
    print("  ┌────────┬──────────────┬──────────────┬──────────────┐")
    print("  │ 维度对 │ 角频率 omega │ 实测旋转角   │ 期望 k*omega │")
    print("  ├────────┼──────────────┼──────────────┼──────────────┤")

    errors = []
    for i in range(0, d_model, 2):
        omega = 1.0 / (10000 ** (i / d_model))
        s0, c0 = pe_pos[i].item(), pe_pos[i + 1].item()
        sk, ck = pe_pos_k[i].item(), pe_pos_k[i + 1].item()

        # 从 (sin, cos) 对反推旋转角
        ang0 = math.atan2(s0, c0)
        angk = math.atan2(sk, ck)
        measured = angk - ang0
        expected = k * omega
        # 角度归一化到 (-pi, pi]
        expected_wrapped = (expected + math.pi) % (2 * math.pi) - math.pi
        errors.append(abs(measured - expected_wrapped))

        if i < 6:
            print(f"  │   {i:2d}   │   {omega:.6f}  │   {measured:+.6f}  │   {expected_wrapped:+.6f}  │")

    print("  └────────┴──────────────┴──────────────┴──────────────┘")
    print(f"\n  全部 {d_model // 2} 个维度对的最大角度误差: {max(errors):.2e}")

    print()
    if max(errors) < 1e-4:
        print("  结论：对每一个维度对，偏移 k 个位置 = 旋转 k·ω 角度。")
        print("        旋转是**线性变换**（矩阵乘），所以 PE(pos+k) = PE(pos) @ M_k，")
        print("        其中 M_k 只跟 k 有关、跟 pos 无关。")
        print()
        print("        **这就是用 sin/cos 而不用 0,1,2,3 的真正原因：**")
        print("        模型只要学会一个线性变换，就能表达「相对位置 k」这个关系。")
        print("        而如果直接给编号 pos，模型得自己去猜「差值」的含义 ——")
        print("        它得从 999 和 1000 里看出「差 1」，这件事对神经网络很难。")
    else:
        print("  结论：⚠️ 误差偏大，检查 positional_encoding 的实现")


# ==========================================================================
# 验证 3 · 为什么不用「0, 1, 2, ...」直接当位置
# ==========================================================================
def check_3_value_range():
    print()
    print("=" * 72)
    print("验证 3 · 为什么不用 0,1,2,... 直接当位置？")
    print("=" * 72)

    for seq_len in (16, 128, 1024):
        idx = torch.arange(seq_len, dtype=torch.float32)
        pe = positional_encoding(seq_len, 16)
        print(f"  序列长度 {seq_len:5d} │ 直接编号的值域 [{idx.min():.0f}, {idx.max():.0f}]"
              f"  │ PE 的值域 [{pe.min():.3f}, {pe.max():.3f}]")

    print()
    print("  结论（两个问题）：")
    print("    ① **值域无界**：长度 1024 时编号最大到 1023，而 PE 永远在 [-1, 1]。")
    print("       无界输入会让数值不稳定，而且它跟 embedding 相加时量级严重不匹配")
    print("       （embedding 每个分量大约在 ±1 附近）。")
    print("    ② **不能外推**：模型训练时只见过 0~511，遇到位置 512 就是全新的输入。")
    print("       而 sin/cos 是周期函数，公式对任意 pos 都成立。")
    print()
    print("  另外注意论文写的是**加法**（PE 直接加到 embedding 上）而不是拼接 ——")
    print("  因为维度相同（都是 d_model），相加不用改变后续层的宽度。")
    print("  代价是位置信息和词义信息混在同一个向量里，无法再分开。")


# ==========================================================================
# 自测
# ==========================================================================
def quiz():
    print()
    print("=" * 72)
    print("自测题（能答上来才算懂）")
    print("=" * 72)
    qs = [
        "为什么自注意力必须配位置编码？用「我打你 / 你打我」举例说明。",
        "论文的 PE 公式里，为什么要 sin 和 cos 各占一半维度，而不是全用 sin？",
        "PE(pos+k) 和 PE(pos) 是什么关系？这个关系为什么重要？",
        "为什么位置编码用「加法」而不是「拼接」？代价是什么？",
        "如果把位置编码换成 0,1,2,3... 直接当输入，会出现哪两个具体问题？",
    ]
    for i, q in enumerate(qs, 1):
        print(f"  {i}. {q}")
    print()
    print("  第 2 题提示：只给 sin 的话，sin(θ) 和 sin(π-θ) 是同一个值 —— 位置会撞车。")


def main():
    check_1_permutation()
    check_2_linear_shift()
    check_3_value_range()
    quiz()
    print()
    print("=" * 72)
    print("三个验证都过了、5 题都答得出来 → 位置编码这一块就结了。")
    print("明天 W2 手写 GPT 时，你会自己把它实现一遍。")
    print("=" * 72)


if __name__ == "__main__":
    main()
