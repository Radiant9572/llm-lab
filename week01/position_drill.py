"""
位置编码：三个可验证的事实
========================================================================

背景：你在论文摘要里写了「位置编码没有看懂」。这个文件专门解决它。

**这个文件不需要你写任何代码。** 直接跑：

    cd E:\\llm-lab\\week01
    python position_drill.py

它会用一份参考实现跑三个实验，把结论直接摆给你看。

（上一版我把实现留成填空了 —— 那是设计错误：**不该用"写代码"挡住"看现象"。**
 先看懂现象，再决定要不要自己写。想挑战的话，文件最后有个可选的对照练习。）
========================================================================
"""

from __future__ import annotations

import math

import torch

# 复用你 W1 写的注意力实现
from attention import scaled_dot_product_attention


# ==========================================================================
# 参考实现：论文 §3.5 的公式（不用你写，看懂就行）
# ==========================================================================
def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """返回 (seq_len, d_model) 的位置编码矩阵。

    论文公式（d_model 为偶数）：
        PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

    下面四行就是它的代码化，**逐行对应**：

        pos   = arange(seq_len)                     # 位置 0,1,2,...,L-1
        omega = 1 / 10000^(2i/d_model)              # 每个维度对的「角频率」
        pe[:, 0::2] = sin(pos * omega)              # 偶数维放 sin
        pe[:, 1::2] = cos(pos * omega)              # 奇数维放 cos

    ⚠️ 注意第 2 行：i 越大，omega 越小 —— 也就是**低维转得快、高维转得慢**。
       这样不同维度覆盖了从「每个位置都变」到「几万个位置才转一圈」的不同尺度。
       一个位置向量里同时包含粗细两种刻度，模型才能分辨近处和远处的差别。
    """
    assert d_model % 2 == 0, "d_model 必须是偶数"

    pos = torch.arange(seq_len, dtype=torch.float32).view(-1, 1)      # (L, 1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)              # (d/2,)
    omega = torch.exp(-math.log(10000.0) * i / d_model)               # (d/2,)

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * omega)   # (L,1)*(d/2,) 广播成 (L, d/2)
    pe[:, 1::2] = torch.cos(pos * omega)
    return pe


# ==========================================================================
# 验证 1 · 没有位置编码时，注意力「分不清」词序
# ==========================================================================
def check_1_permutation():
    print()
    print("=" * 72)
    print("验证 1 · 没有位置编码时，注意力「分不清」词序")
    print("=" * 72)

    torch.manual_seed(0)
    L, d = 4, 8
    X = torch.randn(1, L, d)

    perm = [2, 0, 3, 1]                 # 打乱词序：0,1,2,3 → 2,0,3,1
    X_shuffled = X[:, perm, :]

    out = scaled_dot_product_attention(X, X, X)
    out_shuffled = scaled_dot_product_attention(X_shuffled, X_shuffled, X_shuffled)

    print(f"  原序列 X       : 4 个词，每个 8 维")
    print(f"  打乱后         : 词序变成 {perm}")
    print()
    print("  out 的前 3 维（4 个位置各一行）：")
    for i, row in enumerate(out[0, :, :3].tolist()):
        print(f"    位置 {i}: {[round(v, 4) for v in row]}")
    print("  out_shuffled 的前 3 维：")
    for i, row in enumerate(out_shuffled[0, :, :3].tolist()):
        print(f"    位置 {i}: {[round(v, 4) for v in row]}")

    gather = out[:, perm, :]            # 打乱后第 i 个位置 ← 原序列第 perm[i] 个位置
    diff = (gather - out_shuffled).abs().max().item()

    print()
    print(f"  把 out 按 perm 重排后再比 → 最大误差: {diff:.2e}")
    print()
    print("  结论：误差是浮点量级（1e-7），**两个结果本质上是同一个东西**。")
    print("        纯自注意力对词的顺序「完全不敏感」——")
    print("        '我打你' 和 '你打我' 在它眼里一模一样的。")
    print("        → 这就是位置编码存在的唯一理由：必须人为把位置信息注进去。")


# ==========================================================================
# 验证 2 · PE(pos+k) 是 PE(pos) 的「固定旋转」（sin/cos 的精髓）
# ==========================================================================
def check_2_linear_shift():
    print()
    print("=" * 72)
    print("验证 2 · PE(pos+k) = PE(pos) 经过一个「只跟 k 有关的旋转」")
    print("=" * 72)

    d_model, seq_len, k = 16, 32, 5
    pe = positional_encoding(seq_len, d_model)

    pe_pos, pe_pos_k = pe[0], pe[k]     # PE(0) 和 PE(k)

    print(f"  d_model = {d_model}，位置偏移 k = {k}")
    print(f"  PE(0) 的前 4 维: {[round(v, 4) for v in pe_pos[:4].tolist()]}")
    print(f"  PE({k}) 的前 4 维: {[round(v, 4) for v in pe_pos_k[:4].tolist()]}")
    print()
    print("  每个 (sin, cos) 维度对，本质上是一个单位圆上的点。")
    print("  位置从 0 挪到 k，就是这个点旋转了一个角度。实测：")
    print()
    print("  ┌────────┬──────────────┬──────────────┬──────────────┐")
    print("  │ 维度对 │ 角频率 omega │ 实测旋转角   │ 期望 k*omega │")
    print("  ├────────┼──────────────┼──────────────┼──────────────┤")

    errors = []
    for idx in range(0, d_model, 2):
        omega = 1.0 / (10000 ** (idx / d_model))
        ang0 = math.atan2(pe_pos[idx].item(), pe_pos[idx + 1].item())
        angk = math.atan2(pe_pos_k[idx].item(), pe_pos_k[idx + 1].item())
        measured = angk - ang0
        expected = (k * omega + math.pi) % (2 * math.pi) - math.pi
        errors.append(abs(measured - expected))
        if idx < 6:
            print(f"  │   {idx:2d}   │   {omega:.6f}  │   {measured:+.6f}  │   {expected:+.6f}  │")
    print("  └────────┴──────────────┴──────────────┴──────────────┘")
    print(f"\n  全部 {d_model // 2} 个维度对的最大角度误差: {max(errors):.2e}")

    print()
    print("  结论：**偏移 k 个位置 = 旋转 k·ω 角度**，而旋转是线性变换（矩阵乘）。")
    print("        所以 PE(pos+k) = PE(pos) @ M_k，其中 M_k 只跟 k 有关、跟 pos 无关。")
    print()
    print("  ★ 这就是用 sin/cos 而不用 0,1,2,3 的真正原因：")
    print("    模型只要学一个线性变换，就能表达「相对位置 k」这个关系，")
    print("    而且这个变换不随 pos 变化 —— 学一次就能到处用。")
    print("    如果直接给编号，模型得从 999 和 1000 里自己发现「差 1」，")
    print("    这件事对神经网络非常难。")


# ==========================================================================
# 验证 3 · 为什么不用「0,1,2,...」直接当位置
# ==========================================================================
def check_3_value_range():
    print()
    print("=" * 72)
    print("验证 3 · 为什么不用 0,1,2,... 直接当位置？")
    print("=" * 72)
    print()
    for seq_len in (16, 128, 1024):
        idx = torch.arange(seq_len, dtype=torch.float32)
        pe = positional_encoding(seq_len, 16)
        print(f"  序列长度 {seq_len:5d} │ 直接编号值域 [{idx.min():.0f}, {idx.max():.0f}]"
              f" │ PE 值域 [{pe.min():.3f}, {pe.max():.3f}]")

    print()
    print("  两个具体问题：")
    print("    ① **值域无界**：长度 1024 时编号到 1023，而 PE 永远在 [-1, 1]。")
    print("       词嵌入每个分量大约在 ±1 附近，把一个上千的数直接加进去，")
    print("       量级完全不匹配，训练会不稳。")
    print("    ② **不能外推**：训练时只见过位置 0~511，遇到 512 就是全新输入。")
    print("       而 sin/cos 是固定公式，对任意 pos 都成立。")
    print()
    print("  还有一个更隐蔽的点：即使把编号归一化到 [0,1]，模型仍得自己推出")
    print("  「0.999 和 1.0 差一个位置」。差值信息藏在两个数里，很难学。")
    print("  这正是验证 2 那个旋转性质要解决的问题。")


# ==========================================================================
# 可选挑战 · 自己写一遍，和参考实现对照
# ==========================================================================
def positional_encoding_mine(seq_len: int, d_model: int) -> torch.Tensor:
    """想挑战的话在这里自己写一遍（不看上面的参考实现）。

    写完运行 compare_with_reference() 会自动对照。
    不想写就跳过 —— **今晚不需要写代码。**
    """
    raise NotImplementedError("（可选）自己实现位置编码")


def compare_with_reference():
    print()
    print("=" * 72)
    print("可选挑战 · 你的实现 vs 参考实现")
    print("=" * 72)
    try:
        mine = positional_encoding_mine(32, 16)
    except NotImplementedError:
        print("  还没写（没关系，今晚不用写）。")
        print("  想练的话：回到文件顶部看那 4 行「逐行对应」，然后自己打一遍。")
        return
    ref = positional_encoding(32, 16)
    diff = (mine - ref).abs().max().item()
    print(f"  最大误差: {diff:.2e}")
    print("  ✓ 完全一致，位置编码你会写了。" if diff < 1e-6 else "  ✗ 和参考实现不一致，检查维度和 sin/cos 的位置。")


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
        "论文的 PE 公式里，sin 和 cos 为什么各占一半维度，而不是全用 sin？",
        "PE(pos+k) 和 PE(pos) 是什么关系？这个关系为什么重要？",
        "位置编码用「加法」而不是「拼接」，为什么？代价是什么？",
        "为什么低维的 omega 大、高维的 omega 小？这样设计有什么好处？",
    ]
    for i, q in enumerate(qs, 1):
        print(f"  {i}. {q}")
    print()
    print("  第 2 题提示：只给 sin 的话，sin(θ) 和 sin(π−θ) 是同一个值 —— 位置会撞车。")
    print("  第 5 题提示：见 `positional_encoding` 的 docstring 里的 ⚠️ 那一段。")


def main():
    check_1_permutation()
    check_2_linear_shift()
    check_3_value_range()
    compare_with_reference()
    quiz()
    print()
    print("=" * 72)
    print("三个验证看完、5 题答得出来 → 位置编码这一块就结了。")
    print("明天 W2 手写 GPT 时，你会把它实现进模型里。")
    print("=" * 72)


if __name__ == "__main__":
    main()
