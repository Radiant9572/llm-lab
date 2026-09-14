"""
W2 · 手写一个完整的小型 GPT（GPT-2 风格）
========================================================================

验收标准（就两条）：
    ① 前向传播 shape 全部正确
    ② **参数量与理论值对得上** —— 而且要能自己手算出那个理论值

========================================================================
架构全景（先看懂这张图，再往下填）
========================================================================

一层 Block 里发生的事：

        x
        │
        ├──────────────┐          ← 残差：把输入留一份
        ▼              │
     LayerNorm         │
        │              │
   MultiHeadAttention  │
        │              │
        ▼              │
       (+) ◄───────────┘          ← 相加
        │
        ├──────────────┐          ← 又一次残差
        ▼              │
     LayerNorm         │
        │              │
        FeedForward    │          ← 这一层的参数最多（约占 2/3）
        │              │
        ▼              │
       (+) ◄───────────┘
        │
        x_out

整个模型：

    token_ids ──► Token Embedding ──┐
                                    (+) ──► [Block] × n_layer ──► LayerNorm ──► LM Head ──► logits
    positions ──► Pos Embedding ────┘

**为什么要有残差连接**：它让梯度有一条"高速公路"直接从底层流到顶层。
没有它，堆到十几层就训不动了。这就是"能训深"的关键。

**为什么 LayerNorm 放在前面（Pre-LN）**：原论文是 Post-LN（LN 在残差相加之后），
但后来的实践发现 Pre-LN（LN 在子层之前）训练更稳、更不容易散。GPT-2 用的就是 Pre-LN。
**这个改动本身就是一次"架构消融"的结果** —— 你 W4 要做的就是这类实验。

========================================================================
怎么用：
    9 处填空，都有 `# 填空 N/9` 标记。填完跑：

        cd E:\\llm-lab\\week02
        python mini_gpt.py

    测试全绿就过关。判过关的标准不是"绿了"，是**明天能从空白重写一遍**。
========================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _todo(where: str):
    raise NotImplementedError(f"{where} 还没填")


# ==========================================================================
# 配置
# ==========================================================================
@dataclass
class GPTConfig:
    vocab_size: int = 65      # 字符级词表大小（莎士比亚数据集就是这个量级）
    ctx_len: int = 32         # 上下文长度（能看多长的历史）
    d_model: int = 64         # 隐藏维度 C
    n_head: int = 4
    n_layer: int = 2
    ffn_mult: int = 4         # FFN 隐层宽度 = ffn_mult × d_model
    dropout: float = 0.0


# ==========================================================================
# 1 · 多头自注意力
# ==========================================================================
class MultiHeadAttention(nn.Module):
    """多头自注意力。

    形状走向（记住这几步，填空就靠它）：
        x        (B, L, C)
        qkv      (B, L, 3C)        ← 一次线性投影算出 Q、K、V
        拆成三份 (B, L, C) × 3
        拆头      (B, nh, L, hd)   ← C = nh × hd，把 C 拆成"头数 × 每头维度"
        打分      (B, nh, L, L)
        softmax   (B, nh, L, L)
        加权      (B, nh, L, hd)
        合头      (B, L, C)
        输出投影  (B, L, C)

    为什么要"拆头"：让不同的头在不同的子空间里各自关注不同的东西
    （有的头学语法、有的头学指代……）。总维度不变，只是分成几组并行算。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model 必须能被 n_head 整除"
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model

        # 填空 1/9 —— 定义两个线性层：
        #   ① qkv 投影：把 C 维一次投影成 3C 维（Q、K、V 一起算，比开三个层快）
        #   ② 输出投影：C → C
        # 提示：nn.Linear(in_features, out_features)，默认带偏置
        self.qkv = _todo("1")
        self.out_proj = _todo("1b")

        # 因果掩码：下三角为 True，表示"位置 i 只能看到 j <= i"
        # 用 register_buffer 而不是普通属性 —— 这样它会跟着模型搬到 GPU，但不会被当成参数
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        # ---- ① 一次投影出 Q、K、V ----------------------------------------
        qkv = self.qkv(x)                                 # (B, L, 3C)
        q, k, v = qkv.split(C, dim=-1)                     # 各 (B, L, C)

        # ---- ② 拆头：(B, L, C) → (B, nh, L, hd) ---------------------------
        # 先把最后一维 C 拆成 (nh, hd)，再把 L 和 nh 换位置
        # 填空 2/9
        q = _todo("2")
        k = _todo("2")
        v = _todo("2")

        # ---- ③ 缩放点积 + 掩码 + softmax ---------------------------------
        # 这一步和你 W1 写的完全一样，只是多了"头"这一维（当作额外的 batch 维）
        # 填空 3/9：算 scores（要除以 sqrt(head_dim)）
        scores = _todo("3")
        # 填空 4/9：用 self.mask 把未来位置盖掉（填 -inf）
        #   ⚠️ self.mask 是 (L, L)，scores 是 (B, nh, L, L) —— 广播会自动对齐
        scores = _todo("4")
        attn = F.softmax(scores, dim=-1)

        # ---- ④ 加权求和 + 合头 -------------------------------------------
        out = attn @ v                                     # (B, nh, L, hd)
        # 填空 5/9：合头 → (B, L, C)
        #   思路：先 transpose 回 (B, L, nh, hd)，再 reshape 成 (B, L, C)
        #   ⚠️ transpose 之后内存不连续，必须 .contiguous() 才能 .view()
        out = _todo("5")

        # ---- ⑤ 输出投影 ---------------------------------------------------
        return self.out_proj(out)


# ==========================================================================
# 2 · 前馈网络
# ==========================================================================
class FeedForward(nn.Module):
    """两层 MLP：C → ffn_mult*C → C。

    **注意：这是整个 Block 里参数最多的地方。**
    d_model = C 时，两层加起来是 8C² + 5C，而整个注意力部分只有 4C² + 4C。
    所以"Transformer 的参数大部分在前馈层" —— 这个结论你会在实验里亲眼看到。

    位置编码/注意力管的是"信息怎么流动"，FFN 管的是"每个位置各自算什么"。
    """

    def __init__(self, cfg: GPTConfig):
        hidden = cfg.ffn_mult * cfg.d_model
        super().__init__()
        # 填空 6/9 —— 定义两层线性层
        self.fc1 = _todo("6")
        self.fc2 = _todo("6b")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 填空 7/9 —— 两层的顺序是什么？中间要过什么激活函数？
        # 提示：GPT-2 用的是 GELU；中间的激活是逐元素的，不改变形状
        return _todo("7")


# ==========================================================================
# 3 · Block
# ==========================================================================
class Block(nn.Module):
    """Pre-LN 的 Transformer Block：LN → 子层 → 残差相加，重复两次。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 填空 8/9 —— 两次「残差 + Pre-LN」
        # 形状要求：输入输出都是 (B, L, C)，所以残差是直接相加
        # 提示：注意 LN 放在子层**之前**（Pre-LN），别写反了
        x = _todo("8a")
        x = _todo("8b")
        return x


# ==========================================================================
# 4 · 完整模型
# ==========================================================================
class MiniGPT(nn.Module):
    """字符级 GPT：Embedding → N×Block → LN → LM Head。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # 权重共享（weight tying）：让输出层和输入嵌入用同一张表。
        # 理由：都是"词表 ↔ 向量"的映射，共用能省下一大块参数（V×C 个），
        #       而且实测效果更好。GPT-2 就是这么做的。
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, L) 的整数张量（token id）
        返回: (B, L, vocab_size) 的 logits
        """
        B, L = idx.shape
        assert L <= self.cfg.ctx_len, f"序列长度 {L} 超过 ctx_len {self.cfg.ctx_len}"

        # 填空 9/9 —— 词嵌入 + 位置嵌入，然后过 blocks、ln_f、lm_head
        #   ① 位置编号怎么造？提示：torch.arange(L)
        #   ② 两个 embedding 是**相加**（不是拼接），因为维度都是 C
        #   ③ 别忘了最后过 self.ln_f
        tok = _todo("9a")
        pos = _todo("9b")
        x = _todo("9c")
        for block in self.blocks:
            x = block(x)
        x = _todo("9d")
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 20) -> torch.Tensor:
        """贪心生成（W3 会改成带 temperature / top-k 的采样）。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.ctx_len:]
            logits = self(idx_cond)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ==========================================================================
# 5 · 理论参数量公式（这是验收标准的核心）
# ==========================================================================
def theoretical_parameter_count(cfg: GPTConfig) -> int:
    """返回模型的**理论参数量**（一个整数）。

    ★ 要求你自己推导出来，不许直接 `sum(p.numel())`。
      能推出来，才说明你知道模型里到底有什么。

    推导用的零件清单（按本文件的实现）：

        记号：V = vocab_size，T = ctx_len，C = d_model，H = ffn_mult*C，L = n_layer

        ┌─ 嵌入部分 ────────────────────────────────┐
        │ Token Embedding     : V × C               │
        │ Position Embedding  : T × C               │
        └───────────────────────────────────────────┘
        ┌─ 每个 Block（共 L 个）─────────────────────┐
        │ LayerNorm × 2       : 2 × 2C              │  权重 C + 偏置 C，两个 LN
        │ qkv 投影            : C×3C + 3C           │  权重 + 偏置
        │ 输出投影            : C×C  + C            │
        │ FFN 第一层          : C×H  + H            │
        │ FFN 第二层          : H×C  + C            │
        └───────────────────────────────────────────┘
        ┌─ 收尾 ────────────────────────────────────┐
        │ 最后的 LayerNorm    : 2C                  │
        │ LM Head             : **0**               │  ← 和 Token Embedding 共享权重
        └───────────────────────────────────────────┘

        合计 = V×C + T×C + L×(上面那个 Block 的和) + 2C

    先用笔算出来，再写成一个公式（用 cfg 里的字段）。
    """
    raise NotImplementedError("theoretical_parameter_count 还没填")


# ==========================================================================
# 测试（不需要改动）
# ==========================================================================
CFG = GPTConfig()


def _make_model():
    torch.manual_seed(0)
    return MiniGPT(CFG)


def test_shapes():
    m = _make_model()
    B, L = 2, CFG.ctx_len
    idx = torch.randint(0, CFG.vocab_size, (B, L))
    out = m(idx)
    assert out.shape == (B, L, CFG.vocab_size), f"logits 形状应为 {(B, L, CFG.vocab_size)}，实际 {tuple(out.shape)}"


def test_attention_shape():
    m = MultiHeadAttention(CFG)
    x = torch.randn(2, 8, CFG.d_model)
    y = m(x)
    assert y.shape == x.shape, f"注意力输出应为 {tuple(x.shape)}，实际 {tuple(y.shape)}"


def test_ffn_shape():
    m = FeedForward(CFG)
    x = torch.randn(2, 8, CFG.d_model)
    y = m(x)
    assert y.shape == x.shape, f"FFN 输出应为 {tuple(x.shape)}，实际 {tuple(y.shape)}"


def test_block_shape():
    m = Block(CFG)
    x = torch.randn(2, 8, CFG.d_model)
    y = m(x)
    assert y.shape == x.shape, f"Block 输出应为 {tuple(x.shape)}，实际 {tuple(y.shape)}"


def test_causality():
    """★ 因果性：改动最后一个 token，**前面位置**的 logits 不能有任何变化。

    如果你忘了加因果掩码，这个测试会失败 —— 而且这是最容易漏、最难自己发现的一步。
    """
    m = _make_model()
    m.eval()
    L = 8
    a = torch.randint(0, CFG.vocab_size, (1, L))
    b = a.clone()
    b[0, -1] = (a[0, -1] + 7) % CFG.vocab_size          # 只改最后一个 token

    with torch.no_grad():
        la, lb = m(a), m(b)

    diff_prefix = (la[:, :-1] - lb[:, :-1]).abs().max().item()
    diff_last = (la[:, -1] - lb[:, -1]).abs().max().item()

    assert diff_prefix < 1e-6, (
        f"前面位置的 logits 变了（最大差 {diff_prefix:.2e}）—— 因果掩码没起作用！"
    )
    assert diff_last > 0, "最后一个位置的 logits 应该会变（说明测试本身有效）"


def test_parameter_count():
    """★ 验收标准：模型实际参数量 == 你推导出的理论值。"""
    m = _make_model()
    actual = sum(p.numel() for p in m.parameters())
    expected = theoretical_parameter_count(CFG)
    assert expected == actual, (
        f"理论值 {expected:,} ≠ 实际值 {actual:,}（差 {actual - expected:+,}）\n"
        f"    提示：检查每一项的系数 —— 特别是 FFN 两层和 LayerNorm 的偏置"
    )


def test_weight_tying():
    """确认 LM Head 和 Token Embedding 真的共享了同一份权重。"""
    m = _make_model()
    assert m.lm_head.weight is m.token_emb.weight, "权重共享没生效"
    n_unique = len(list(m.parameters()))
    n_state = len(m.state_dict())
    print(f"    （顺带一个坑：parameters() 有 {n_unique} 项，state_dict() 有 {n_state} 项，差 {n_state - n_unique} 项）")
    print("      差值来自两处：① 共享的 LM Head 权重在 state_dict 里出现两次；")
    print("                    ② 因果掩码用 register_buffer 注册，进 state_dict 但不进 parameters")
    print("      → 所以统计参数量只能用 parameters()，用 state_dict 会多算一份 V×C）")


def test_generate():
    m = _make_model()
    m.eval()
    idx = torch.randint(0, CFG.vocab_size, (1, 3))
    with torch.no_grad():
        out = m.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 8), f"生成结果应为 (1, 8)，实际 {tuple(out.shape)}"


TESTS = [
    ("前向 shape 正确",           test_shapes),
    ("多头注意力 shape 不变",      test_attention_shape),
    ("FFN shape 不变",            test_ffn_shape),
    ("Block shape 不变",          test_block_shape),
    ("★ 因果性（前面不受影响）",    test_causality),
    ("★ 参数量与理论值一致",       test_parameter_count),
    ("权重共享生效",               test_weight_tying),
    ("贪心生成能跑",               test_generate),
]


def run_tests():
    print()
    print("=" * 70)
    print("MiniGPT 测试")
    print("=" * 70)
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
            print(f"  [失败]  {name}")
            print(f"          {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [报错]  {name}")
            print(f"          {type(e).__name__}: {e}")
            failed += 1
    print("-" * 70)
    print(f"通过 {passed} / 失败 {failed} / 待填 {todo}   (共 {len(TESTS)} 项)")
    return passed, failed, todo


# ==========================================================================
# 实验：参数都在哪儿？
# ==========================================================================
def experiment_param_breakdown():
    print()
    print("=" * 70)
    print("实验 · 参数都在哪儿？（看清「注意力」和「FFN」谁是大头）")
    print("=" * 70)
    try:
        m = _make_model()
    except NotImplementedError:
        print("  （填完才能跑）")
        return

    total = sum(p.numel() for p in m.parameters())
    groups = {
        "Token Embedding": m.token_emb.weight.numel(),
        "Position Embedding": m.pos_emb.weight.numel(),
        "最后 LayerNorm": sum(p.numel() for p in m.ln_f.parameters()),
        "注意力（全部 Block）": sum(
            p.numel() for b in m.blocks for p in b.attn.parameters()),
        "FFN（全部 Block）": sum(
            p.numel() for b in m.blocks for p in b.ffn.parameters()),
        "LayerNorm（全部 Block）": sum(
            p.numel() for b in m.blocks for p in (list(b.ln1.parameters()) + list(b.ln2.parameters()))),
    }
    print(f"  d_model={CFG.d_model}, n_layer={CFG.n_layer}, n_head={CFG.n_head}, "
          f"vocab={CFG.vocab_size}, ctx={CFG.ctx_len}")
    print()
    print("  ┌──────────────────────────┬────────────┬─────────┐")
    print("  │ 组件                     │ 参数量     │ 占比    │")
    print("  ├──────────────────────────┼────────────┼─────────┤")
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  │ {k:24s} │ {v:10,d} │ {v/total:6.1%} │")
    print("  ├──────────────────────────┼────────────┼─────────┤")
    print(f"  │ {'合计':24s} │ {total:10,d} │ 100.0%  │")
    print("  └──────────────────────────┴────────────┴─────────┘")
    print()
    print("  两个值得记住的结论：")
    print("    · **FFN 是参数大头**（比注意力多一倍左右）—— 'Transformer 全靠注意力' 是错觉")
    print("    · Embedding 的占比随 d_model 变大而下降（它是 V×C，线性增长；")
    print("      而 Block 内部是 C² 量级，平方增长）")


def experiment_scale():
    print()
    print("=" * 70)
    print("实验 · 改配置，参数量怎么变？")
    print("=" * 70)
    rows = []
    for d_model, n_head, n_layer, tag in [
        (64, 4, 2, "默认"),
        (64, 4, 4, "层数 ×2"),
        (128, 4, 2, "d_model ×2"),
        (128, 8, 2, "d_model + 头数都 ×2"),
    ]:
        cfg = GPTConfig(d_model=d_model, n_head=n_head, n_layer=n_layer)
        rows.append((tag, cfg, theoretical_parameter_count(cfg)))
    print("  ┌──────────────┬─────────┬────────┬────────┬──────────────┐")
    print("  │ 配置         │ d_model │ n_head │ n_layer│ 理论参数量   │")
    print("  ├──────────────┼─────────┼────────┼────────┼──────────────┤")
    for tag, cfg, n in rows:
        print(f"  │ {tag:12s} │ {cfg.d_model:7d} │ {cfg.n_head:6d} │ {cfg.n_layer:6d} │ {n:12,d} │")
    print("  └──────────────┴─────────┴────────┴────────┴──────────────┘")
    print()
    print("  观察：d_model 翻倍，参数量翻了大约 **4 倍**（因为 Block 内部是 C²）。")
    print("        而层数翻倍只是翻倍。**这就是为什么「模型变大」主要是加宽而不是加深。**")


def quiz():
    print()
    print("=" * 70)
    print("自测题")
    print("=" * 70)
    for i, q in enumerate([
        "一层 Block 里有哪些子层？顺序是什么？",
        "残差连接解决什么问题？没有它会怎样？",
        "Pre-LN 和 Post-LN 差在哪？GPT-2 用哪个？",
        "为什么多头要注意的是「拆 C 维」而不是「多算几遍」？总参数量变了吗？",
        "整个模型里参数最多的是哪部分？大约占多少？",
        "为什么 LM Head 可以和 Token Embedding 共享权重？省下多少参数？",
        "统计参数量为什么必须用 parameters() 而不是 state_dict()？",
        "训练时为什么必须加因果掩码？如果不加，loss 会怎样（变好还是变坏）？",
    ], 1):
        print(f"  {i}. {q}")
    print()
    print("  第 8 题是个陷阱题：不加掩码 loss 会**变得更低**（因为模型偷看了答案），")
    print("  但生成时会完全崩掉。**loss 更低不代表模型更好** —— 这是评测里的经典陷阱。")


def main():
    passed, failed, todo = run_tests()
    experiment_param_breakdown()
    experiment_scale()
    quiz()
    print()
    print("=" * 70)
    if failed == 0 and todo == 0:
        print("全部通过。下一步：合上这个文件，从空白重写一遍。")
        print("能重写出来，W2 才算真过。")
    else:
        print("还有未通过项。改完再跑一次 —— 报错信息会直接告诉你哪里对不上。")
    print("=" * 70)


if __name__ == "__main__":
    main()
