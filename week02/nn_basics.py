"""
nn 模块的最小知识 —— 免填空，直接跑，看现象
========================================================================

运行：
    cd E:\\llm-lab\\week02
    python nn_basics.py

这个文件不用你填任何代码。它就是把你「没见过的东西」一个个跑给你看，
每一节结束都有一行「→ 结论」。

为什么是免填空的：你缺的是「见过」，不是「练过」。先看完现象，再回去填
tensor_api_drill.py 的 ex19 / ex20，那时候你会发现它们是水到渠成的事。

配套阅读：NN_BASICS.md（同一目录）
========================================================================
"""

import torch
import torch.nn as nn


def sep(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ==========================================================================
sep("§0 · 先分清两个层次：你已经会的 vs 你现在要学的")

print("""
  你已经会的那一层（张量运算）：
      x = torch.randn(5, 3)
      y = x @ W.T + b            ← 纯函数式：给输入、拿输出，没有"状态"

  你现在的这一层（nn）：
      layer = nn.Linear(3, 2)    ← 一个**对象**，它自己带着参数 W 和 b
      y = layer(x)

  区别只有一个：**参数存在对象里了。**
      → 好处：不用手动把 W、b 传来传去；优化器能自己找到它们。
      仅此而已。nn 没有新数学，只有"把参数和计算打包"。

  nn.Module 就是 PyTorch 给这个"打包"定的统一协议。
  MultiHeadAttention / FeedForward / Block / MiniGPT —— 全都是这套协议。
""")

print("  → 结论：你不需要学新数学，只需要学 4 条约定（下面 §4–§8）。")


# ==========================================================================
sep("§1 · nn.Linear 到底做了什么")

lin = nn.Linear(3, 2)
print(f"  nn.Linear(3, 2)  —— 输入 3 维，输出 2 维")
print()
print(f"  lin.weight.shape = {tuple(lin.weight.shape)}   ← ★ 注意是 (2, 3)，不是 (3, 2)")
print(f"  lin.bias.shape   = {tuple(lin.bias.shape)}")
print(f"  参数量           = {sum(p.numel() for p in lin.parameters())}  (2*3 + 2)")

x = torch.randn(5, 3)
y = lin(x)
manual = x @ lin.weight.T + lin.bias
diff = (y - manual).abs().max().item()

print()
print(f"  x.shape = {tuple(x.shape)}  →  y.shape = {tuple(y.shape)}")
print(f"  它做的运算：y = x @ W.T + b")
print(f"  我手写 x @ lin.weight.T + lin.bias 与 lin(x) 的最大误差 = {diff:.2e}")

print("""
  → 结论：nn.Linear 就是「一个存着 W 和 b 的矩阵乘」，没有别的东西。

  ★ 那个 (2, 3) 是最常见的困惑点：
      nn.Linear(in_features, out_features) 的参数顺序是 (in, out)，
      但 weight 的形状是 (out_features, in_features) —— **反的**。
      因为 PyTorch 做的是 x @ W.T（用转置），而不是 x @ W。
      你会在 mini_gpt 的参数量公式里用到这一点。
""")


# ==========================================================================
sep("§2 · nn.Embedding 就是一张查找表")

emb = nn.Embedding(10, 4)          # 10 个词，每个词 4 维向量
idx = torch.tensor([3, 7, 3])
out = emb(idx)
via_index = emb.weight[idx]

print(f"  nn.Embedding(10, 4)  —— 词表大小 10，每个 token 用 4 维表示")
print(f"  emb.weight.shape = {tuple(emb.weight.shape)}   ← 参数量 10*4 = {emb.weight.numel()}")
print()
print(f"  输入 idx = {idx.tolist()}   shape {tuple(idx.shape)}")
print(f"  输出 out        shape {tuple(out.shape)}   ← 输入形状 + (4,)")
print(f"  它做的就是查表：emb(idx) 与 emb.weight[idx] 完全相同 = {torch.equal(out, via_index)}")
print(f"  （注意 idx=[3,7,3] 里 3 出现了两次，输出第 0 行和第 2 行也相同 = {torch.equal(out[0], out[2])}）")

# 梯度只回传给被用到的行
emb2 = nn.Embedding(10, 4)
emb2(torch.tensor([3, 7])).sum().backward()
grad_rows = (emb2.weight.grad.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
print()
print(f"  反向传播后，weight.grad 里非零的行 = {grad_rows}")
print(f"  → 只有被查过的第 3、7 行有梯度，其余 8 行梯度是 0（不会更新）")

print("""
  → 结论：Embedding = 查表 = 一个 (词表大小, 维度) 的矩阵按行索引。

  ★ W2 里的位置编码就是这么实现的：`nn.Embedding(ctx_len, d_model)`。
    它和正弦位置编码的区别只是「这张表是自己学的」而不是公式算的。
""")


# ==========================================================================
sep("§3 · 那 nn.Module 到底解决了什么？")

print("""  先看不用 nn.Module 时，你要自己扛多少事：

      W = torch.randn(2, 3, requires_grad=True)
      b = torch.zeros(2, requires_grad=True)
      params = [W, b]              # ← 手动维护参数列表

  层一多，麻烦就来了：
      · 优化器要接收完整参数列表 —— 你得手动收集每一层的 W/b
      · 想搬到 GPU，得手动 .to(device) 每一个 —— 漏一个就报错
      · 想切换训练/推理模式（Dropout、BN 行为不同）—— 得手动递归
      · 想保存/加载模型 —— 得手动组织 state_dict
      · 想在某个模块前后插 hook 做调试 —— 无从下手

  nn.Module 把这些全部统一了：**你只要把参数放进 self 的属性里，
  剩下的（收集、搬运、模式切换、保存、hook）它自动做。**
""")

# 实测：手写的和 nn.Linear 参数量一致
W = torch.randn(2, 3, requires_grad=True)
b = torch.zeros(2, requires_grad=True)
manual_n = W.numel() + b.numel()
print(f"  手写 (W + b) 参数量 = {manual_n}   |   nn.Linear(3, 2) 参数量 = {sum(p.numel() for p in lin.parameters())}")
print("  数字一样 —— nn 没有多做什么，只是**帮你把 W 和 b 注册进了一个统一的地方**。")

print("  → 结论：nn.Module 是「参数注册 + 生命周期管理」的协议，不是新运算。")


# ==========================================================================
sep("§4 · 最小 nn.Module：三个必须写的部分（这就是 ex19）")

print("""      class Doubler(nn.Module):          # ① 继承 nn.Module
          def __init__(self):
              super().__init__()          # ② 第一行必须调它
              # 在这里定义层

          def forward(self, x):           # ③ 计算写在 forward 里
              return x * 2
""")


class Doubler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2


d = Doubler()
inp = torch.tensor([1.0, 2.0])
print(f"  Doubler()(torch.tensor([1.0, 2.0])) = {d(inp).tolist()}")

# 带参数的版本
class ScaleShift(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n))     # nn.Parameter = 会被训练的张量
        self.shift = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return x * self.scale + self.shift


ss = ScaleShift(3)
print(f"\n  ScaleShift(3) 的 parameters() 有 {len(list(ss.parameters()))} 项"
      f"，参数量 {sum(p.numel() for p in ss.parameters())}")
print(f"  输入 [1,2,3] → 输出 {ss(torch.tensor([1.0, 2.0, 3.0])).tolist()}   （scale=1, shift=0 时是恒等）")

print("""
  → 结论：三个部分 = 继承 nn.Module / __init__ 里 super().__init__() / 写 forward。
     层写在 __init__，计算写在 forward。
""")


# ==========================================================================
sep("§5 · 约定二：super().__init__() 不写会怎样？（实测报错）")

print("  试一下省略 super().__init__()：\n")

class Bad1(nn.Module):
    def __init__(self):
        # super().__init__()   ← 故意注释掉
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


try:
    Bad1()
    print("  （没报错？）")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

print("""
  → 结论：报错原文是 "cannot assign module before Module.__init__() call"。
     它的原因：nn.Module 的 __init__ 会**初始化那些用来记录子模块的内部字典**。
     你不调它，那些字典还不存在，所以 self.fc = ... 无处可存，直接抛错。

  ★ 这个报错的特征是「错误发生在赋值那一行，而不是 super() 那一行」，
    所以第一次遇到会觉得莫名其妙的。记住它。
""")


# ==========================================================================
sep("§6 · 约定三：用 m(x)，不要用 m.forward(x)")

log = []

class Watched(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


w = Watched()
w.register_forward_hook(lambda module, args, output: log.append("hook 被触发了") or output)

log.clear()
y1 = w.forward(torch.randn(1, 2))       # 直接调 forward
print(f"  w.forward(x)  调用后，hook 记录 = {log}")

log.clear()
y2 = w(torch.randn(1, 2))               # 正常调用
print(f"  w(x)          调用后，hook 记录 = {log}")

print("""
  → 结论：数值上两者一样，但 `w(x)` 走的是 nn.Module.__call__，
     它会在前后插入 PyTorch 的整套机制（hook、自动混合精度、编译等）。
     直接调 forward 会**跳过**这些。

  ★ 所以规则是：永远写 m(x)。
    顺带一个实际影响：mini_gpt 里做因果性测试时，要靠 forward hook 抓
    中间张量 —— 如果你在代码里写 m.forward(x)，hook 就抓不到。
""")


# ==========================================================================
sep("§7 · 参数是怎么被「自动找到」的？—— 两个真实的坑")

print("""  规则：只有 nn.Parameter（或 nn.Module 子模块）才会被 parameters() 找到。
        普通的 torch.Tensor 属性 **不会被注册**。

  坑 ①  用普通张量当参数：
""")


class Bad2(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.ones(3)          # ← 普通张量，不是 nn.Parameter

    def forward(self, x):
        return x * self.scale


b2 = Bad2()
print(f"      Bad2().parameters() 有 {len(list(b2.parameters()))} 项"
      f"   state_dict 有 {len(b2.state_dict())} 项")
print(f"      → 参数被「看见了」吗：{'是' if len(list(b2.parameters())) else '否'}")
print("""      → 后果：这个 scale 永远不会被优化器更新，保存模型时也会丢掉。
        但它照样参与前向计算 —— 所以**训练能跑、loss 也降**，只是这一层学不动。
        这类 bug 特别隐蔽：不报错，效果差一点，你会以为是超参问题。""")

print("""
  坑 ②  用普通 Python list 装子模块：
""")


class Bad3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 2), nn.Linear(2, 2)]     # ← 普通 list

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Good3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])   # ← ModuleList

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


print(f"      普通 list    : parameters() = {len(list(Bad3().parameters()))} 项，参数量 {sum(p.numel() for p in Bad3().parameters())}")
print(f"      nn.ModuleList: parameters() = {len(list(Good3().parameters()))} 项，参数量 {sum(p.numel() for p in Good3().parameters())}")
print("""      → 前者是 0 项，后者是 4 项。用普通 list，那些层的参数**不会出现在
        parameters() 里**，优化器看不到它们，等于白定义。

  → 结论：可训练的东西必须挂成 nn.Parameter 或 nn.Module 子模块；
     装一列子模块必须用 nn.ModuleList，不能用普通 list。

  ★ mini_gpt 里 `self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])`
    就是为了避开坑 ②。你现在知道那行为什么必须这么写了。
""")


# ==========================================================================
sep("§8 · 参数量怎么数：parameters() vs state_dict()")

seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
n_params = sum(p.numel() for p in seq.parameters())
print(f"  nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,2))")
print(f"      手算：(4*8 + 8) + (8*2 + 2) = 40 + 18 = 58")
print(f"      代码：sum(p.numel() for p in seq.parameters()) = {n_params}   ← 对上了")
print(f"  （nn.ReLU 没有参数，所以不计入）")

# parameters() 与 state_dict() 的差别
class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.head = nn.Linear(4, 10, bias=False)
        self.head.weight = self.emb.weight              # 权重共享
        self.register_buffer("mask", torch.tril(torch.ones(3, 3, dtype=torch.bool)))

    def forward(self, idx):
        return self.head(self.emb(idx))


tt = Tied()
print()
print(f"  另一个例子（权重共享 + register_buffer）：")
print(f"      parameters() 有 {len(list(tt.parameters()))} 项")
print(f"      state_dict() 有 {len(tt.state_dict())} 项")
print(f"      parameters() 参数量 = {sum(p.numel() for p in tt.parameters())}   ← 共享权重只算一次")
print(f"      state_dict 里的键  = {list(tt.state_dict().keys())}")
print("""      → 差值来自两处：
          ① 共享的权重在 state_dict 里出现两次（emb.weight 和 head.weight 是同一块内存）
          ② register_buffer 的 mask 进 state_dict，但**不进 parameters**（它不是可训练参数）

  → 结论：**统计参数量只能用 parameters()**。用 state_dict 会多算一份。
""")

print("  对照 mini_gpt 的验收标准：")
print("      theoretical_parameter_count() == sum(p.numel() for p in model.parameters())")
print("      两个测试都在 mini_gpt.py 的 test_parameter_count 里。")


# ==========================================================================
sep("§9 · 自测题（不用回我，能答上来就过）")

questions = [
    "§1  nn.Linear(3, 2) 的 weight 形状是什么？为什么不是 (3, 2)？",
    "§2  nn.Embedding(10, 4) 收到 torch.tensor([3, 7, 3]) 会输出什么形状？它做了什么运算？",
    "§3  用 nn.Module 相比手写 W/b 列表，省掉了哪四件事？",
    "§4  写一个最小 nn.Module，必须有的三个部分是什么？层定义写在哪儿、计算写在哪儿？",
    "§5  漏写 super().__init__() 时，报错出现在哪一行？为什么？",
    "§6  m(x) 和 m.forward(x) 数值一样，为什么必须用前者？",
    "§7  用 self.scale = torch.ones(3)（普通张量）当参数，会静默地出什么错？",
    "§7  self.layers = [nn.Linear(2,2), nn.Linear(2,2)] 和 nn.ModuleList 的区别是什么？",
    "§8  self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]) —— 为什么不能写成普通 list？",
    "§8  nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,2)) 的参数量是多少？怎么算的？",
    "§8  统计参数量为什么用 parameters() 而不是 state_dict()？",
    "★   mini_gpt 的 Block.forward 里要写「残差 + Pre-LN」，用到的 self.ln1 是在哪儿定义的？",
    "★   如果你忘了在 Block.__init__ 里写 self.ln1 = nn.LayerNorm(...)，报错会在哪一步出现？",
]
for i, q in enumerate(questions, 1):
    print(f"  {i:2d}. {q}")

print()
print("=" * 72)
print("""
看完这些，回去填 tensor_api_drill.py 的 ex19 和 ex20 —— 它们现在是同一件事：

    ex19 = §4 的 Doubler（三个部分 + 用 m(x) 调用）
    ex20 = §8 的 sum(p.numel() for p in model.parameters())

下一步就是 mini_gpt.py。那 9 处填空里，没有一处超出这个文件的范围：
    填空 1/9  ← §1（nn.Linear 的参数顺序）
    填空 6/9  ← §1
    填空 9/9  ← §2（Embedding）+ §7（ModuleList）
    其余      ← 你 W1 已经会的东西（张量运算、掩码、形状）
""")
print("=" * 72)
