"""
W2 热身 · PyTorch 张量 API 词汇练习（20 题，约 40 分钟）
========================================================================

为什么做这个：
    你在 LeetCode 那套（list / dict / set）已经补完了。
    但 PyTorch 是**另一套词汇表** —— 你现在卡在位置编码，不是不懂原理，
    是不知道 `torch.arange` / `torch.zeros` / 切片赋值这些名字。

    **好消息：和上一套一样，它是有限集。** 20 道题覆盖 W2 手写 GPT 需要的一切。

========================================================================
不会写的时候的标准动作（比这 20 道题更值钱）
========================================================================

遇到"完全不会写"，先别下这个结论。按顺序做这 5 步：

  ① 先分清是哪种"不会"：
       · 不知道**要做什么**             → 概念问题 → 回去看文档/论文，别硬写
       · 知道要做什么、不知道**写什么**   → 词汇问题 → 继续往下走
       · 知道怎么写但**怕写错**          → 直接写，跑一下就知道

  ② 把要做的事写成**中文伪代码**，一行一步。写不出来就是①里的概念问题。

  ③ 对每一行问：这是哪一类操作？
       · **造**张量   → torch.zeros / ones / arange / randn / full
       · **变**形状   → view / reshape / unsqueeze / transpose / expand
       · **算**数值   → + - * / 、torch.exp/log/sin/cos/sqrt、@、.sum/.mean/.max
       · **取**元素   → 切片、切片赋值、masked_fill
     定位到类别，范围就从"几千个 API"缩到"不到十个"。

  ④ 去**查**，不要凭记忆硬想：
       · VS Code 里打出 `x.` 或 `torch.`，按 **Ctrl+Space** 看补全列表
         （你的自动补全被关了，但手动唤出留着 —— 就是为这个场景留的）
       · 或翻本文件末尾的速查表

  ⑤ **卡 20 分钟以上**才问人。问的时候带上：伪代码 + 写到哪一步 + 完整报错。

========================================================================
怎么用：
    每道题都写明了**确切的输入**和**要返回什么**，所以结果唯一、能自动判题。
    把 `_todo("exNN")` 那一行整行替换成你的代码，跑 `python tensor_api_drill.py`。
    卡 3 分钟以上就翻文件末尾的速查表 —— 这不是算法题，认识它才是目的。
========================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _todo(where: str):
    raise NotImplementedError(f"{where} 还没填")


# ==========================================================================
# Part 1 · 造张量
# ==========================================================================
def ex01_zeros():
    """返回一个 2 行 3 列的全 0 张量（float32）。"""
    # 填空 1/20
    return torch.zeros(2,3)


def ex02_arange():
    """返回 tensor([0., 1., 2., 3., 4.]) —— 注意必须是**浮点**。

    提示：torch.arange(5) 默认是 int64；要 float 得显式加 dtype
        ⚠️ int 张量和 float 张量混着算会报 dtype 不匹配
    """
    # 填空 2/20
    return torch.arange(5,dtype=torch.float32)


def ex03_randn_seeded():
    """返回一个 (2, 3) 的标准正态随机张量，**同一个种子必须得到同一个结果**。

    要求：函数内部先用 torch.manual_seed(42) 再采样。
        （测试会连调两次，比较是否完全相同 —— 做实验时这一步不能省）
    """
    # 填空 3/20
    return (torch.manual_seed(42),torch.randn(2,3))[1]


def ex04_from_list():
    """把 Python 列表 [[1, 2], [3, 4]] 变成 float32 张量返回。

    提示：torch.tensor(..., dtype=torch.float32)
        torch.tensor 是「拷贝一份」；torch.as_tensor 是「共享内存，不复制」
    """
    # 填空 4/20
    return torch.tensor([[1,2],[3,4]],dtype=torch.float32)


# ==========================================================================
# Part 2 · 变形状
# ==========================================================================
def ex05_view_col():
    """对 `torch.arange(5, dtype=torch.float32)` 操作，返回形状 (5, 1)。

    提示：x.view(-1, 1) —— -1 表示「这一维你替我算」
        也可以 x.unsqueeze(-1)，两种都行
    """
    # 填空 5/20
    x = torch.arange(5,dtype=torch.float32)
    return x.view(-1,1)


def ex06_unsqueeze():
    """对 `torch.zeros(3, 4)` 操作，返回形状 (1, 3, 4)。

    提示：unsqueeze(0) 在最前面插一维；unsqueeze(-1) 在最后面插
    """
    # 填空 6/20
    x = torch.zeros(3,4)
    return x.unsqueeze(0)


def ex07_transpose():
    """对 `torch.zeros(2, 3, 4)` 操作，返回形状 (2, 4, 3)。

    提示：transpose(-1, -2)
        ⚠️ 别写成 transpose(-1.-2) —— 那在 Python 里是个浮点数，会报 "must be int"
    """
    # 填空 7/20
    x = torch.zeros(2,3,4)
    return x.transpose(-1,-2)


def ex08_max_keepdim():
    """对 `torch.zeros(2, 3, 4)` 沿最后一维取最大值，返回形状 (2, 3, 1)。

    提示：x.max(dim=-1, keepdim=True)
        ⚠️ 返回的是 namedtuple，要取 `.values` 才是张量
    """
    # 填空 8/20
    x = torch.zeros(2,3,4)
    return x.max(dim=-1,keepdim=True).values


# ==========================================================================
# Part 3 · 算数值
# ==========================================================================
def ex09_exp_then_log():
    """对 `torch.tensor([[1., 2.], [3., 4.]])` 先 exp 再 log，返回结果。

    期望拿回原张量（这就是 softmax 里 exp 和 log 的关系）。
    提示：torch.exp(x) / torch.log(x)，逐元素运算，形状不变
    """
    # 填空 9/20
    x = torch.tensor([[1.,2.],[3.,4.]])
    return torch.log(torch.exp(x))


def ex10_sin_tensor():
    """对 `torch.zeros(2, 2)` 逐元素取 sin，返回结果。

    ★ 关键区别：`math.sin` 只能算**单个 Python 数**；
      `torch.sin` 才能算**整个张量**（逐元素广播）。
      位置编码里的 sin 必须用 torch.sin。
    """
    # 填空 10/20
    x = torch.zeros(2,2)
    return torch.sin(x)


def ex11_sum_keepdim():
    """对 `torch.zeros(2, 3)` 沿第 0 维求和并保留维度，返回形状 (1, 3)。

    提示：x.sum(dim=0, keepdim=True)
        不写 keepdim 会得到形状 (3,) —— 这就是"归约后形状丢了"的经典问题
    """
    # 填空 11/20
    x = torch.zeros(2,3)
    return x.sum(dim=0,keepdim=True)


def ex12_matmul():
    """`torch.zeros(2, 3)` 与 `torch.zeros(3, 4)` 做矩阵乘，返回形状 (2, 4)。

    提示：直接用 `@` 运算符，或 torch.matmul(a, b)
        矩阵乘**只作用在最后两维**，前面的维度按广播规则对齐
    """
    # 填空 12/20
    x = torch.zeros(2,3)
    y = torch.zeros(3,4)
    return x@y


def ex13_broadcast_mul():
    """`torch.zeros(2, 3, 1)` 与 `torch.zeros(4)` 逐元素相乘，返回形状 (2, 3, 4)。

    这就是广播：把 (4,) 看成 (1, 1, 4)，再和 (2, 3, 1) 逐维对齐 —— 不匹配的维长 1 时自动展开。
    位置编码那行 `pos * omega`（(L,1) × (d/2,)）走的是同一套规则。
    """
    # 填空 13/20
    x = torch.zeros(2,3,1)
    y = torch.zeros(4)
    return x*y


# ==========================================================================
# Part 4 · 取元素 / 切片赋值（位置编码要用）
# ==========================================================================
def ex14_slice_read():
    """对 `torch.zeros(4, 6)` 取**所有行、偶数列**，返回形状 (4, 3)。

    提示：x[:, 0::2] —— 语法是 起:止:步长，省略「止」就是到末尾
    """
    # 填空 14/20
    x = torch.zeros(4,6)
    return x[:,0::2]


def ex15_slice_write():
    """造一个 (4, 4) 全 0 张量，把偶数列（0, 2）设成 1，奇数列（1, 3）设成 2，返回它。

    期望结果：
        [[1, 2, 1, 2],
         [1, 2, 1, 2],
         [1, 2, 1, 2],
         [1, 2, 1, 2]]

    这是**切片赋值**：`pe[:, 0::2] = 1.0`
    位置编码就是这么填进去的 —— 偶数维放 sin，奇数维放 cos。
    ⚠️ 右边可以是标量或能广播的张量，但**不能改变形状**
    """
    # 填空 15/20
    x = torch.zeros(4,4)
    x[:,0::2]=1
    x[:,1::2]=2
    return x


def ex16_masked_fill():
    """把 `torch.tensor([[1., -2.], [-3., 4.]])` 里的负数换成 0，返回结果。

    期望 [[1, 0], [0, 4]]。
    提示：x.masked_fill(x < 0, 0.0)
        你 W1 写的因果掩码用的是同一个方法：`scores.masked_fill(~mask, -inf)`
    """
    # 填空 16/20
    x = torch.tensor([[1.0,-2.],[-3.,4.]])
    return x.masked_fill(x<0,0.0)


# ==========================================================================
# Part 5 · nn 基础（W2 手写 GPT 必需）
# ==========================================================================
def ex17_linear_params():
    """返回 `nn.Linear(8, 16)` 的参数量（一个整数）。

    提示：Linear 计算 y = x @ W.T + b，W 形状 (out, in) = (16, 8)
        参数量 = 16*8（权重）+ 16（偏置）= 144
        用 sum(p.numel() for p in layer.parameters())
    """
    # 填空 17/20

    return sum(p.numel() for p in nn.Linear(8,16).parameters())


def ex18_embedding_shape():
    """用 `nn.Embedding(100, 16)`，输入 `torch.zeros(2, 5, dtype=torch.long)`。

    返回一个二元组：(参数量, 输出形状元组)
    提示：Embedding 就是一张 (num_embeddings, embedding_dim) 的查找表
        参数量 = 100 * 16
        输出形状 = 输入形状 + (embedding_dim,)
        ⚠️ Embedding 的输入必须是整数类型（long），传 float 会报错
    """
    # 填空 18/20
    emb = nn.Embedding(100, 16)
    out = emb(torch.zeros(2, 5, dtype=torch.long))
    return sum(p.numel() for p in emb.parameters()), tuple(out.shape)


def ex19_min_module():
    """补完一个最小的 nn.Module：把输入张量的每个元素乘 2。

    输入 torch.tensor([1.0, 2.0])，期望返回 tensor([2.0, 4.0])。

    三个必须写的部分：
      ① __init__ 里先调 super().__init__()   ← 不写参数注册不上
      ② 定义层（这题不需要层）
      ③ def forward(self, x): 返回结果
    PyTorch 里用 `m(x)` 调用，它会自动转去调 `m.forward(x)`
    —— 所以**永远不要直接写 m.forward(x)**。

    写法：
        class Doubler(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return ...
        return Doubler()(torch.tensor([1.0, 2.0]))
    """
    # 填空 19/20 —— 自己写一个 Doubler 类并调用它
    return _todo("ex19")


def ex20_count_params():
    """统计 `nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))` 的参数量。

    提示：sum(p.numel() for p in model.parameters())
    先自己手算一遍对照：(4*8 + 8) + (8*2 + 2) = 40 + 18 = 58
    **能手算出来，才说明你理解参数量是怎么来的** —— 这是 W2 的验收标准之一。
    """
    # 填空 20/20
    return _todo("ex20")


# ==========================================================================
# 测试（不需要改动）
# ==========================================================================
def _expected():
    t = torch
    return [
        ("ex01 torch.zeros",            ex01_zeros,          t.zeros(2, 3)),
        ("ex02 arange 浮点",            ex02_arange,         t.arange(5, dtype=t.float32)),
        ("ex03 可复现的随机张量",        ex03_randn_seeded,   "deterministic"),
        ("ex04 从列表造张量",            ex04_from_list,      t.tensor([[1, 2], [3, 4]], dtype=t.float32)),
        ("ex05 view(-1,1)",             ex05_view_col,       t.arange(5, dtype=t.float32).view(-1, 1)),
        ("ex06 unsqueeze(0)",           ex06_unsqueeze,      t.zeros(1, 3, 4)),
        ("ex07 transpose(-1,-2)",       ex07_transpose,      t.zeros(2, 4, 3)),
        ("ex08 max(keepdim=True)",      ex08_max_keepdim,    t.zeros(2, 3, 1)),
        ("ex09 exp 再 log 还原",        ex09_exp_then_log,   t.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ("ex10 torch.sin 张量",         ex10_sin_tensor,     t.zeros(2, 2)),
        ("ex11 sum(dim=0,keepdim)",     ex11_sum_keepdim,    t.zeros(1, 3)),
        ("ex12 矩阵乘 @",               ex12_matmul,         t.zeros(2, 4)),
        ("ex13 广播相乘",               ex13_broadcast_mul,  t.zeros(2, 3, 4)),
        ("ex14 切片读偶数列",           ex14_slice_read,     t.zeros(4, 3)),
        ("ex15 切片赋值",               ex15_slice_write,
            t.tensor([[1.0, 2.0, 1.0, 2.0]] * 4)),
        ("ex16 masked_fill",            ex16_masked_fill,    t.tensor([[1.0, 0.0], [0.0, 4.0]])),
        ("ex17 Linear 参数量",          ex17_linear_params,  144),
        ("ex18 Embedding 参数量+形状",   ex18_embedding_shape, (1600, (2, 5, 16))),
        ("ex19 最小 nn.Module",         ex19_min_module,     t.tensor([2.0, 4.0])),
        ("ex20 两层 Sequential 参数量",  ex20_count_params,   58),
    ]


def _check(exp, got):
    if exp == "deterministic":
        return True     # ex03 单独处理
    if isinstance(exp, torch.Tensor):
        return (isinstance(got, torch.Tensor)
                and tuple(got.shape) == tuple(exp.shape)
                and torch.allclose(got.float(), exp.float(), atol=1e-6))
    if isinstance(exp, tuple) and exp and isinstance(exp[0], int):
        return isinstance(got, tuple) and len(got) == 2 \
               and got[0] == exp[0] and tuple(got[1]) == tuple(exp[1])
    return got == exp


def run_tests():
    print()
    print("=" * 70)
    print("PyTorch 张量 API 词汇练习")
    print("=" * 70)
    passed = failed = todo = 0
    for name, fn, exp in _expected():
        try:
            got = fn()
            if exp == "deterministic":
                ok = (got == fn()).all().item() if isinstance(got, torch.Tensor) else False
                if not ok:
                    print(f"  [失败]  {name}")
                    print("          连调两次结果不同 —— 忘了 torch.manual_seed 吧？")
                    failed += 1
                    continue
            elif not _check(exp, got):
                print(f"  [失败]  {name}")
                print(f"          期望 {exp!r}")
                print(f"          实际 {got!r}")
                failed += 1
                continue
            print(f"  [通过]  {name}")
            passed += 1
        except NotImplementedError:
            print(f"  [待填]  {name}")
            todo += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [报错]  {name}")
            print(f"          {type(e).__name__}: {e}")
            failed += 1
    print("-" * 70)
    print(f"通过 {passed} / 失败 {failed} / 待填 {todo}   (共 {len(_expected())} 项)")
    return passed, failed, todo


# ==========================================================================
# 速查表（卡住就翻这里）
# ==========================================================================
def cheatsheet():
    print()
    print("=" * 70)
    print("速查表 · 四类操作各自的 API 家族")
    print("=" * 70)
    groups = [
        ("① 造张量", [
            "torch.zeros(2, 3)              全 0，默认 float32",
            "torch.ones(2, 3)               全 1",
            "torch.arange(5)                0..4，默认 int64 → 要 float 加 dtype",
            "torch.randn(2, 3)              标准正态（先用 manual_seed 固定）",
            "torch.full((2, 3), 7.0)        用某个值填满",
            "torch.tensor([[1, 2]])         从 Python 列表造（拷贝一份）",
            "torch.eye(3)                   单位矩阵",
        ]),
        ("② 变形状", [
            "x.view(-1, 1) / x.reshape(-1, 1)   -1 = 这维你替我算",
            "x.unsqueeze(0) / x.unsqueeze(-1)   前面 / 后面插一维",
            "x.squeeze()                    去掉所有长度为 1 的维",
            "x.transpose(-1, -2)            交换最后两维（负数索引更安全）",
            "x.permute(0, 2, 1, 3)          任意重排维度",
            "x.expand(2, 3, 4)              广播扩展（不真正复制内存）",
        ]),
        ("③ 算数值", [
            "x + y, x * y                   逐元素（形状要能广播）",
            "torch.exp/log/sqrt/sin/cos(x)  逐元素数学函数",
            "math.sin(0.5)                  只能算单个 Python 数，不能算张量",
            "x.sum(dim=, keepdim=)          归约；keepdim=True 保留被归约的维",
            "x.mean(dim=) / x.max(dim=)",
            "x.max(dim=-1).values           ⚠️ 返回 namedtuple，要取 .values",
            "a @ b / torch.matmul(a, b)     矩阵乘（只作用最后两维）",
            "x.item() / x.tolist()          取成 Python 数 / 列表",
        ]),
        ("④ 取元素", [
            "x[:, 0::2]                     切片：所有行、偶数列",
            "x[:, 0::2] = 1.0               切片赋值（位置编码靠这个）",
            "x.masked_fill(mask, -inf)      按布尔掩码填值",
            "x[x < 0] = 0                   布尔索引赋值",
            "x.shape / x.dtype / x.device   常用属性",
        ]),
    ]
    for title, items in groups:
        print(f"\n  {title}")
        for it in items:
            print(f"    {it}")

    print()
    print("=" * 70)
    print("nn 模块的最小知识（W2 必需）")
    print("=" * 70)
    print("""
  class M(nn.Module):
      def __init__(self):
          super().__init__()              # ← 必须，否则参数注册不上
          self.fc = nn.Linear(4, 8)       # ← 层在 __init__ 里定义

      def forward(self, x):
          return self.fc(x)               # ← 计算过程在 forward 里定义

  m = M()
  y = m(x)          # ← 这样调用，PyTorch 自动转去调 forward
  # m.forward(x)    # ← 永远不要这样直接调

  sum(p.numel() for p in m.parameters())   # ← 统计参数量
  m.train() / m.eval()                     # ← 训练 / 推理模式（影响 Dropout）
""")


def main():
    passed, failed, todo = run_tests()
    cheatsheet()
    print()
    print("=" * 70)
    if failed == 0 and todo == 0:
        print("20/20 全绿。这套词汇补完，W2 手写 GPT 不会再因为「不知道写什么 API」卡住。")
    elif failed == 0:
        print(f"还有 {todo} 题待填。卡 3 分钟以上就翻上面的速查表。")
    else:
        print("有报错的项。看报错信息 —— 它通常直接告诉你缺哪个方法名。")
    print("=" * 70)


if __name__ == "__main__":
    main()
