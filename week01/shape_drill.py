"""
W1 加练 · 形状推演（L2 专项）

这个文件不教你新的数学，只练一件事：
    看到一个"形状变化"，条件反射地知道该用哪个操作。

用法：
    conda activate llm
    cd E:\\llm-lab\\week01
    python shape_drill.py

================================================================
怎么填空？（重要，先读这段）
================================================================

每道题里都有一行长这样的代码：

    y = _todo("ex01")

**这一行是占位符，不是你要保留的东西。**
你要做的是：把它整行删掉，替换成你自己的代码。

举例 —— ex01 要求把 (2, 3, 4) 转置最后两维，那么：

    原来：  y = _todo("ex01")
    改成：  y = x.transpose(-2, -1)

就完事了。然后运行 `python shape_drill.py`，看到 [通过] 就是对了。

在 VS Code 里按 Ctrl+F 搜索 **填空**，可以依次跳到全部 10 处。
每处上方都有一行注释标明序号，比如 `填空 3/10`。

================================================================
形状变化 ↔ 操作 对照表（这个练习的全部内容）
================================================================

  想做的事                          用什么
  ------------------------------------------------------------------
  交换最后两维                       .transpose(-2, -1)
  按任意顺序重排维度                 .permute(...)
  改变形状（元素总数不变）           .reshape(...)
  在最前面插一个长度 1 的维度        .unsqueeze(0)
  在最后面插一个长度 1 的维度        .unsqueeze(-1)
  去掉长度为 1 的维度                .squeeze(...)
  两个张量做批量矩阵乘               @   （只作用在最后两维）
  沿某一维归约但保留维度             .sum(dim=..., keepdim=True)
  把某些位置换成别的值               .masked_fill(条件, 值)
  ------------------------------------------------------------------

核心规律只有一条：
    **矩阵乘（@）永远只看最后两维，前面的维度原样保留，
      且必须相同或可广播。**

把这条记住，注意力那四个 TODO 就是照着形状表翻译，不需要"猜"。
================================================================
"""

from __future__ import annotations

import math

import torch


def _todo(tag: str):
    """占位符。把它换成你的代码，这一行就不会再被触发了。"""
    raise NotImplementedError(f"{tag} 还没实现")


# ==========================================================================
# 1–3 · 改变形状
# ==========================================================================

def ex01_transpose_last_two():
    """(2, 3, 4) → (2, 4, 3)：交换最后两维。

    这一步对应注意力里的 Kᵀ。
    """
    x = torch.arange(24.0).reshape(2, 3, 4)

    # 填空 1/10 —— 把下一行替换成你的代码
    y = x.transpose(-1,-2)

    assert y.shape == (2, 4, 3), f"期望 (2, 4, 3)，实际 {tuple(y.shape)}"
    # 额外确认语义：转置要真的交换元素，不是只改形状
    assert torch.equal(y[0, 0], x[0, :, 0]), "只改了形状没交换数据"


def ex02_insert_dim():
    """(5,) → (1, 5)：在最前面插入一个长度 1 的维度。

    用途：广播。比如把 bias 加到 batch 上时经常需要。
    """
    x = torch.arange(5.0)

    # 填空 2/10 —— 把下一行替换成你的代码
    y = x.unsqueeze(0)

    assert y.shape == (1, 5), f"期望 (1, 5)，实际 {tuple(y.shape)}"


def ex03_insert_dim_last():
    """(5,) → (5, 1)：在最后面插入一个长度 1 的维度。

    用途：这个就是 softmax 数值稳定版里 keepdim=True 想达到的效果。
    """
    x = torch.arange(5.0)

    # 填空 3/10 —— 把下一行替换成你的代码
    y = x.unsqueeze(-1)

    assert y.shape == (5, 1), f"期望 (5, 1)，实际 {tuple(y.shape)}"


# ==========================================================================
# 4–7 · 批量矩阵乘与广播（注意力里最容易出错的部分）
# ==========================================================================

def ex04_batch_matmul():
    """Q (2, 3, 4) 与 K (2, 5, 4) 做批量矩阵乘，得到 (2, 3, 5)。

    这就是注意力里的"打分"这一步。
    提示：K 的最后两维是 (5, 4)，要变成 (4, 5) 才能和 Q 的 (3, 4) 相乘。
    """
    torch.manual_seed(0)
    Q = torch.randn(2, 3, 4)
    K = torch.randn(2, 5, 4)

    # 填空 4/10 —— 把下一行替换成你的代码
    scores = Q @ (K.transpose(-1,-2))

    assert scores.shape == (2, 3, 5), f"期望 (2, 3, 5)，实际 {tuple(scores.shape)}"


def ex05_keepdim_sum():
    """(2, 3, 5) 沿最后一维求和，但要保留维度 → (2, 3, 1)。

    为什么必须保留？因为下一步要拿它去做除法。
    (2, 3, 5) / (2, 3, 1) 可以广播 → (2, 3, 5)
    (2, 3, 5) / (2, 3)    会报错，右对齐后是 (5,) vs (2, 3)——对不上
    """
    x = torch.randn(2, 3, 5)

    # 填空 5/10 —— 把下一行替换成你的代码
    y = x.sum(dim = -1,keepdim = True)

    assert y.shape == (2, 3, 1), f"期望 (2, 3, 1)，实际 {tuple(y.shape)}"


def ex06_max_keepdim():
    """(2, 3, 5) 沿最后一维取最大值，保留维度 → (2, 3, 1)。

    注意：torch.max(dim=...) 返回的是个 namedtuple（values 和 indices），
    不是张量。必须取 .values，否则后面的形状对不上。
    这个坑第一次踩几乎没人躲得过。
    """
    x = torch.randn(2, 3, 5)

    # 填空 6/10 —— 把下一行替换成你的代码
    y = x.max(dim = -1,keepdim = True).values

    assert isinstance(y, torch.Tensor), f"返回的是 {type(y).__name__}，不是张量——忘了取 .values 吧？"
    assert y.shape == (2, 3, 1), f"期望 (2, 3, 1)，实际 {tuple(y.shape)}"


def ex07_broadcast_divide():
    """用 a 除以 b：a 的形状是 (2, 3, 5)，b 的形状是 (2, 3, 1)，结果应该是 (2, 3, 5)。

    这一步就是 softmax 的核心：e / e.sum(keepdim=True)

    ⚠️ 注意顺序：**被除数在前，除数在后**（大的 ÷ 小的）。
    反过来写形状也对（广播让两者一致），但数值完全错误。
    """
    a = torch.randn(2, 3, 5)
    b = torch.randn(2, 3, 1).abs() + 0.5

    # 填空 7/10 —— 把下一行替换成你的代码
    y = a / b

    assert y.shape == (2, 3, 5), f"期望 (2, 3, 5)，实际 {tuple(y.shape)}"
    # 数值校验：y 应该满足 y * b == a（等价于 y == a / b）
    # 上面只查形状是不够的 —— 顺序写反时形状一样，必须再查数值
    assert torch.allclose(y * b, a, atol=1e-6), \
        "形状对了但数值不对 —— 除数和被除数的顺序是不是反了？"


def ex08_mask_broadcast():
    """把 mask 为 False 的位置填成 -inf。

    scores 是 (2, 3, 5)，mask 是 (3, 5) —— 形状不一样，能不能直接运算？
    **能。** 这就是广播：右对齐后 (3, 5) 对 (2, 3, 5)，前面缺的那一维自动当成 1。
    所以一份 (L, S) 的掩码可以复用到任意 batch 上，这也是因果掩码只造一次就够的原因。

    提示：masked_fill 的第一个参数是"要填的条件"，我们想填的是 mask 为 False 的位置，
          所以要用 ~mask 取反。
    """
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 5)
    mask = torch.tensor([
        [True,  True,  False, False, False],
        [True,  True,  True,  False, False],
        [True,  True,  True,  True,  False],
    ])

    # 填空 8/10 —— 把下一行替换成你的代码
    y = scores.masked_fill(~mask,float("-inf"))

    assert y.shape == (2, 3, 5), f"期望 (2, 3, 5)，实际 {tuple(y.shape)}"
    assert torch.isinf(y[..., ~mask]).all(), "mask 为 False 的位置应该被填成 -inf"
    assert not torch.isinf(y[..., mask]).any(), "mask 为 True 的位置不该被动"


# ==========================================================================
# 9–10 · reshape 与 permute（W2 多头注意力会用到）
# ==========================================================================

def ex09_permute_heads():
    """(2, 4, 3, 8) → (2, 3, 4, 8)：把第 1 维和第 2 维交换。

    这是 W2 多头注意力里的关键一步：
        (batch, 序列长度, 头数, 每头维度) → (batch, 头数, 序列长度, 每头维度)
    因为注意力要按"头"分组算，头数必须和 batch 挨着。

    注意：这里交换的不是最后两维，所以 transpose(-2, -1) 不够用，
          需要用 permute 指定所有维度的新顺序。
    """
    x = torch.arange(2 * 4 * 3 * 8, dtype=torch.float32).reshape(2, 4, 3, 8)

    # 填空 9/10 —— 把下一行替换成你的代码
    y = x.permute(0,2,1,3)

    assert y.shape == (2, 3, 4, 8), f"期望 (2, 3, 4, 8)，实际 {tuple(y.shape)}"


def ex10_reshape_after_transpose():
    """把 (2, 3, 4).transpose(-2, -1) 变成 (2, 12)。

    ⚠️ 这里有个坑：转置之后张量的内存不连续，
       用 .view() 会直接报 RuntimeError。
       要用 .reshape()（它会在需要时自动复制一份连续内存）。

    这是新手最常撞的墙之一：报错信息是
        RuntimeError: view size is not compatible with input tensor's size and stride
    看到这个错，第一反应就该是"我是不是在 transpose 之后用了 view"。
    """
    x = torch.arange(24.0).reshape(2, 3, 4).transpose(-2, -1)

    # 填空 10/10 —— 把下一行替换成你的代码
    y = x.reshape(2,12)

    assert y.shape == (2, 12), f"期望 (2, 12)，实际 {tuple(y.shape)}"


# ==========================================================================
# 测试脚手架（以下不需要改动）
# ==========================================================================

EXERCISES = [
    ("ex01 交换最后两维", ex01_transpose_last_two),
    ("ex02 最前面插维度", ex02_insert_dim),
    ("ex03 最后面插维度", ex03_insert_dim_last),
    ("ex04 批量矩阵乘 + 转置", ex04_batch_matmul),
    ("ex05 归约并保留维度", ex05_keepdim_sum),
    ("ex06 取最值并保留维度", ex06_max_keepdim),
    ("ex07 广播相除", ex07_broadcast_divide),
    ("ex08 掩码广播 + masked_fill", ex08_mask_broadcast),
    ("ex09 permute 交换中间维度", ex09_permute_heads),
    ("ex10 转置后 reshape", ex10_reshape_after_transpose),
]

TOTAL = len(EXERCISES)


def main():
    print()
    print("=" * 62)
    print(f"形状推演练习　（共 {TOTAL} 处填空，Ctrl+F 搜索「填空」可逐个跳转）")
    print("=" * 62)

    passed = failed = todo = 0
    for name, fn in EXERCISES:
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

    print("-" * 62)
    print(f"通过 {passed} / 失败 {failed} / 待填 {todo}   (共 {TOTAL} 项)")
    print()

    if failed == 0 and todo == 0:
        print("全部通过。现在回到 attention.py，你应该能一口气填完了。")
    elif todo > 0 and failed == 0:
        print(f"还有 {todo} 处没填。报 [待填] 是正常的，说明占位符还在。")
    else:
        print("有报错的项。读报错信息本身就是在学习 —— 它通常已经指出哪里不对了。")
    print("=" * 62)


if __name__ == "__main__":
    main()
