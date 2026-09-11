# shape_drill.py 参考答案

> 整理于 2026-09-10。
> **所有「等价写法」都经过实测验证**（形状与数值都一致），不是想当然写下来的。

---

## 总览

| 题 | 标准答案 | 你的写法 | 判定 |
|---|---|---|---|
| ex01 | `x.transpose(-2, -1)` | `x.transpose(-1, -2)` | ✅ 等价 |
| ex02 | `x.unsqueeze(0)` | `x.unsqueeze(0)` | ✅ |
| ex03 | `x.unsqueeze(-1)` | `x.unsqueeze(-1)` | ✅ |
| ex04 | `Q @ K.transpose(-2, -1)` | `Q @ (K.transpose(-1, -2))` | ✅ 等价 |
| ex05 | `x.sum(dim=-1, keepdim=True)` | `x.sum(dim=-1, keepdim=True)` | ✅ |
| ex06 | `x.max(dim=-1, keepdim=True).values` | 同左 | ✅ |
| ex07 | `a / b` | `b / a` | ❌ **顺序反了** |
| ex08 | `scores.masked_fill(~mask, float("-inf"))` | 同左 | ✅ |
| ex09 | `x.permute(0, 2, 1, 3)` | 同左 | ✅ |
| ex10 | `x.reshape(2, 12)` | 同左 | ✅ |

**9 / 10 正确，1 处语义错误（ex07）。**

---

## 等价写法速查表

| 想做的事 | 主写法 | 等价写法 |
|---|---|---|
| 交换最后两维 | `x.transpose(-2, -1)` | `x.transpose(1, 2)`、`x.permute(0, 2, 1)`、`x.mT` |
| 最前面插维度 | `x.unsqueeze(0)` | `x[None, :]`、`x.view(1, -1)` |
| 最后面插维度 | `x.unsqueeze(-1)` | `x[:, None]`、`x.view(-1, 1)` |
| 批量矩阵乘 | `Q @ K.transpose(-2, -1)` | `torch.matmul(...)`、`Q @ K.mT` |
| 沿维归约保留维度 | `x.sum(dim=-1, keepdim=True)` | `x.sum(-1, keepdim=True)` |
| 取最值保留维度 | `x.max(dim=-1, keepdim=True).values` | `torch.amax(x, dim=-1, keepdim=True)` |
| 掩码填值 | `s.masked_fill(~m, float("-inf"))` | `torch.where(m, s, float("-inf"))` |
| 交换相邻两维（任意位置） | `x.permute(0, 2, 1, 3)` | `x.transpose(1, 2)` |
| 转置后改形状 | `x.reshape(2, 12)` | `x.contiguous().view(2, 12)` |

> 等价写法不是"次优解"。PyTorch 的 API 有多条路径能到同一结果，
> 认识它们，读别人的代码时才不吃力。

---

## 三个值得记住的细节

### 1. `transpose` 和 `permute` 怎么选

- 只交换**两个**维度 → 用 `transpose`，更短
- 需要**重排多个**维度 → 用 `permute`

**ex09 就是个例子**：`x.permute(0, 2, 1, 3)` 和 `x.transpose(1, 2)` 效果完全一样，但后者更短。能用 transpose 就别用 permute。

### 2. `max(dim=...)` 返回的是 namedtuple，不是张量

```python
x.max(dim=-1)                         # 返回 (values, indices)，直接拿来用会出错
x.max(dim=-1).values                  # 只取值
x.max(dim=-1, keepdim=True).values    # 取值 + 保留维度
torch.amax(x, dim=-1, keepdim=True)   # 一步到位，且不可能忘记取 .values
```

**`torch.amax` 是更安全的选择**——它只返回值，杜绝了"忘取 `.values`"这类错误。你这次用 `.max(...).values` 也对，但记住还有 amax 这个选项。

### 3. `-2, -1` 比具体数字更稳

你写的是 `transpose(-1, -2)`，我写的是 `transpose(-2, -1)`——**在这个例子里完全等价**（都是交换最后两维）。

推荐用负数索引，因为：**如果张量以后多了一层 batch 维，写死的 `transpose(1, 2)` 就错了，而 `transpose(-2, -1)` 永远正确。** 代码复用到不同维度的张量上时，这一点很关键。

---

## ex07 为什么是这次练习里最重要的一题

`a / b` 和 `b / a` 的**形状完全一样**（广播让两者都变成 `(2, 3, 5)`），只有**数值**不同。

所以只检查形状的断言是**假绿**——它通过了，但答案是错的。补上数值检查（`y * b == a`）之后，错误立刻暴露。

这个教训直接对应你后面要做的每一件事：

| 场景 | 只查 A 会漏掉什么 |
|---|---|
| 只查 shape | 所有数值错误（比如顺序写反、公式抄错） |
| 只查 loss 是否下降 | 模型可能学到了错误的规律 |
| 只查平均值 | 长尾上的严重 failure |
| 只查单次结果 | 随机性导致的偶然 |

**面试里问"你怎么确认这个改动真的有效"，考的就是这个。** 能把"我的断言覆盖了什么、漏掉了什么"讲清楚的人，比能写出漂亮代码的人少得多——而后者是招聘方更想要的。

这正是你的统计背景该发挥作用的地方：**假设检验的核心不是算 p 值，而是先想清楚"什么情况下我会判断错"。**

---
*与 `shape_drill.py`、`attention.py`、`ATTENTION_CONCEPTS.md` 配套*
