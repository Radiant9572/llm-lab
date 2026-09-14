# nn 模块的最小知识 —— 手写 GPT 之前必须过的四道关

> **配套可运行演示：`nn_basics.py`**（免填空，直接跑，`python nn_basics.py`）
> 本文是速查与索引，所有结论都在 `nn_basics.py` 里实测过。

---

## 0 · 先说清楚你缺的到底是什么

`tensor_api_drill.py` 你填了 **ex01–ex18 全过（18/20）**，剩两道正好都是 `nn`。

而这两道要的东西，**不是一整套新知识，是 4 条约定**：

| 关 | 约定 | 对应 |
|---|---|---|
| 一 | 继承 `nn.Module`，层写在 `__init__`、计算写在 `forward` | ex19 / mini_gpt 全部 9 处填空 |
| 二 | `__init__` 第一行必须 `super().__init__()` | 同上 |
| 三 | 用 `m(x)`，不写 `m.forward(x)` | 影响调试与 hook |
| 四 | 可训练的东西必须是 `nn.Parameter` / 子模块；装一列子模块用 `nn.ModuleList` | mini_gpt 填空 9/9 |

再加上一个算式：**参数量 = `sum(p.numel() for p in model.parameters())`**（ex20 与 W2 验收标准）。

**四关，两小时以内。** 原文里没有一处需要新数学。

---

## 1 · 两个层次：你已经会的 vs 现在要学的

```python
# 你会的（张量运算）：纯函数式，无状态
y = x @ W.T + b

# 现在要学的（nn）：把参数和计算打包成一个对象
layer = nn.Linear(3, 2)
y = layer(x)
```

**唯一的区别是「参数存在对象里了」。** 好处有三：不用手动把 W/b 传来传去；优化器能自己找到它们；`.to(device)` / `train()` / 保存加载都自动。

`nn.Linear(3, 2)` 和手写 `x @ W.T + b` 的参数量完全一样（都是 8）——**nn 没有引入任何新的数学**。

---

## 2 · `nn.Linear`：参数顺序是反的

```python
lin = nn.Linear(3, 2)
lin.weight.shape   # (2, 3)  ← ★ 不是 (3, 2)
lin.bias.shape     # (2,)
参数量              # 2*3 + 2 = 8
```

| 事实 | 值 |
|---|---|
| 构造参数顺序 | `(in_features, out_features)` = `(3, 2)` |
| `weight` 形状 | `(out_features, in_features)` = `(2, 3)` —— **反的** |
| 因为它算的是 | `y = x @ W.T + b`（**带转置**） |
| 验证 | `x @ lin.weight.T + lin.bias` 与 `lin(x)` 最大误差 **0.00e+00** |

**`bias=False` 时没有 bias**，参数量只剩 `out × in`。mini_gpt 的 `lm_head` 就是这么写的（因为要和 `token_emb` 共享权重）。

---

## 3 · `nn.Embedding`：就是查表

```python
emb = nn.Embedding(10, 4)          # 10 个词 × 4 维
emb.weight.shape                   # (10, 4)，参数量 40

idx = torch.tensor([3, 7, 3])
emb(idx).shape                     # (3, 4)  ← 输入形状 + (4,)
```

| 事实 | 实测 |
|---|---|
| 它做的运算 | 按行索引：`emb(idx)` 与 `emb.weight[idx]` **完全相同** |
| 相同索引 → 相同输出 | `idx=[3,7,3]`，第 0 行与第 2 行相同 = True |
| 梯度 | 反向传播后只有被查过的第 **3、7** 行梯度非零，其余 8 行是 0 |
| 输入类型 | **必须是整数（long）**，传 float 直接报错 |

⚠️ **mini_gpt 的位置编码就是 `nn.Embedding(ctx_len, d_model)`** —— 它和正弦位置编码的区别只是「这张表是学出来的」而不是公式算的。

---

## 4 · `nn.Module` 解决的是「生命周期管理」

不用它，你得自己扛：优化器参数列表、`.to(device)`、`train()/eval()` 递归切换、保存加载、插 hook。

**用了它，你只要把参数放进 `self` 的属性里，剩下全自动。**

---

## 5 · 最小 Module：三个部分

```python
class Doubler(nn.Module):          # ① 继承
    def __init__(self):
        super().__init__()          # ② 第一行，必须
        # 层定义在这里

    def forward(self, x):           # ③ 计算写这里
        return x * 2

Doubler()(torch.tensor([1.0, 2.0]))   # → tensor([2.0, 4.0])
```

**记忆锚点：层在 `__init__`，算在 `forward`。**

### 漏写 `super().__init__()` 会怎样（实测）

```
AttributeError: cannot assign module before Module.__init__() call
```

**报错出现在「赋值那一行」，不是 `super()` 那一行** —— 因为 `nn.Module.__init__` 负责初始化那些记录子模块的内部字典，不调它，`self.fc = ...` 无处可存。

---

## 6 · 用 `m(x)`，不写 `m.forward(x)`

| 调用方式 | 结果相同？ | hook 会触发？ |
|---|---|---|
| `w(x)` | ✅ | **✅** |
| `w.forward(x)` | ✅ | **❌** |

`w(x)` 走的是 `nn.Module.__call__`，它负责插入 hook、混合精度、编译等机制。直接调 `forward` 会**静默跳过**这些。

**规则：永远写 `m(x)`。**
实测影响：mini_gpt 的因果性测试要靠 forward hook 抓中间张量，写成 `m.forward(x)` 就抓不到。

---

## 7 · 两个会「静默出错」的坑

### 坑 ① 用普通张量当参数

```python
self.scale = torch.ones(3)        # ✗ 普通张量
self.scale = nn.Parameter(torch.ones(3))   # ✓
```

| | `parameters()` 项数 | `state_dict` 项数 |
|---|---|---|
| `self.scale = torch.ones(3)` | **0** | **0** |
| `self.scale = nn.Parameter(...)` | 1 | 1 |

**后果极其隐蔽**：这个参数照样参与前向计算，**训练能跑、loss 也降**，但它永远不会被优化器更新，保存模型时也会丢。你会以为是超参问题，查很久。

### 坑 ② 用普通 list 装子模块

```python
self.layers = [nn.Linear(2,2), nn.Linear(2,2)]                 # ✗ parameters() 0 项
self.layers = nn.ModuleList([nn.Linear(2,2), nn.Linear(2,2)])  # ✓ parameters() 4 项
```

**差 0 项 vs 4 项。** 普通 list 里的层，参数不会出现在 `parameters()` 里，优化器看不到 = 白定义。

★ **mini_gpt 里 `self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])` 就是为了避开坑 ②。**

**总结一句话：可训练的东西必须挂成 `nn.Parameter` 或 `nn.Module` 子模块；装一列子模块必须用 `nn.ModuleList`。**

---

## 8 · 统计参数量：只能用 `parameters()`

```python
nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
# 手算：(4*8 + 8) + (8*2 + 2) = 40 + 18 = 58
sum(p.numel() for p in seq.parameters())   # 58 ✓
```

`nn.ReLU` 没有参数，不计入。

### `parameters()` vs `state_dict()`

一个「权重共享 + register_buffer」的例子：

| | 项数 | 说明 |
|---|---|---|
| `parameters()` | **1** | 共享权重只算一次 |
| `state_dict()` | **3** | `['mask', 'emb.weight', 'head.weight']` |

差值来自两处：
1. **共享的权重在 `state_dict` 里出现两次**（`emb.weight` 和 `head.weight` 是同一块内存）
2. **`register_buffer` 的 `mask` 进 `state_dict`，但不进 `parameters()`**（它不是可训练参数）

**结论：统计参数量只能用 `parameters()`。** 用 `state_dict()` 会多算一份，永远对不上。

---

## 9 · 与 `mini_gpt.py` 的对应表

mini_gpt 的 9 处填空，**没有一处超出本文范围**：

| 填空 | 需要什么 | 出自 |
|---|---|---|
| 1/9 定义两个线性层 | `nn.Linear(C, C)`，注意参数顺序 | §2 |
| 6/9 FFN 两层线性层 | `nn.Linear(C, 4C)` / `nn.Linear(4C, C)` | §2 |
| 9/9 词嵌入 + 位置嵌入 | `nn.Embedding` + `nn.ModuleList` | §3 / §7 |
| 2–5、7、8 | 张量运算、掩码、形状、残差 | **W1 你已经会的** |

W2 的验收标准「参数量与理论值对得上」= §8 那一行。

---

## 10 · 自测题

跑完 `nn_basics.py` 后逐条回答（答不上就回看对应小节）：

1. `nn.Linear(3, 2)` 的 `weight` 形状是什么？为什么不是 `(3, 2)`？
2. `nn.Embedding(10, 4)` 收到 `[3, 7, 3]` 输出什么形状？它做了什么运算？
3. 用 `nn.Module` 相比手写 W/b 列表，省掉了哪四件事？
4. 最小 `nn.Module` 必须有的三个部分是什么？层写哪儿、计算写哪儿？
5. 漏写 `super().__init__()` 时，报错出现在哪一行？为什么？
6. `m(x)` 和 `m.forward(x)` 数值一样，为什么必须用前者？
7. `self.scale = torch.ones(3)` 当参数，会静默地出什么错？
8. `self.layers = [nn.Linear(2,2), ...]` 和 `nn.ModuleList` 的区别？
9. 为什么 `self.blocks` 必须用 `nn.ModuleList`？
10. `nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,2))` 参数量多少？怎么算的？
11. 统计参数量为什么用 `parameters()` 而不是 `state_dict()`？

---
*与 `nn_basics.py`、`tensor_api_drill.py`、`mini_gpt.py` 配套*
