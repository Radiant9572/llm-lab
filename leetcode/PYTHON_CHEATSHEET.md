# LeetCode Python 最小词汇表

> 配套练习：`python_drill.py`（22 题）　答案：`python_drill_ANSWERS.md`

---

## 0 · 先说清一件事

**你卡住的不是 Python，是"没背过这套词汇"。**

它是一个**有限集**：下面 14 组工具，覆盖 Hot 100 里 90% 的语法需求。**背熟它，从 D2 到 D21 不会再因为"这行怎么打"停下来。**

更重要的是这个判断：

> **对算法题来说，"会用什么工具"和"想到什么算法"往往是同一件事。**

你在两数之和里写的"可以利用哈希表来找"——那个"洞察"之所以成立，恰恰是因为 `dict` 的查询是 O(1)。**如果你不知道 `dict.get()` 怎么写，这个洞察就永远停在纸上。** 所以补词汇不是"补基础"，是直接补解题能力。

**怎么用这份表**：不要通读，当字典查。做 `python_drill.py` 时卡 3 分钟以上就翻对应的那一节。

---

## 1 · 遍历与下标

| 写法 | 作用 |
|---|---|
| `for i, x in enumerate(xs):` | 同时拿下标和元素 ← **用得最多** |
| `for i, x in enumerate(xs, 1):` | 下标从 1 开始 |
| `for a, b in zip(xs, ys):` | 两个列表配对遍历 |
| `for i in range(n-1, -1, -1):` | 倒序遍历 |
| `for i in range(0, n, 2):` | 步长为 2 |

```python
for i, ch in enumerate(["a", "b"]):
    print(i, ch)          # 0 a / 1 b
```

> 坑：`zip` 返回的是**迭代器**，要 `list(zip(...))` 才能打印或比较。

---

## 2 · 排序（Hot 100 出现率最高的工具之一）

| 写法 | 说明 |
|---|---|
| `sorted(xs)` | **返回新列表**，原列表不变 |
| `xs.sort()` | **原地排序**，返回 `None` |
| `sorted(xs, reverse=True)` | 降序 |
| `sorted(words, key=len)` | 按长度排 |
| `sorted(pairs, key=lambda p: p[1])` | 按元组第二个元素排 |
| `sorted(d.items(), key=lambda kv: kv[1], reverse=True)` | 字典按值排序 |

**三条必须记住的区别**：

```python
a = [3, 1, 2]
b = sorted(a)      # a 还是 [3,1,2]，b 是 [1,2,3]
a.sort()           # a 变成 [1,2,3]，返回值是 None
```

> ⚠️ `x = xs.sort()` 是经典 bug —— `x` 会是 `None`。

> ⚠️ `key=len` 传的是**函数本身**，不是 `len()`。写成 `key=len()` 会报错。

**对应哪天**：D6（三数之和，要先把数组排序）、D14（合并区间，`sorted(intervals, key=lambda x: x[0])`）、D2（异位词规范形式）

---

## 3 · dict —— "用空间换时间"的那个空间

| 写法 | 作用 |
|---|---|
| `d[k] = v` | 赋值 |
| **`d.get(k, default)`** | **取不到时返回默认值，不报错** ← 最高频 |
| `d.setdefault(k, []).append(v)` | 键不存在时先建空列表再追加 |
| `k in d` | 判断键是否存在，**O(1)** |
| `d.pop(k, default)` | 删除并返回 |
| `for k, v in d.items():` | 同时遍历键值 |
| `d.keys()` / `d.values()` | 只有键 / 只有值 |
| `len(d)` | 键的个数 |

**最经典的惯用法（手写词频统计）**：

```python
d = {}
for w in words:
    d[w] = d.get(w, 0) + 1
```

> ⚠️ `d[k]` 在键不存在时直接 `KeyError`。**不确定键在不在，一律用 `.get()`。**

**对应哪天**：D1（两数之和的哈希版本）、D10（和为 K 的子数组，`d.get(prefix, 0)`）、D12、D21

---

## 4 · `defaultdict` 与 `Counter`（D2 直接要用）

```python
import collections
```

| 工具 | 用途 | 一行示例 |
|---|---|---|
| `collections.defaultdict(list)` | 分组，省掉"键不存在就先建空列表"的判断 | `dd[k].append(v)` |
| `collections.defaultdict(int)` | 计数，省掉 `.get(k, 0)` | `dd[k] += 1` |
| `collections.Counter` | 词频统计 | `Counter(words)` |
| `Counter.most_common(n)` | 前 n 个高频 | `Counter(w).most_common(2)` |
| `Counter` 直接比相等 | **判断两个字符串是否异位词** | `Counter(s) == Counter(p)` |
| `collections.deque` | 双端队列（滑动窗口最大值要用） | `deque(maxlen=k)` |

**分组的标准写法**：

```python
dd = collections.defaultdict(list)
for k, v in pairs:
    dd[k].append(v)          # 不用判断 k 在不在
result = dict(dd)            # 转回普通 dict
```

> 对比一下用普通 dict 要写几行：
> ```python
> if k not in d:
>     d[k] = []
> d[k].append(v)
> ```
> **defaultdict 就是为这种模式生的。**

**对应哪天**：D2（49 异位词分组）、D9（438，`Counter` 相等比较）、D11（239，`deque`）、D12（76）

---

## 5 · set —— O(1) 成员检查

| 写法 | 作用 |
|---|---|
| `set(xs)` | 去重 |
| `s.add(x)` | 加入 |
| `s.remove(x)` | 删除，**不存在会 KeyError** |
| **`s.discard(x)`** | 删除，**不存在也不报错** |
| `x in s` | 判断存在，**O(1)** |
| `s & t` / `s \| t` / `s - t` | 交 / 并 / 差 |

**这一步是本文件最重要的概念**：

```python
3 in my_list    # O(n) —— 从头扫一遍
3 in my_set     # O(1) —— 直接算位置
```

**这就是"用空间换时间"的全部内容。** 多花 O(n) 的空间建一个 set/dict，换来每次查询从 O(n) 变 O(1)。

**对应哪天**：D3（最长连续序列，核心就是 set）、D8（无重复字符最长子串）、D17

---

## 6 · 字符串

| 写法 | 作用 |
|---|---|
| `"".join(sorted(s))` | **异位词的标准形式** ← D2 第一行 |
| `"-".join(xs)` | 用分隔符拼接列表 |
| `s.split()` | 按空白拆分 |
| `s.split(",")` | 按指定字符拆 |
| `s.lower()` / `s.upper()` | 大小写 |
| `s.strip()` | 去首尾空白 |
| `s.isalnum()` / `isdigit()` / `isalpha()` | 类型判断 |
| `len(s)` / `s[i]` / `s[1:3]` | 长度 / 取字符 / 切片 |

**三个坑**：

1. **字符串不可变** —— 没有 `s.sort()`、不能 `s[0] = "A"`。要 `list(s)` 改完再 `"".join()`。
2. **`join` 的方向**：`"分隔符".join(列表)`，不是 `列表.join("分隔符")`。
3. `"".join(sorted(w))` 里，`sorted` 返回的是**字符列表**（不是字符串），所以必须 join 回去。

**对应哪天**：D2、D8、D9、D12

---

## 7 · 惯用法（让代码从 20 行变 2 行）

| 写法 | 作用 |
|---|---|
| `[x*2 for x in xs if x % 2 == 0]` | 列表推导 |
| `{k: v for k, v in pairs}` | 字典推导 |
| `{x for x in xs}` | 集合推导 |
| `xs[::-1]` | 反转 |
| `a, b = b, a` | 交换 |
| `float("inf")` / `float("-inf")` | 正负无穷，配合 `min`/`max` |
| `x if cond else y` | 三元表达式 |
| `list(zip(*matrix))` | **矩阵转置** |
| `max(xs)` / `min(xs)` / `sum(xs)` / `len(xs)` | 聚合 |
| `all(...)` / `any(...)` | 全真 / 有一个真 |

**原地修改的坑（接下来三周最容易踩）**：

```python
def f(nums):
    nums = [0] * len(nums)      # ✗ 只改了局部变量，调用方的列表没变
    nums[:] = [0] * len(nums)   # ✓ 切片赋值才是真的原地修改
```

283（移动零）、189（轮转数组）、73（矩阵置零）都是"**原地**修改"题，写错会直接被判错，**而且不报错**。

**对应哪天**：D4（移动零）、D15（轮转数组）、D18（矩阵置零）、D19（螺旋矩阵）、D20（旋转图像）

---

## 8 · 双指针模板（D4–D7 连续四天都是它）

```python
left, right = 0, len(nums) - 1
while left < right:
    # 根据条件移动 left 或 right
    if 条件:
        left += 1
    else:
        right -= 1
```

**两个变体**：

| 类型 | 写法 |
|---|---|
| 对撞指针（从两端往中间） | `left, right = 0, n-1`，`left < right` |
| 快慢指针（同向） | `slow, fast = 0, 0`，`fast < n` |

---

## 9 · 复杂度直觉（背下来，面试会问）

| 操作 | 复杂度 |
|---|---|
| `x in list` / `list.index(x)` | O(n) |
| `x in set` / `x in dict` | **O(1)** |
| `sorted(xs)` | O(n log n) |
| `x in s`（字符串） | O(n) |
| `xs.append(x)` | O(1) 均摊 |
| `xs.pop()` | O(1) |
| `xs.pop(0)` / `xs.insert(0, x)` | **O(n)** ← 别在循环里用 |
| 切片 `xs[i:j]` | O(j-i) |

> 看到"把 list 当队列用、每次 `pop(0)`"就要警觉 —— 那会退化成 O(n²)。用 `collections.deque`。

---

## 10 · 接下来 21 天的工具对照

| 天 | 题目 | 会用到 |
|---|---|---|
| D2 | 49 异位词分组 | `defaultdict(list)` / `Counter`、`"".join(sorted(w))` |
| D3 | 128 最长连续序列 | `set`、`in` |
| D4 | 283 移动零 | 双指针、`nums[:] =` |
| D5 | 11 盛最多水的容器 | 对撞双指针、`min/max` |
| D6 | 15 三数之和 | `sorted()`、双指针、去重 |
| D7 | 42 接雨水 | 双指针 / 前后缀 `max` |
| D8 | 3 无重复最长子串 | `set`、`add`/`discard`、`max` |
| D9 | 438 找所有异位词 | `Counter` 相等比较 |
| D10 | 560 和为 K 的子数组 | `d.get(k, 0)` 前缀和 |
| D11 | 239 滑动窗口最大值 | **`collections.deque`** |
| D12 | 76 最小覆盖子串 | `Counter`、`float("inf")` |
| D14 | 56 合并区间 | **`sorted(key=lambda x: x[0])`** |
| D15 | 189 轮转数组 | 切片、`nums[:] =` |
| D18 | 73 矩阵置零 | 二维 list、`set` |
| D20 | 48 旋转图像 | **`list(zip(*matrix[::-1]))`** |

**标粗的那几个是"不知道就一定写不出来"的**，优先背。

---

## 11 · 补完的标准

1. `python_drill.py` 22/22 全绿
2. **第二天不看这份表，再跑一遍还是全绿**
3. 做 D2 的时候，能想到 `defaultdict(list)` 和 `"".join(sorted(w))`

第 2 条是关键。**认识 ≠ 会用**，而算法题考的是"能不能在 25 分钟内打出来"。

---

*相关：`python_drill.py`（练习）、`README.md`（记录格式）、`SCHEDULE.md`（100 天题号）*
