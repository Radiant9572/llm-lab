# `python_drill.py` 参考答案

> 建议顺序：**先自己填 → 卡 3 分钟翻 `PYTHON_CHEATSHEET.md` → 还不行再看这里。**
> 这份文件里每道题都写了"等价写法"，那些不是次优解，是同样正确的另一条路——认全了读别人代码才不吃力。

---

## Part 1 · 遍历与下标

**ex01 / ex02 enumerate**

```python
return [(i, ch) for i, ch in enumerate(["a", "b", "c"])]        # ex01
return [(i, ch) for i, ch in enumerate(["a", "b", "c"], 1)]     # ex02
```

等价写法（更常见的循环写法）：

```python
out = []
for i, ch in enumerate(["a", "b", "c"]):
    out.append((i, ch))
return out
```

> `enumerate(x)` 每次吐出一个 `(下标, 元素)`，可以直接解包成两个变量。

**ex03 zip**

```python
return list(zip([1, 2, 3], ["a", "b", "c"]))
```

> ⚠️ `zip` 返回**迭代器**，打印是一串内存地址。必须 `list()` 一下。

**ex04 range 倒序**

```python
return list(range(3, -1, -1))
```

> 三个参数是 `range(start, stop, step)`，**stop 取不到**。所以想包含 0 就得写到 `-1`。

---

## Part 2 · 排序

**ex05 sorted vs sort**

```python
srt = sorted(original)
return srt, original
```

| | 返回 | 改原列表 |
|---|---|---|
| `sorted(x)` | 新列表 | ✗ |
| `x.sort()` | `None` | ✓ |

> ⚠️ `x = xs.sort()` 是经典 bug，`x` 会是 `None`。

**ex06 按长度排序**

```python
return sorted(words, key=len)
```

> ⚠️ 是 `key=len`（函数本身），**不是** `key=len()`。

**ex07 key=lambda 降序**

```python
return sorted(pairs, key=lambda p: p[1], reverse=True)
```

**ex08 字典按值排序**

```python
return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
```

> `d.items()` 给出 `(k, v)` 对。这是"前 K 个高频元素"那类题的通用开头。
> 等价写法：`sorted(counts, key=counts.get, reverse=True)` 也能排，但拿不到值。

---

## Part 3 · dict

**ex09 get 默认值**

```python
return {"a": 1}.get("b", 0)
```

> ⚠️ `d["b"]` 会直接 `KeyError`。**不确定键在不在，一律 `.get()`。**

**ex10 手写词频统计**

```python
d = {}
for w in words:
    d[w] = d.get(w, 0) + 1
return d
```

**ex11 setdefault 分组**

```python
d = {}
for k, v in pairs:
    d.setdefault(k, []).append(v)
return d
```

> 一行顶三行。`setdefault(k, [])`：键不存在就用 `[]` 建一个并返回它，存在就返回已有的值。

---

## Part 4 · defaultdict 与 Counter

**ex12 defaultdict 分组**

```python
dd = collections.defaultdict(list)
for w in words:
    dd[w[0]].append(w)
return dict(dd)
```

> 对比普通 dict 的写法：
> ```python
> if w[0] not in dd:
>     dd[w[0]] = []
> dd[w[0]].append(w)
> ```
> **`defaultdict` 就是为这种模式生的。**

**ex13 Counter.most_common**

```python
return collections.Counter(words).most_common(2)
```

> 返回的是 `[(元素, 次数)]`，已经是降序。并列时顺序不保证。

**ex14 Counter 相等比较**

```python
return collections.Counter(s) == collections.Counter(p)
```

> 这一行就是"判断两个字符串是否异位词"。
> **等价写法**（手写计数）：`all(s.count(ch) == p.count(ch) for ch in set(s))` ——
> 能跑，但每个 `count` 都是 O(n)，整体退化。**Counter 的写法是 O(n)。**

---

## Part 5 · set

**ex15 去重**

```python
return sorted(set(nums))
```

> ⚠️ `set` 无序，`list(set(nums))` 的顺序**不保证**。要排序结果就必须套 `sorted()`。

**ex16 成员检查**

```python
return 3 in set(nums)
```

> 这一步的意义：`3 in list` 是 O(n)，`3 in set` 是 O(1)。
> **这就是"用空间换时间"的全部内容** —— 也是你在 D1 记录里写对的那句话。

**ex17 交并集**

```python
a, b = set(a), set(b)
return sorted(a & b), sorted(a | b)
```

> 等价写法：`a.intersection(b)` / `a.union(b)`。

---

## Part 6 · 字符串

**ex18 异位词规范形式**

```python
return "".join(sorted(w))
```

> **这是 49 题的第一行。** `sorted("tea")` 返回 `['a','e','t']`（字符列表，不是字符串），所以必须 join 回去。
> `"eat"` / `"tea"` / `"ate"` 都得到 `"aet"` —— 于是"同一组异位词"变成了"同一个 key"，分组问题就退化成普通的字典分组。

**ex19 split / join**

```python
return "-".join(s.split())
```

> ⚠️ 方向别写反：是 **`"分隔符".join(列表)`**，不是 `列表.join("分隔符")`。

**ex20 字符串不可变**

```python
ch = list(s)
ch[0] = ch[0].upper()
return "".join(ch)
```

> ⚠️ 字符串没有 `.sort()`，也不能 `s[0] = "A"`。**它是不可变的。**
> 这是最容易被忽略的一条：看到 `s.sort()` 报 `AttributeError` 就该想到"字符串不可变，先转 list"。

---

## Part 7 · 惯用法

**ex21 列表推导**

```python
return [x * 2 for x in nums if x % 2 == 0]
```

**ex22 切片反转**

```python
return nums[::-1]
```

> `[::-1]` 返回**新列表**。要原地反转用 `nums.reverse()` 或 `nums[:] = nums[::-1]`。

---

## 附加题答案

**`extra_inplace_modify` —— 原地修改的坑**

```python
nums[:] = [0 if x < 0 else x for x in nums]   # ✓
nums = [0 if x < 0 else x for x in nums]      # ✗ 调用方的列表没变
```

| 写法 | 效果 |
|---|---|
| `nums = [...]` | 只是让**局部变量**指向新列表。函数返回后调用方拿到的还是旧列表 |
| `nums[:] = [...]` | **切片赋值**：把新内容写回原列表对象。调用方看得到 |

**接下来三周会连着撞上三次**：D4 移动零、D15 轮转数组、D18 矩阵置零——都是"原地修改"题，**写错不报错，但直接判错**。

**`extra_transpose` —— 矩阵转置**

```python
return [list(row) for row in zip(*matrix)]
```

> `*matrix` 把每一行拆成独立参数，`zip` 就把同列的元素聚到一起。
> D20（旋转图像）的答案就是 `[list(r) for r in zip(*matrix[::-1])]`：
> **先上下翻转，再转置 = 顺时针旋转 90°。**

---

## 自测题答案

1. `sorted(x)` 返回**新列表**、不改原列表；`x.sort()` **原地**排序、返回 `None`。
2. `in list` 是 **O(n)**（从头扫），`in set` 是 **O(1)**（算哈希）。
3. `d[k]` 键不存在 → `KeyError`；`d.get(k)` → 返回 `None`（或你给的默认值）。
4. 省掉了 **"键不存在就先建一个空列表"** 那两行判断。
5. `Counter(s) == Counter(p)` 直接比相等，O(n)。
6. 字符串**不可变**，没有 `.sort()` 方法。
7. 它把"同一组异位词"映射到**同一个 key**，于是分组问题退化成普通字典分组。
8. `nums = [...]` 只重绑定了局部变量；`nums[:] = [...]` 是**切片赋值**，写回原列表对象。

---

## 最后一步

22 题全绿之后，**第二天合上所有文档再跑一遍。**

第 2 遍还是全绿，这套词汇才算真的进你手里了。**认识 ≠ 会用**，而面试考的是"能不能在 25 分钟内打出来"。
