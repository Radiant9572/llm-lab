"""
LeetCode Python 词汇练习 —— 22 题，约 40 分钟
========================================================================

为什么要做这个：
    你写不出 49. 字母异位词分组，不是算法问题 —— 是不知道 defaultdict 怎么写。
    这类卡点的特点是：**它是有限集，且可以一次补完。**

    下面 22 道题覆盖的工具，足够你走完 Hot 100 的前 21 天（数组与窗口阶段），
    也覆盖了整份 Hot 100 里 90% 的语法需求。

怎么用：
    1. 每道题的注释里写了「要什么」，把 `_todo("exNN")` 那一行整行替换成你的代码
    2. 跑 `python python_drill.py`，看到 [通过] 就对了
    3. **卡 3 分钟以上就翻 `PYTHON_CHEATSHEET.md`** —— 这不是算法题，
       想不出来没有价值，认识它、会用才是目的

过关标准：
    **22 题全绿，且第二天能不看 cheat sheet 重做一遍。**
========================================================================
"""

from __future__ import annotations

import collections


def _todo(where: str):
    """占位符。把这一行替换成你的代码，它就不会再被触发。"""
    raise NotImplementedError(f"{where} 还没填")


# ==========================================================================
# Part 1 · 遍历与下标
# ==========================================================================
def ex01_enumerate() -> list[tuple[int, str]]:
    """用 enumerate 把 ["a","b","c"] 变成 [(0,"a"), (1,"b"), (2,"c")]。

    提示：enumerate(x) 每次吐出一个 (下标, 元素) 元组，可以直接解包
         for i, ch in enumerate(x):
    """
    # 填空 1/22 —— 把下一行替换成你的代码

    return [(i,x) for i,x in enumerate(["a","b","c"])]


def ex02_enumerate_start() -> list[tuple[int, str]]:
    """同上，但下标从 1 开始：[(1,"a"), (2,"b"), (3,"c")]。"""
    # 填空 2/22
    return [(i,x) for i,x in enumerate(["a","b","c"],1)]


def ex03_zip() -> list[tuple[int, str]]:
    """用 zip 把 [1,2,3] 和 ["a","b","c"] 配成 [(1,"a"), (2,"b"), (3,"c")]。

    提示：zip 返回的是迭代器，要 list() 一下才是列表
    """
    # 填空 3/22
    return list((a,b) for a,b in zip([1,2,3],["a","b","c"]))


def ex04_reverse_range() -> list[int]:
    """用 range 倒序产生 [3, 2, 1, 0]。

    提示：range(stop, stop, -1)
    """
    # 填空 4/22
    return [i for i in range(3,-1,-1)]


# ==========================================================================
# Part 2 · 排序（Hot 100 里出现率最高的工具之一）
# ==========================================================================
def ex05_sorted_returns_new(original: list[int]) -> tuple[list[int], list[int]]:
    """关键区别：`sorted()` 返回**新列表**，不改原列表；`list.sort()` 是**原地**排序。

    返回 (排序结果, 排序后原列表) —— 所以第二个应该还是原样。
    """
    # 填空 5/22 —— 把下一行替换成你的代码（提示：sorted(original)）
    srt = sorted(original)
    return srt, original


def ex06_sorted_by_length(words: list[str]) -> list[str]:
    """按字符串长度升序排序。长度相同时保持原顺序。

    提示：key=len —— key 接收一个**函数**，不是调用结果
    """
    # 填空 6/22
    return sorted(words,key=len)


def ex07_sorted_by_key_lambda(pairs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """把这个 [(name, score)] 按 score **从高到低**排序。

    提示：key=lambda p: p[1]  配合 reverse=True
    """
    # 填空 7/22
    return sorted(pairs,key=lambda p: p[1],reverse=True)


def ex08_sorted_dict_by_value(counts: dict[str, int]) -> list[tuple[str, int]]:
    """把 {词: 次数} 按次数从高到低排成 [(词, 次数)]。

    提示：d.items() 给出 (k, v) 对，key=lambda kv: kv[1]
        （这一步是 692、347 那类"前 K 个高频元素"题的通用开头）
    """
    # 填空 8/22
    return sorted(counts.items(),key = lambda p: p[1],reverse=True)


# ==========================================================================
# Part 3 · dict 基础（"用空间换时间"的那个空间）
# ==========================================================================
def ex09_get_default() -> int:
    """从 {"a": 1} 里取 "b"，不存在时返回 0。

    提示：d.get(key, default) —— **不要写 d["b"]，那会直接 KeyError**
    """
    # 填空 9/22
    return {"a": 1}.get("b", 0)


def ex10_count_words(words: list[str]) -> dict[str, int]:
    """统计词频，**不许用 Counter**。

    这是最经典的 dict 惯用法。提示：
        for w in words:
            d[w] = d.get(w, 0) + 1
    """
    # 填空 10/22
    d = {}
    for w in words:
        d[w] = d.get(w,0)+1
    return d


def ex11_group_by_setdefault(pairs: list[tuple[str, int]]) -> dict[str, list[int]]:
    """把 [(k, v)] 按 k 分组：{"a": [1, 3], "b": [2]}。

    提示：d.setdefault(k, []).append(v) —— 一行顶三行
    """
    # 填空 11/22
    d = {}
    for w in pairs:
        d.setdefault(w[0],[]).append(w[1])
    return d


# ==========================================================================
# Part 4 · defaultdict 与 Counter（D2 会直接用到）
# ==========================================================================
def ex12_defaultdict_group(words: list[str]) -> dict[str, list[str]]:
    """用 collections.defaultdict(list) 把词按**首字母**分组。

    提示：dd = collections.defaultdict(list)
         然后 dd[key].append(...) 不用先判断 key 存不存在
         最后要 dict(dd) 转回普通 dict 才能和期望值比较
    """
    # 填空 12/22
    dd = collections.defaultdict(list)
    for w in words:
        dd[w[0]].append(w)
    return dict(dd)


def ex13_counter_top2(words: list[str]) -> list[tuple[str, int]]:
    """用 collections.Counter 统计词频，取出现最多的前 2 个。

    提示：Counter(words).most_common(2)
    """
    # 填空 13/22
    return collections.Counter(words).most_common(2)


def ex14_counter_equal(s: str, p: str) -> bool:
    """判断 s 和 p 的字母构成是否完全相同（异位词）。

    提示：Counter(s) == Counter(p) —— 这也是 D9（438 题）的核心工具
    """
    # 填空 14/22
    return collections.Counter(s) == collections.Counter(p) 


# ==========================================================================
# Part 5 · set（D3 会直接用到）
# ==========================================================================
def ex15_set_dedupe(nums: list[int]) -> list[int]:
    """去重，返回排序后的列表。

    提示：set(nums) 去重，但 set 无序，所以外面再套一层 sorted()
    """
    # 填空 15/22
    return sorted(set(nums))


def ex16_in_check(nums: list[int]) -> bool:
    """用 set 判断 3 在不在 nums 里。

    这一步的意义：`3 in list` 是 O(n)，`3 in set` 是 O(1)。
    **这就是你昨天写的"用空间换时间"的本质。**
    """
    # 填空 16/22
    s = set(nums)
    return 3 in s


def ex17_set_ops(a: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    """返回 (交集, 并集)，都要排序后的列表。

    提示：集合运算符 & 和 |
    """
    # 填空 17/22
    s_a = set(a)
    s_b = set(b)
    s_1 = s_a & s_b
    s_2 = s_a | s_b
    return sorted(s_1),sorted(s_2)


# ==========================================================================
# Part 6 · 字符串（D2 会直接用到）
# ==========================================================================
def ex18_canonical(w: str) -> str:
    """把单词转成"字母异位词的标准形式"：字母排序后拼回字符串。

    "eat" / "tea" / "ate" 都应该得到 "aet"。
    提示：sorted("eat") 得到 ['a','e','t']，要用 "".join(...) 拼回去
    **这是 49 题的第一行。**

    注意：字符串没有 .sort()，因为字符串不可变
    """
    # 填空 18/22
    n_w = "".join(sorted(w))
    return n_w


def ex19_split_join(s: str) -> str:
    """把 "a b c" 用 split 拆开、再用 "-" join 起来，得到 "a-b-c"。

    提示：join 是**分隔符.join(列表)**，别写反了
    """
    # 填空 19/22
    n_s = "-".join(s.split())
    return n_s


def ex20_str_immutable(s: str) -> str:
    """把 s 的第 0 个字符换成大写（其余不变）。

    提示：字符串不可变，必须先 list(s) 改完再 "".join() 回来
    """
    l = list(s)
    l[0] = l[0].upper()
    n_s = "".join(l)
    return n_s


# ==========================================================================
# Part 7 · 惯用法（这些能让你的代码从 20 行变成 2 行）
# ==========================================================================
def ex21_list_comp(nums: list[int]) -> list[int]:
    """用列表推导取出所有偶数，并各乘 2。

    提示：[f(x) for x in xs if 条件]
    """
    # 填空 21/22
    return [2*x for x in nums if x%2==0]


def ex22_slice_reverse(nums: list[int]) -> list[int]:
    """用切片把列表反转。

    提示：nums[::-1]
    """
    # 填空 22/22
    return nums[::-1]


# ==========================================================================
# 附加（不计入 22 题，但这几个你下周就会用到，先看着）
# ==========================================================================
def extra_inplace_modify(nums: list[int]) -> None:
    """把 nums 里的负数全部变成 0，**必须在原列表上改**。

    关键：`nums = [x if x > 0 else 0 for x in nums]` 是**错的** ——
         它只是让局部变量 nums 指向了新列表，调用方的列表没变。
         要写 `nums[:] = [...]`。

    283（移动零）、189（轮转数组）、73（矩阵置零）都是"原地修改"题，
    这类题搞错这一点会直接被判错。
    """
    nums[:] = [x if x > 0 else 0 for x in nums]


def extra_transpose(matrix: list[list[int]]) -> list[list[int]]:
    """转置矩阵。

    `zip(*matrix)` 是最漂亮的写法：* 把行拆开，zip 把同列的元素聚到一起。
    48（旋转图像）就是 `[list(r) for r in zip(*matrix[::-1])]`。
    """
    return [list(row) for row in zip(*matrix)]


# ==========================================================================
# 测试
# ==========================================================================
TESTS = [
    ("ex01 enumerate",              ex01_enumerate,             (),                            [(0, "a"), (1, "b"), (2, "c")]),
    ("ex02 enumerate start=1",      ex02_enumerate_start,       (),                            [(1, "a"), (2, "b"), (3, "c")]),
    ("ex03 zip",                    ex03_zip,                   (),                            [(1, "a"), (2, "b"), (3, "c")]),
    ("ex04 range 倒序",             ex04_reverse_range,         (),                            [3, 2, 1, 0]),
    ("ex05 sorted 返回新列表",      ex05_sorted_returns_new,    ([3, 1, 2],),                  ([1, 2, 3], [3, 1, 2])),
    ("ex06 按长度排序",             ex06_sorted_by_length,      (["bbb", "a", "cc"],),         ["a", "cc", "bbb"]),
    ("ex07 key=lambda 降序",        ex07_sorted_by_key_lambda,  ([("a", 3), ("b", 9), ("c", 5)],), [("b", 9), ("c", 5), ("a", 3)]),
    ("ex08 字典按值排序",           ex08_sorted_dict_by_value,  ({"a": 3, "b": 9, "c": 5},),   [("b", 9), ("c", 5), ("a", 3)]),
    ("ex09 dict.get 默认值",        ex09_get_default,           (),                            0),
    ("ex10 手写词频统计",           ex10_count_words,           (["a", "b", "a"],),            {"a": 2, "b": 1}),
    ("ex11 setdefault 分组",        ex11_group_by_setdefault,   ([("a", 1), ("b", 2), ("a", 3)],), {"a": [1, 3], "b": [2]}),
    ("ex12 defaultdict 分组",       ex12_defaultdict_group,     (["apple", "avocado", "banana"],), {"a": ["apple", "avocado"], "b": ["banana"]}),
    ("ex13 Counter.most_common",    ex13_counter_top2,          (["a", "b", "a", "c", "a", "b"],), [("a", 3), ("b", 2)]),
    ("ex14 Counter 相等比较",       ex14_counter_equal,         ("anagram", "nagaram"),        True),
    ("ex15 set 去重",               ex15_set_dedupe,            ([3, 1, 3, 2, 1],),            [1, 2, 3]),
    ("ex16 set 成员检查",           ex16_in_check,              ([1, 2, 3],),                  True),
    ("ex17 交并集",                 ex17_set_ops,               ([1, 2, 3], [2, 3, 4]),        ([2, 3], [1, 2, 3, 4])),
    ("ex18 异位词规范形式",         ex18_canonical,             ("tea",),                      "aet"),
    ("ex19 split / join",           ex19_split_join,            ("a b c",),                    "a-b-c"),
    ("ex20 字符串不可变",           ex20_str_immutable,         ("abc",),                      "Abc"),
    ("ex21 列表推导",               ex21_list_comp,             ([1, 2, 3, 4],),               [4, 8]),
    ("ex22 切片反转",               ex22_slice_reverse,         ([1, 2, 3],),                  [3, 2, 1]),
]


def run_tests():
    print()
    print("=" * 70)
    print("词汇练习")
    print("=" * 70)
    passed = failed = todo = 0
    for name, fn, args, expected in TESTS:
        try:
            got = fn(*args)
            if got == expected:
                print(f"  [通过]  {name}")
                passed += 1
            else:
                print(f"  [失败]  {name}")
                print(f"          期望 {expected!r}")
                print(f"          实际 {got!r}")
                failed += 1
        except NotImplementedError:
            print(f"  [待填]  {name}")
            todo += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [报错]  {name}")
            print(f"          {type(e).__name__}: {e}")
            failed += 1
    print("-" * 70)
    print(f"通过 {passed} / 失败 {failed} / 待填 {todo}   (共 {len(TESTS)} 项)")
    return passed, failed, todo


def demo_extras():
    print()
    print("=" * 70)
    print("附加：两个容易踩的写法（已给实现，看懂即可）")
    print("=" * 70)

    nums = [3, -1, 4, -1, 5]
    print(f"  原地修改前: {nums}")
    extra_inplace_modify(nums)
    print(f"  原地修改后: {nums}   ← nums[:] = [...] 真的改了原列表")

    bad = [3, -1, 4]
    def wrong(lst):
        lst = [x if x > 0 else 0 for x in lst]      # noqa: F841
    wrong(bad)
    print(f"  用 lst = [...] 的效果: {bad}   ← 调用方的列表没变，这是最常见的静默 bug")

    m = [[1, 2, 3], [4, 5, 6]]
    print(f"\n  原矩阵: {m}")
    print(f"  转置:   {extra_transpose(m)}")


def quiz():
    print()
    print("=" * 70)
    print("学完必须能答的 8 个问题")
    print("=" * 70)
    qs = [
        "sorted(x) 和 x.sort() 的区别是什么？（返回值 / 是否改原对象）",
        "为什么 3 in my_list 慢而 3 in my_set 快？复杂度各是多少？",
        "d[k] 和 d.get(k) 在键不存在时分别会发生什么？",
        "defaultdict(list) 比普通 dict 省掉了哪一步判断？",
        "Counter 除了统计词频，还能怎么直接比较两个字符串是否为异位词？",
        '为什么不能写 s.sort()？（提示：可变性）',
        '"".join(sorted(w)) 这一步在解 49 题时起什么作用？',
        '为什么 nums = [x for x in nums ...] 改不了调用方的列表，而 nums[:] = [...] 可以？',
    ]
    for i, q in enumerate(qs, 1):
        print(f"  {i}. {q}")
    print()
    print("  第 8 题是接下来三周最容易踩的坑：283 移动零、189 轮转数组、")
    print("  73 矩阵置零 都是「原地修改」题，写错了会直接被判错而且不报错。")


def main():
    passed, failed, todo = run_tests()
    demo_extras()
    quiz()
    print()
    print("=" * 70)
    if failed == 0 and todo == 0:
        print("22/22 全绿。明天不看 cheat sheet 再跑一遍，还是全绿就真的会了。")
    elif failed == 0:
        print(f"还有 {todo} 题待填。继续 —— 卡 3 分钟以上就翻 PYTHON_CHEATSHEET.md。")
    else:
        print("有报错的项。看报错信息，它通常会直接告诉你缺哪个方法名。")
    print("=" * 70)


if __name__ == "__main__":
    main()
