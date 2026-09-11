"""
1. 两数之和   [简单]
日期：2026-09-11   用时：__ 分钟   限时：15 分钟   结果：独立做出

关键洞察：
    ★ 待填（一句话）
      提示：暴力解的内层循环在干什么？它在"找 target - nums[i] 这个值"。
      那么"找一个值"这件事，有没有比"把数组从头看一遍"更快的做法？
      把答案写成一句话 —— 就是这里要写的东西。

复杂度：
    我的解法：O(n²) 时间 / O(1) 空间      ← 两层循环，每个元素都要往后扫一遍
    最优解法：O(n)  时间 / O(n)  空间
    差距在哪：★ 待填（一句话。提示：多花的 O(n) 空间买到了什么？）

踩坑：
    ★ 待填（提示：内层循环为什么从 i+1 开始，而不是从 0 开始？
             这一条值得写下来。没有其他卡点就写"无"。）

备注：
    面试官几乎一定会追问"能不能优化到 O(n)"。暴力解过了不代表这题过了。
    有精力的话，今天就把哈希表版本也写一遍（写在同一文件里，加注释说明）。
"""

from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            target_number = target - nums[i]
            for j in range(i + 1, len(nums)):
                if nums[j] == target_number:
                    return [i, j]
