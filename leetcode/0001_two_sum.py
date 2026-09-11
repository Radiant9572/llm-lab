"""
1. 两数之和   [简单]
日期：2026-09-11   用时：5 分钟   限时：15 分钟   结果：独立做出

关键洞察：
    可以利用哈希表来找，但是这方面的代码我不会写
    → 待补：把哈希表版本写出来（O(n)），见文末「备注」

复杂度：
    我的解法：O(n²) 时间 / O(1) 空间
    最优解法：O(n)  时间 / O(n)  空间
    差距在哪：也许多花 O(n) 的空间是哈希表用来记录数组中的 n 个值，用空间换时间

踩坑：
    内层循环从 i+1 开始写，是因为查找到第 i 个值的时候，如果前 i-1 个数中有目标值的话，
    那么应该早就配对成功了，所以为了节省时间，直接从 i+1 开始查找

备注：
    面试官几乎一定会追问"能不能优化到 O(n)"。暴力解过了不代表这题过了。
    TODO：写哈希表版本，写在同一文件里并加注释说明思路。
"""

from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            target_number = target - nums[i]
            for j in range(i + 1, len(nums)):
                if nums[j] == target_number:
                    return [i, j]
