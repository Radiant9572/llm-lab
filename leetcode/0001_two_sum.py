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
# ============================================================================
# ↓ 这版是参考题解，不是我独立写出来的
#   所以 docstring 的「结果」栏保持"看题解后做出"，不要改成"独立做出"
#
#   重写打卡：____ 月 ____ 日   用时 ____ 分钟   不看题解独立写完 ☐
#   打卡通过后，才把结果栏改成"独立做出"，并补一句"第一次卡在哪"
# ============================================================================
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        #哈希表
        #1.定义哈希表，key为元素值，value为数组下标
        dict1 = {}

        #2.遍历数组
        for index,item in enumerate(nums):
            #3.判断哈希表中有无符合要求的另一个数
            if (target-item) in dict1:
                return [index,dict1[target-item]]
            #4.没有则加入哈希表
            else:
                dict1[item] = index

        #5.未找到返回[]
        return []

#我尝试自己写出哈希表解法，但是用了15min分钟没有得到正确答案，原因是我不知道enumerate这个方法以及不懂哈希表的语法规则。
#
#（补充，2026-09-12）上面这段"我没写出来"是事实，保留。
#   但要注意：`enumerate` 只是让代码好看，真正卡住我的是
#   `dict1[target-item]` / `dict1[item] = index` 这套语法 ——
#   这正好是 python_drill.py 的 ex09/ex16 和 PYTHON_CHEATSHEET.md 第 3 节的内容。