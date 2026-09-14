class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        number = 0
        for n in nums:
            if n==0:
                number +=1
        nums[:]=[x for x in nums if x!=0]
        while number>0:
            nums.append(0)
            number -=1
        return nums
#nums[:]=[x for x in nums if x!=0]，这一句是我搜索得到的，因为我不知道如何原地修改数组