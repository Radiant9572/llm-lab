class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if len(nums)==0:
            return 0
        d = set(nums)
        l = 1
        ch = []
        for n in d:
            if n-1 in d:
                continue 
            while n+1 in d:
                l+=1
                n+=1
            ch.append(l)
            l = 1
        return max(ch)
#一开始我忘记经过sorted()方法之后得到的是一个列表，导致超时了。后来我把 d = sorted(set(nums))改为 d = set(nums)就通过了。