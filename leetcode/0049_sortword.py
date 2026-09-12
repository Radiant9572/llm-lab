'''
1. 字母异位分组   [中等]
日期：2026-09-12   用时：15 分钟   限时：15 分钟   结果：没写出来

关键洞察：
    没有思路
    → 待补：把哈希表版本写出来（O(n)），见文末「备注」

复杂度：
   
踩坑：
'''   


class Solution:

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

        d = {}  #

        for s in strs: #ps：我经常会写for i in range(len(strs)), 然后用strs[i]来表示，但是这样太麻烦了，要改正
            sorted_s = ''.join(sorted(s))  # 把 s 排序，作为 dict 的 key
# ps：我知道sorted方法，但是不知道这样会得到一个列表，所以还需要''.join()
            if sorted_s not in d:  # 首次遇到 sorted_s

                d[sorted_s] = []  # 创建列表
# ps：这是我想不到也是困住我的一句代码，我不知道将排序后相同的字符串聚在同一个列表中
            d[sorted_s].append(s)  # 排序后相同的字符串，保存到同一组中
#ps：同理，我想不到这一句代码，因为我之前根本没有见过
        return list(d.values())  # 哈希表的所有 value 就是分组结果

#