
import torch

x = torch.zeros(2,2,2)
x = x + 1
y = torch.ones(2,2,2)
y = y + 1
x[1,0, 0] = 8
x[1,1, 1] = 9
#print("x形状：",x)
#print("y形状：",y)
#print("x+y形状：",x+y)
#print("x*y形状：",x*y)
#print("x@y形状：",x@y)

s = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print(s.sum(dim=-1))                    # → tensor([ 6., 15.])   沿最后一维求和，该维消失
print(s.sum(dim=-1, keepdim=True))      # → tensor([[ 6.], [15.]]) 保留成长度 1，才能广播

print(s.max(dim=-1))                    # ⚠️ 返回的是个 namedtuple，不是张量！
print(s.max(dim=-1).values)             # 取最大值
print(s.max(dim=-1).indices)            # 取最大值的位置
print(s.max(dim=-1, keepdim=True).values)   # 值 + 保留维度
