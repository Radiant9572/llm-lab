import torch

x = torch.zeros(2,3,1)
x = x+2
y = torch.zeros(4)
y = y+3
print(x*y)