import torch
x = torch.randn(10000,10000).cuda()
y = torch.matmul(x,x)
print("OK")
