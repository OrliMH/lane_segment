
import torch 

n, h, w, c = 2, 5, 5, 3
a = torch.ones(n, h*w, c)
print(torch.sum(torch.exp(a), 2).shape)
print(torch.sum(torch.exp(a), 2))
print(torch.exp(a)/torch.sum(torch.exp(a), 2).reshape(n, h*w, -1))
print((torch.exp(a)/torch.sum(torch.exp(a), 2).reshape(n, h*w, -1)).shape)
