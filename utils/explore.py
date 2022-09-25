
import numpy as np
import torch

a = [1,2,3]
a = np.array(a, dtype=np.long)
print(a.dtype)
new_a = torch.from_numpy(a.copy())
print(new_a.dtype)

another_a = new_a.type(torch.LongTensor)
print(another_a.dtype)
