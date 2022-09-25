import torch.nn as nn
def init_weights(m):
    print("m:{}".format(m))
    if type(m) == nn.Linear:
        # m.weight.fill_(1.0)
        print("m.weight:{}".format(m.weight))
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
# net.apply(init_weights)
for m in net.modules():
    print(m)
