import torch
from model.unet import ResNetUNet

class Config():
    def __init__(self, nbclasses):
        super(Config, self).__init__()
        self.NUM_CLASSES = nbclasses
if __name__ == '__main__':
    input_ = torch.rand(2, 3, 5, 5)
    config = Config(3)

    unet = ResNetUNet(config)
    unet.forward(input_)
