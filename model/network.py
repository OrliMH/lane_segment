import torch.nn as nn
import torchvision.models as models
from model.module import Block, SEBottleneck, SEDownBottleneck, Layer, SELayer, DownBottleneck, Bottleneck, CBAMBlock, CBAMBottleneck, CBAMResBlock, CBAMDownBottleneck, CBAMLayer
class ResNet101v2(nn.Module):
    def __init__(self):
        super(ResNet101v2, self).__init__()
        self.conv1 = Block(3, 64, 7, 3, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2_1 = DownBottleneck(64, 256, stride=1)
        self.conv2_2 = Bottleneck(256, 256)
        self.conv2_3 = Bottleneck(256, 256)
        self.layer3 = Layer(256, [512]*2, "resnet")
        self.layer4 = Layer(512, [1024]*23, "resnet")
        self.layer5 = Layer(1024, [2048]*3, "resnet")

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(self.pool1(f1))))
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        return [f1, f2, f3, f4, f5]

class UnetEncoder(nn.Module):
    def __init__(self):
        super(UnetEncoder, self).__init__()
        self.conv1 = Block(3, 64, 7, 3, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2_1 = SEDownBottleneck(64, 128, stride=1)
        self.conv2_2 = SEBottleneck(128, 128)
        self.conv2_3 = SEBottleneck(128, 128)
        self.layer3 = SELayer(128, [256]*2, "resnet")
        self.layer4 = SELayer(256, [512]*23, "resnet")
        self.layer5 = SELayer(512, [1024]*3, "resnet")

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(self.pool1(f1))))
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        return [f1, f2, f3, f4, f5]
class CBAMUnetEncoder(nn.Module):
    def __init__(self):
        super(CBAMUnetEncoder, self).__init__()
        self.conv1 = CBAMBlock(3, 64, 7, 3, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2_1 = CBAMDownBottleneck(64, 128, stride=1)
        self.conv2_2 = CBAMBottleneck(128, 128)
        self.conv2_3 = CBAMBottleneck(128, 128)
        self.layer3 = CBAMLayer(128, [256]*2, "resnet")
        self.layer4 = CBAMLayer(256, [512]*23, "resnet")
        self.layer5 = CBAMLayer(512, [1024]*3, "resnet")

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(self.pool1(f1))))
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        return [f1, f2, f3, f4, f5]
