import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch 

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False) # 7 1 3 1 same conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # n 1 h w
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x) # n 1 h w
        x = self.sigmoid(x)
        return x

class Block(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out
class CBAMBlock(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(CBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.chan = ChannelAttention(out_ch)
        self.space = SpatialAttention()
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.chan(out)
        out = self.space(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return out
class CBAMResBlock(nn.Module):
    def __init__(self, in_ch,out_ch, kernel_size=3, padding=1, stride=1):
        super(CBAMResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.chan = ChannelAttention(out_ch)
        self.space = SpatialAttention()

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.chan(out)
        out = self.space(out)
        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(Bottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out
class CBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(CBAMBottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)
        self.chan = ChannelAttention(out_chans)
        self.space = SpatialAttention()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.chan(out)
        out = self.space(out)
        out += identity
        return out


class DownBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=2):
        super(DownBottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0, stride=stride)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, padding=0, stride=stride)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        identity = self.conv1(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out
class CBAMDownBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=2):
        super(CBAMDownBottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0, stride=stride)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, padding=0, stride=stride)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)
        self.chan = ChannelAttention(out_chans)
        self.space = SpatialAttention()

    def forward(self, x):
        identity = self.conv1(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.chan(out)
        out = self.space(out)
        out += identity
        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(SEBottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(out_chans, out_chans//4, 1, bias=False)
        self.fc2 = nn.Conv2d(out_chans//4, out_chans, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        squ = self.global_pool(out)
        squ = self.fc1(squ)
        squ = self.relu(squ)
        squ = self.fc2(squ)
        squ = self.sigmoid(squ)
        out = out*squ
        out += identity
        return out


class SEDownBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=2):
        super(SEDownBottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0, stride=stride)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, padding=0, stride=stride)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(out_chans, out_chans//4, 1, bias=False)
        self.fc2 = nn.Conv2d(out_chans//4, out_chans, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.conv1(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        squ = self.global_pool(out)
        squ = self.fc1(squ)
        squ = self.relu(squ)
        squ = self.fc2(squ)
        squ = self.sigmoid(squ)
        out = out*squ
        out += identity
        return out


def make_layers(in_channels, layer_list, name="vgg"):
    layers = []
    if name == "vgg":
        for v in layer_list:
            layers += [Block(in_channels, v)]
            in_channels = v
    elif name == "resnet":
        layers += [DownBottleneck(in_channels, layer_list[0])]
        in_channels = layer_list[0]
        for v in layer_list[1:]:
            layers += [Bottleneck(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list, net_name):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list, name=net_name)

    def forward(self, x):
        out = self.layer(x)
        return out
def make_selayers(in_channels, layer_list, name="vgg"):
    layers = []
    if name == "vgg":
        for v in layer_list:
            layers += [Block(in_channels, v)]
            in_channels = v
    elif name == "resnet":
        layers += [SEDownBottleneck(in_channels, layer_list[0])]
        in_channels = layer_list[0]
        for v in layer_list[1:]:
            layers += [SEBottleneck(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)


class SELayer(nn.Module):
    def __init__(self, in_channels, layer_list, net_name):
        super(SELayer, self).__init__()
        self.layer = make_selayers(in_channels, layer_list, name=net_name)

    def forward(self, x):
        out = self.layer(x)
        return out
def make_cbamlayers(in_channels, layer_list, name="vgg"):
    layers = []
    if name == "vgg":
        for v in layer_list:
            layers += [CBAMBlock(in_channels, v)]
            in_channels = v
    elif name == "resnet":
        layers += [CBAMDownBottleneck(in_channels, layer_list[0])]
        in_channels = layer_list[0]
        for v in layer_list[1:]:
            layers += [CBAMBottleneck(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)


class CBAMLayer(nn.Module):
    def __init__(self, in_channels, layer_list, net_name):
        super(CBAMLayer, self).__init__()
        self.layer = make_cbamlayers(in_channels, layer_list, name=net_name)

    def forward(self, x):
        out = self.layer(x)
        return out


class ASPP(nn.Module):

    def __init__(self, in_chans, out_chans, rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(out_chans)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_chans * 5, out_chans, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5_avg(x)
        global_feature = self.branch5_relu(self.branch5_bn(self.branch5_conv(global_feature)))
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result