import torch
import torch.nn as nn
from model.network import ResNet101v2, UnetEncoder, CBAMUnetEncoder
from model.module import Block

class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out
class SEResUNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(SEResUNetConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_chans)

        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_chans)

        self.identityconv = nn.Conv2d(in_chans, out_chans, 1)

        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(out_chans, out_chans//4, 1, bias=False)
        self.fc2 = nn.Conv2d(out_chans//4, out_chans, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        

    def forward(self, x):
        # print(x.shape)
        identity = self.identityconv(x)  
    
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        # print(out.shape)
        squ = self.relu(self.fc1((self.global_pool(out))))
        squ = self.sigmoid(self.fc2(squ))
        # print(squ.shape)
        out = out*squ 
        # print(out.shape)
        out += identity

        # torch.Size([8, 2048, 24, 64])
        # torch.Size([8, 1024, 24, 64])
        # torch.Size([8, 1024, 1, 1])
        # torch.Size([8, 1024, 24, 64])

        return out

class SEResUNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, padding):
        super(SEResUNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = SEResUNetConvBlock(in_chans, out_chans, padding, True)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([crop1, up], dim=1)
        out = self.conv_block(out)

        return out
class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, padding):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_chans, out_chans, padding, True)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([crop1, up], dim=1)
        out = self.conv_block(out)

        return out

class ResNetUNet(nn.Module):
    def __init__(
        self,
        config
    ):
        super(ResNetUNet, self).__init__()
        self.n_classes = config.NUM_CLASSES
        self.padding = 1
        self.up_mode = 'upconv'
        assert self.up_mode in ('upconv', 'upsample')
        self.encode = ResNet101v2()
        prev_channels = 2048
        self.up_path = nn.ModuleList()
        for i in range(3):
            self.up_path.append(
                UNetUpBlock(prev_channels, prev_channels // 2, self.up_mode, self.padding)
            )
            prev_channels //= 2

        self.cls_conv_block1 = Block(prev_channels, 32)
        self.cls_conv_block2 = Block(32, 16)
        self.last = nn.Conv2d(16, self.n_classes, kernel_size=1)
        
        log = open('../init_.log', "w")
        for m in self.modules():
            log.write(str(m))
            log.write("\n")
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]
        blocks = self.encode(x)
        x = blocks[-1]
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])
        x = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(x)
        x = self.cls_conv_block1(x)
        x = self.cls_conv_block2(x)
        x = self.last(x)
        return x
class UpConvResNetUNet(nn.Module):
    def __init__(
        self,
        config
    ):
        super(UpConvResNetUNet, self).__init__()
        self.n_classes = config.NUM_CLASSES
        self.padding = 1
        self.up_mode = 'upconv'
        assert self.up_mode in ('upconv', 'upsample')
        self.encode = UnetEncoder()
        prev_channels = 1024
        self.up_path = nn.ModuleList()
        for i in range(4):
            self.up_path.append(
                SEResUNetUpBlock(prev_channels, prev_channels // 2, self.up_mode, self.padding)
            )
            prev_channels //= 2

        self.last_up = nn.ConvTranspose2d(prev_channels, 32, 7, 2, 3, 1)
        # self.cls_conv_block1 = Block(prev_channels, 32)
        self.cls_conv_block2 = Block(32, 16)
        self.last = nn.Conv2d(16, self.n_classes, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("input:")
        # print(x.shape)
        input_size = x.size()[2:]
        blocks = self.encode(x)
        # print("###"*8)
        # print(blocks[0].shape)
        # print(blocks[1].shape)
        # print(blocks[2].shape)
        # print(blocks[3].shape)
        # print(blocks[4].shape)
        # print("###"*8)
        x = blocks[-1]
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])
            # print("*"*8)
        # x = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(x)

        # print(x.shape)
        x = self.last_up(x)
        x = self.cls_conv_block2(x)
        x = self.last(x)
        # print(x.shape)
        return x
class CBAMResNetUNet(nn.Module):
    def __init__(
        self,
        config
    ):
        super(CBAMResNetUNet, self).__init__()
        self.n_classes = config.NUM_CLASSES
        self.padding = 1
        self.up_mode = 'upconv'
        assert self.up_mode in ('upconv', 'upsample')
        self.encode = CBAMUnetEncoder()
        prev_channels = 1024
        self.up_path = nn.ModuleList()
        for i in range(4):
            self.up_path.append(
                SEResUNetUpBlock(prev_channels, prev_channels // 2, self.up_mode, self.padding)
            )
            prev_channels //= 2

        self.last_up = nn.ConvTranspose2d(prev_channels, 32, 7, 2, 3, 1)
        # self.cls_conv_block1 = Block(prev_channels, 32)
        self.cls_conv_block2 = Block(32, 16)
        self.last = nn.Conv2d(16, self.n_classes, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("input:")
        # print(x.shape)
        input_size = x.size()[2:]
        blocks = self.encode(x)
        # print("###"*8)
        # print(blocks[0].shape)
        # print(blocks[1].shape)
        # print(blocks[2].shape)
        # print(blocks[3].shape)
        # print(blocks[4].shape)
        # print("###"*8)
        x = blocks[-1]
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])
            # print("*"*8)
        # x = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(x)

        # print(x.shape)
        x = self.last_up(x)
        x = self.cls_conv_block2(x)
        x = self.last(x)
        # print(x.shape)
        return x
