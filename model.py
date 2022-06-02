import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from util import *

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class HICSATUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, CA, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
        self.sp = SpatialAttention()
        self.ch = CA
        self.sigmoid = nn.Sigmoid()
        self.conv_1by1 = nn.Conv2d(2, 1, 1, stride=1, padding=0, bias=False)

    def forward(self, x1, x2, hint):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd


        # x2 attention
        x2 = x2 * self.sp(x2) * self.ch(x2) * self.sigmoid(self.conv_1by1(hint))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# from CBAM : https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 7, stride=1, padding=7//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class HINT_CSATUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(HINT_CSATUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.ch1 = SpatialAttention(32)
        self.ch2 = SpatialAttention(64)
        self.ch3 = SpatialAttention(128)
        self.ch4 = SpatialAttention(256)

        self.resize4 = transforms.Resize((32, 32))
        self.resize3 = transforms.Resize((64, 64))
        self.resize2 = transforms.Resize((128, 128))
        self.resize1 = transforms.Resize((256, 256))

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = HICSATUp(1024, 512 // factor, self.ch1, bilinear)
        self.up2 = HICSATUp(512, 256 // factor, self.ch2, bilinear)
        self.up3 = HICSATUp(256, 128 // factor, self.ch3, bilinear)
        self.up4 = HICSATUp(128, 64, self.ch4, bilinear)
        self.outc = OutConv(64, n_classes)


    def forward(self, x, hint):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        hint_1 = self.resize1(hint)
        hint_2 = self.resize2(hint)
        hint_3 = self.resize3(hint)
        hint_4 = self.resize4(hint)

        x = self.up1(x5, x4, hint_4)
        x = self.up2(x, x3, hint_3)
        x = self.up3(x, x2, hint_2)
        x = self.up4(x, x1, hint_1)
        logits = self.outc(x)
        return logits