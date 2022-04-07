import torch
import torch.nn as nn


# Source https://github.com/wanghuanphd/MDvsFA_cGAN
class UCAN64(nn.Module):
    def __init__(self, in_channels:int = 3):
        super(UCAN64, self).__init__()
        self.in_channels = in_channels
        chn = 64
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.leakyrelu5 = nn.LeakyReLU(0.2)
        self.leakyrelu6 = nn.LeakyReLU(0.2)
        self.leakyrelu7 = nn.LeakyReLU(0.2)
        self.leakyrelu8 = nn.LeakyReLU(0.2)
        self.leakyrelu9 = nn.LeakyReLU(0.2)
        self.leakyrelu10 = nn.LeakyReLU(0.2)
        self.leakyrelu11 = nn.LeakyReLU(0.2)
        self.leakyrelu12 = nn.LeakyReLU(0.2)
        self.leakyrelu13 = nn.LeakyReLU(0.2)

        self.g2_conv1 = nn.Conv2d(self.in_channels, chn, 3, dilation=1, padding=1)
        self.g2_conv2 = nn.Conv2d(chn, chn, 3, dilation=2, padding=2)
        self.g2_conv3 = nn.Conv2d(chn, chn, 3, dilation=4, padding=4)
        self.g2_conv4 = nn.Conv2d(chn, chn, 3, dilation=8, padding=8)
        self.g2_conv5 = nn.Conv2d(chn, chn, 3, dilation=16, padding=16)
        self.g2_conv6 = nn.Conv2d(chn, chn, 3, dilation=32, padding=32)
        self.g2_conv7 = nn.Conv2d(chn, chn, 3, dilation=64, padding=64)
        self.g2_conv8 = nn.Conv2d(chn, chn, 3, dilation=32, padding=32)
        self.g2_conv9 = nn.Conv2d(chn*2, chn, 3, dilation=16, padding=16)
        self.g2_conv10 = nn.Conv2d(chn*2, chn, 3, dilation=8, padding=8)
        self.g2_conv11 = nn.Conv2d(chn*2, chn, 3, dilation=4, padding=4)
        self.g2_conv12 = nn.Conv2d(chn*2, chn, 3, dilation=2, padding=2)
        self.g2_conv13 = nn.Conv2d(chn*2, chn, 3, dilation=1, padding=1)
        self.g2_conv14 = nn.Conv2d(chn, 1,   1, dilation=1)

        self.g2_bn1 = nn.BatchNorm2d(chn)
        self.g2_bn2 = nn.BatchNorm2d(chn)
        self.g2_bn3 = nn.BatchNorm2d(chn)
        self.g2_bn4 = nn.BatchNorm2d(chn)
        self.g2_bn5 = nn.BatchNorm2d(chn)
        self.g2_bn6 = nn.BatchNorm2d(chn)
        self.g2_bn7 = nn.BatchNorm2d(chn)
        self.g2_bn8 = nn.BatchNorm2d(chn)
        self.g2_bn9 = nn.BatchNorm2d(chn)
        self.g2_bn10 = nn.BatchNorm2d(chn)
        self.g2_bn11 = nn.BatchNorm2d(chn)
        self.g2_bn12 = nn.BatchNorm2d(chn)
        self.g2_bn13 = nn.BatchNorm2d(chn)

    def forward(self, input_images): # Input[B, 3, H, W],输出[B, 1, H, W]
        net1 = self.g2_conv1(input_images)
        net1 = self.g2_bn1(net1)
        net1 = self.leakyrelu1(net1)

        net2 = self.g2_conv2(net1)
        net2 = self.g2_bn2(net2)
        net2 = self.leakyrelu2(net2)

        net3 = self.g2_conv3(net2)
        net3 = self.g2_bn3(net3)
        net3 = self.leakyrelu3(net3)

        net4 = self.g2_conv4(net3)
        net4 = self.g2_bn4(net4)
        net4 = self.leakyrelu4(net4)

        net5 = self.g2_conv5(net4)
        net5 = self.g2_bn5(net5)
        net5 = self.leakyrelu5(net5)

        net6 = self.g2_conv6(net5)
        net6 = self.g2_bn6(net6)
        net6 = self.leakyrelu6(net6)

        net7 = self.g2_conv7(net6)
        net7 = self.g2_bn7(net7)
        net7 = self.leakyrelu7(net7)

        net8 = self.g2_conv8(net7)
        net8 = self.g2_bn8(net8)
        net8 = self.leakyrelu8(net8)

        net9 = torch.cat([net6, net8], dim=1)

        net9 = self.g2_conv9(net9)
        net9 = self.g2_bn9(net9)
        net9 = self.leakyrelu9(net9)

        net10 = torch.cat([net5, net9], dim=1)

        net10 = self.g2_conv10(net10)
        net10 = self.g2_bn10(net10)
        net10 = self.leakyrelu10(net10)

        net11 = torch.cat([net4, net10], dim=1)

        net11 = self.g2_conv11(net11)
        net11 = self.g2_bn11(net11)
        net11 = self.leakyrelu11(net11)

        net12 = torch.cat([net3, net11], dim=1)

        net12 = self.g2_conv12(net12)
        net12 = self.g2_bn12(net12)
        net12 = self.leakyrelu12(net12)

        net13 = torch.cat([net2, net12], dim=1)

        net13 = self.g2_conv13(net13)
        net13 = self.g2_bn13(net13)
        net13 = self.leakyrelu13(net13)

        net14 = self.g2_conv14(net13)

        return net14
