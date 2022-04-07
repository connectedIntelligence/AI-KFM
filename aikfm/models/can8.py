import torch.nn as nn


# Source https://github.com/wanghuanphd/MDvsFA_cGAN
class CAN8(nn.Module):
    def __init__(self, in_channels:int = 3):
        super(CAN8, self).__init__()
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

        # Hout = (Hin + 2 * padding_0 - dilation_0*(kernel_size_0 - 1) -1)/stride_0 + 1
        # Wout = (Win + 2 * padding_1 - dilation_1*(kernel_size_1 - 1) -1)/stride_1 + 1
        self.g1_conv1 = nn.Conv2d(self.in_channels,     chn,   3, dilation=1, padding=1)
        self.g1_conv2 = nn.Conv2d(chn,   chn,   3, dilation=1, padding=1)
        self.g1_conv3 = nn.Conv2d(chn,   chn*2, 3, dilation=2, padding=2)
        self.g1_conv4 = nn.Conv2d(chn*2, chn*4, 3, dilation=4, padding=4)
        self.g1_conv5 = nn.Conv2d(chn*4, chn*8, 3, dilation=8, padding=8)
        self.g1_conv6 = nn.Conv2d(chn*8, chn*4, 3, dilation=4, padding=4)
        self.g1_conv7 = nn.Conv2d(chn*4, chn*2, 3, dilation=2, padding=2)
        self.g1_conv8 = nn.Conv2d(chn*2, chn,   3, dilation=1, padding=1)
        self.g1_conv9 = nn.Conv2d(chn,   1,     1, dilation=1)

        self.g1_bn1 = nn.BatchNorm2d(chn)
        self.g1_bn2 = nn.BatchNorm2d(chn)
        self.g1_bn3 = nn.BatchNorm2d(chn*2)
        self.g1_bn4 = nn.BatchNorm2d(chn*4)
        self.g1_bn5 = nn.BatchNorm2d(chn*8)
        self.g1_bn6 = nn.BatchNorm2d(chn*4)
        self.g1_bn7 = nn.BatchNorm2d(chn*2)
        self.g1_bn8 = nn.BatchNorm2d(chn)

    def forward(self, input_images): # Input[B, 3, H, W], Output[B, 1, H, W]
        net = self.g1_conv1(input_images)
        net = self.g1_bn1(net)
        net = self.leakyrelu1(net)

        net = self.g1_conv2(net)
        net = self.g1_bn2(net)
        net = self.leakyrelu2(net)

        net = self.g1_conv3(net)
        net = self.g1_bn3(net)
        net = self.leakyrelu3(net)

        net = self.g1_conv4(net)
        net = self.g1_bn4(net)
        net = self.leakyrelu4(net)

        net = self.g1_conv5(net)
        net = self.g1_bn5(net)
        net = self.leakyrelu5(net)

        net = self.g1_conv6(net)
        net = self.g1_bn6(net)
        net = self.leakyrelu6(net)

        net = self.g1_conv7(net)
        net = self.g1_bn7(net)
        net = self.leakyrelu7(net)

        net = self.g1_conv8(net)
        net = self.g1_bn8(net)
        net = self.leakyrelu8(net)

        output = self.g1_conv9(net)


        return output
