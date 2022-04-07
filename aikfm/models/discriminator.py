from statistics import mode

import torch
import torch.nn as nn
import torch.nn.functional as F


# Source https://github.com/wanghuanphd/MDvsFA_cGAN
class discriminator(nn.Module):
    def __init__(self, img_channels:int = 3, seg_maks_channels:int = 1, init_size:int = 200):
        super(discriminator, self).__init__()
        self.in_channels = img_channels + seg_maks_channels
        self.init_size = init_size
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.Tanh1 = nn.Tanh()
        self.Tanh2 = nn.Tanh()
        self.Tanh3 = nn.Tanh()
        self.Softmax = nn.Softmax()


        self.d_conv1 = nn.Conv2d(self.in_channels,  24, 3, dilation=1, padding=1)
        self.d_conv2 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv3 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv4 = nn.Conv2d(24, 1,  3, dilation=1, padding=1)

        self.d_bn1 = nn.BatchNorm2d(24)
        self.d_bn2 = nn.BatchNorm2d(24)
        self.d_bn3 = nn.BatchNorm2d(24)
        self.d_bn4 = nn.BatchNorm2d(1)
        self.d_bn5 = nn.BatchNorm2d(256)
        self.d_bn6 = nn.BatchNorm2d(128)
        self.d_bn7 = nn.BatchNorm2d(64)
        self.d_bn8 = nn.BatchNorm2d(3)

        self.fc1 = nn.Linear(2500, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)


    def forward(self, input_images): # 输入[3B, 4, 1200, 1600],输出[B, 1, 1200, 1600]
        mini_batch_size = input_images.size(0)/3 # Batch size
        net = F.interpolate(input_images, size=(self.init_size, self.init_size), mode='bilinear')
        net = F.max_pool2d(input_images, kernel_size=[2, 2])  # [3B, 4, 100, 100]
        net = F.max_pool2d(net, kernel_size=[2, 2])  # [3B, 4, 50, 50]

        net = self.d_conv1(net)
        net = self.d_bn1(net)
        net = self.leakyrelu1(net)

        net = self.d_conv2(net)
        net = self.d_bn2(net)
        net = self.leakyrelu2(net)

        net = self.d_conv3(net)
        net = self.d_bn3(net)
        net = self.leakyrelu3(net)

        net = self.d_conv4(net)
        net = self.d_bn4(net)
        net1 = self.leakyrelu4(net) # [3B, 1, 50, 50]

        net = net1.view(-1, 2500) # [3B, 2500]
        net = self.fc1(net)      # [3B, 256]
        net = net.unsqueeze(2).unsqueeze(3) # [3B, 256, 1, 1]
        net = self.d_bn5(net) # [3B, 256, 1, 1]
        net = self.Tanh1(net)    # [3B, 256, 1, 1]

        net = net.view(-1, 256) # [3B, 256]
        net = self.fc2(net)      # [3B, 128]
        net = net.unsqueeze(2).unsqueeze(3) # [3B, 128, 1, 1]
        net = self.d_bn6(net) # [3B, 128, 1, 1]
        net = self.Tanh2(net)    # [3B, 128, 1, 1]

        net = net.view(-1, 128) # [3B, 128]
        net = self.fc3(net)      # [3B, 64]
        net = net.unsqueeze(2).unsqueeze(3) # [3B, 64, 1, 1]
        net = self.d_bn7(net) # [3B, 64, 1, 1]
        net = self.Tanh3(net) # [3B, 64, 1, 1]

        net = net.view(-1, 64) # [3B, 64]
        net = self.fc4(net)      # [3B, 3]
        net = net.unsqueeze(2).unsqueeze(3) # [3B, 3, 1, 1]
        net = self.d_bn8(net) # [3B, 3, 1, 1]
        net = self.Softmax(net) # [3B, 3, 1, 1]

        net = net.squeeze(3).squeeze(2) # [3B, 3]

        realscore0, realscore1, realscore2 = torch.split(net, mini_batch_size, dim=0)
        feat0, feat1, feat2 = torch.split(net1, mini_batch_size, dim=0)
        featDist = torch.mean(torch.pow(feat1 - feat2, 2))

        return realscore0, realscore1, realscore2, featDist
