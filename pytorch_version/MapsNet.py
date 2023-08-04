import os
import torch
import numpy as np
import torch.nn as nn
from CD_MapsNet.ASPP import ASPP_block
import torch.nn.functional as F
from CD_MapsNet.CBAM_Attention import Cbam
from torchvision.models import vgg16
from torchsummaryX import summary


class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base, self).__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in {3, 8, 15, 22, 29}:
                results.append(x)
        return results


class BasicBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_ch),
            nn.Dropout(0.5),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class MapsNet(nn.Module):
    def __init__(self):
        super(MapsNet, self).__init__()
        self.base = vgg16_base()

        self.double_conv1 = BasicBlock(768, 512, 512)
        self.double_conv2 = BasicBlock(768, 256, 256)
        self.double_conv3 = BasicBlock(384, 128, 128)
        self.double_conv4 = BasicBlock(192, 64, 64)

        self.conv1 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=1)
        self.conv6 = nn.Conv2d(5, 1, kernel_size=1)

        self.cbam_block_1 = Cbam(in_channels=64)
        self.cbam_block_2 = Cbam(in_channels=128)
        self.cbam_block_3 = Cbam(in_channels=256)
        self.cbam_block_4 = Cbam(in_channels=512)
        self.cbam_block_5 = Cbam(in_channels=512)

        self.pixel_shuffle_1 = nn.PixelShuffle(16)
        self.pixel_shuffle_2 = nn.PixelShuffle(8)
        self.pixel_shuffle_3 = nn.PixelShuffle(4)
        self.pixel_shuffle_4 = nn.PixelShuffle(2)

        # self.sigmoid = nn.Sigmoid()
        self.aspp = ASPP_block(512, 256)

        self.upsample_4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.upsample_3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.upsample_2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.upsample_1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

    def forward(self, t1_input, t2_input):
        t1_list = self.base(t1_input)
        t2_list = self.base(t2_input)
        # encoder
        t1_b1c3, t1_b2c3, t1_b3c3, t1_b4c3, t1_b5c3 = t1_list[0], t1_list[1], t1_list[2], t1_list[3], t1_list[4]
        t2_b1c3, t2_b2c3, t2_b3c3, t2_b4c3, t2_b5c3, = t2_list[0], t2_list[1], t2_list[2], t2_list[3], t2_list[4]
        # decoder
        t1_b5c3 = self.cbam_block_5(t1_b5c3)
        t2_b5c3 = self.cbam_block_5(t2_b5c3)
        concat5 = torch.subtract(t1_b5c3, t2_b5c3)

        t1_b4c3 = self.cbam_block_4(t1_b4c3)
        t2_b4c3 = self.cbam_block_4(t2_b4c3)
        concat4 = torch.subtract(t1_b4c3, t2_b4c3)

        t1_b3c3 = self.cbam_block_3(t1_b3c3)
        t2_b3c3 = self.cbam_block_3(t2_b3c3)
        concat3 = torch.subtract(t1_b3c3, t2_b3c3)

        t1_b2c3 = self.cbam_block_2(t1_b2c3)
        t2_b2c3 = self.cbam_block_2(t2_b2c3)
        concat2 = torch.subtract(t1_b2c3, t2_b2c3)

        t1_b1c3 = self.cbam_block_1(t1_b1c3)
        t2_b1c3 = self.cbam_block_1(t2_b1c3)
        concat1 = torch.subtract(t1_b1c3, t2_b1c3)

        up6 = self.aspp(concat5)
        conv6_1 = F.pixel_shuffle(up6, 16)
        conv6_2 = self.conv1(conv6_1)


        up7 = torch.cat((self.upsample_4(up6), concat4), dim=1)
        conv7_1 = self.double_conv1(up7)
        conv7_2 = F.pixel_shuffle(conv7_1, 8)
        conv7_3 = self.conv2(conv7_2)


        up8 = torch.cat((self.upsample_3(conv7_1), concat3), dim=1)
        conv8_1 = self.double_conv2(up8)
        conv8_2 = F.pixel_shuffle(conv8_1, 4)
        conv8_3 = self.conv3(conv8_2)

        up9 = torch.cat((self.upsample_2(conv8_1), concat2), dim=1)
        conv9_1 = self.double_conv3(up9)
        conv9_2 = F.pixel_shuffle(conv9_1, 2)
        conv9_3 = self.conv4(conv9_2)

        up10 = torch.cat((self.upsample_1(conv9_1), concat1), dim=1)
        conv10_1 = self.double_conv4(up10)
        conv10_2 = self.conv5(conv10_1)
        
        conv11 = torch.cat((conv10_2, conv6_2, conv7_3, conv8_3, conv9_3), dim=1)
        out = self.conv6(conv11)
        return out


import time

start = time.time()
from thop import profile
model = MapsNet()
dummy_input = torch.randn(1, 3, 512, 512)
dummy_input2 = torch.randn(1, 3, 512, 512)

flops, params = profile(model, (dummy_input, dummy_input2 ))

print(summary(model, dummy_input, dummy_input2))
end = time.time()
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
print('该模型计算时间：{}s'.format(end - start))

