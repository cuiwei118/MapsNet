import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size, padding=1, bias=False)
        self.conv3 = nn.Conv2d(1, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True, out=None)[0]

        x = torch.cat([avg_out, max_out], dim=1)
        conv1 = self.sigmoid(self.conv1(x))
        conv2 = self.sigmoid(self.conv2(conv1))
        conv3 = self.sigmoid(self.conv3(conv2))
        out = self.sigmoid(conv3)
        return out


class Cbam(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=3):
        super(Cbam, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out