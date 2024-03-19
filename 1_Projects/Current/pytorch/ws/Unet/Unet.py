import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from torchvision import transforms


class conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, strides=1, padding=0):
        super(conv_2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class upconv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, strides=2):
        super(upconv_2d, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.upconv1(x)))
        return out


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.layer1_conv = conv_2d(1, 64)
        self.layer2_conv = conv_2d(64, 128)
        self.layer3_conv = conv_2d(128, 256)
        self.layer4_conv = conv_2d(256, 512)
        self.layer5_conv = conv_2d(512, 1024)
        self.layer6_conv = conv_2d(1024, 512)
        self.layer7_conv = conv_2d(512, 256)
        self.layer8_conv = conv_2d(256, 128)
        self.layer9_conv = conv_2d(128, 64)
        self.layer10_conv = nn.Conv2d(64, 2, kernel_size=1,
                                      stride=1, padding=0, bias=True)

        self.upconv1 = upconv_2d(1024, 512)
        self.upconv2 = upconv_2d(512, 256)
        self.upconv3 = upconv_2d(256, 128)
        self.upconv4 = upconv_2d(128, 64)

    def forward(self, x):

        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)
        convt1 = self.upconv1(conv5)
        concat1 = torch.cat([conv4, convt1], 1)

        conv6 = self.layer6_conv(concat1)
        convt2 = self.upconv2(conv6)
        concat2 = torch.cat([conv3, convt2], 1)

        conv7 = self.layer7_conv(concat2)
        convt3 = self.upconv3(conv7)
        concat3 = torch.cat([conv2, convt3], 1)

        conv8 = self.layer8_conv(concat3)
        convt4 = self.upconv4(conv8)
        concat4 = torch.cat([conv1, convt4], 1)

        conv9 = self.layer9_conv(concat4)
        out = self.layer10_conv(conv9)
        out = nn.Sigmoid(out)

        return out
