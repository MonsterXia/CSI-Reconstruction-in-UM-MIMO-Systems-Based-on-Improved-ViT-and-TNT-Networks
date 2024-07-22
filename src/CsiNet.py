# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/4/25 15:04
# @Function:

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.relu = nn.LeakyReLU()
        self.forward_net = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        shortcut = x
        x = self.forward_net(x)
        x += shortcut
        x = self.relu(x)
        return x


class CsiEncoder(nn.Module):
    def __init__(self, channels, height, width, encoded_dim):
        super(CsiEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU()
        self.channels = channels
        self.height = height
        self.width = width
        self.img_total = self.channels * self.height * self.width
        self.dense_encoded = nn.Linear(in_features=self.img_total, out_features=encoded_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.reshape(x, [-1, self.img_total])
        encoded = self.dense_encoded(x)
        return encoded


class CsiDecoder(nn.Module):
    def __init__(self, channels, height, width, encoded_dim):
        super(CsiDecoder, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.img_total = self.channels * self.height * self.width
        self.dense_decoded = nn.Linear(in_features=encoded_dim, out_features=self.img_total)
        self.residual_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(2)])

    def forward(self, x):

        x = self.dense_decoded(x)
        x = torch.reshape(x, [-1, self.channels, self.height, self.width])
        for block in self.residual_blocks:
            x = block(x)
        return x


class CsiNet(nn.Module):
    def __init__(self, channels, height, width, encoded_dim):
        super(CsiNet, self).__init__()
        self.encoder = CsiEncoder(channels, height, width, encoded_dim)
        self.decoder = CsiDecoder(channels, height, width, encoded_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
