import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
from PIL import Image
import numpy as np
import cv2
import os


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.conv3_64 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn3_64 = nn.BatchNorm2d(64, track_running_stats=False)
        self.relu3_64 = nn.ReLU(inplace=True)
        self.conv64_64 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn64_64 = nn.BatchNorm2d(64, track_running_stats=False)
        self.relu3_64 = nn.ReLU(inplace=True)

        self.pool_64 = nn.MaxPool2d(2, 2)

        self.conv64_128 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn64_128 = nn.BatchNorm2d(128, track_running_stats=False)
        self.relu64_128 = nn.ReLU(inplace=True)
        self.conv128_128 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn128_128 = nn.BatchNorm2d(128, track_running_stats=False)
        self.relu128_128 = nn.ReLU(inplace=True)

        self.pool_128 = nn.MaxPool2d(2, 2)

        self.conv128_256 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn128_256 = nn.BatchNorm2d(256, track_running_stats=False)
        self.relu128_256 = nn.ReLU(inplace=True)
        self.conv256_256 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn256_256 = nn.BatchNorm2d(256, track_running_stats=False)
        self.relu256_256 = nn.ReLU(inplace=True)

        self.pool_256 = nn.MaxPool2d(2, 2)

        self.conv256_512 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn256_512 = nn.BatchNorm2d(512, track_running_stats=False)
        self.relu256_512 = nn.ReLU(inplace=True)
        self.conv512_512 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn512_512 = nn.BatchNorm2d(512, track_running_stats=False)
        self.relu512_512 = nn.ReLU(inplace=True)

        self.pool_512 = nn.MaxPool2d(2, 2)
        self.conv512_1024 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv1024_512 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.up512 = nn.Upsample(scale_factor=2, mode='nearest')

        self.concat1024 = torch.cat([self.up512, self.relu512_512], dim=1)

        self.conv1024_512_u = nn.Conv2d(1024, 512, 3, 1, 1)
        self.bn1024_512_u = nn.BatchNorm2d(512, track_running_states=False)
        self.relu1024_512_u = nn.ReLU(inplace=False)
        self.conv1024_512_u = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn1024_512_u = nn.BatchNorm2d(256, track_running_states=False)
        self.relu1024_256_u = nn.ReLU(inplace=True)
        self.up256 = nn.Upsample(scale_factor=2, mode='nearest')

        self.concat512 = torch.cat([self.relu256_256, self.up256])

        self.conv512_256_u = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn512_256_u = nn.BatchNorm2d(256, track_running_states=False)
        self.relu512_256_u = nn.ReLU(inplace=False)
        self.conv512_256_u = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn512_512_u = nn.BatchNorm2d(128, track_running_states=False)
        self.relu512_256_u = nn.ReLU(inplace=True)
        self.up128 = nn.Upsample(scale_factor=2, mode='nearest')

        self.concat256 = torch.cat([self.relu128_128, self.up128])

        self.conv256_128_u = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn256_128_u = nn.BatchNorm2d(128, track_running_states=False)
        self.relu256_128_u = nn.ReLU(inplace=False)
        self.conv256_128_u = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn256_128_u = nn.BatchNorm2d(64, track_running_states=False)
        self.relu256_128_u = nn.ReLU(inplace=True)
        self.up64 = nn.Upsample(scale_factor=2, mode='nearest')

        self.concat128 = torch.cat([self.relu3_64, self.up64])

        self.conv256_128_u = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn256_128_u = nn.BatchNorm2d(64, track_running_states=False)
        self.relu256_128_u = nn.ReLU(inplace=False)
        self.conv256_128_u = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn256_128_u = nn.BatchNorm2d(64, track_running_states=False)
        self.relu256_128_u = nn.ReLU(inplace=True)

        self.output = nn.Conv2d(64, 14, 3, 1, 1)

    def copy_weights(self, lua_model_t7):

        lua_unet = lua_load(lua_model_t7)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x