import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
from PIL import Image
import numpy as np
import cv2
import os


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv3_64 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn3_64 = nn.BatchNorm2d(64, track_running_stats=True)
        self.bn3_64.training = False

        self.relu3_64 = nn.ReLU(inplace=True)

        self.conv64_64 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn64_64 = nn.BatchNorm2d(64, track_running_stats=True)
        self.bn64_64.training = False

        self.relu64_64 = nn.ReLU(inplace=True)
        self.pool_64 = nn.MaxPool2d(2, 2)

        self.conv64_128 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn64_128 = nn.BatchNorm2d(128, track_running_stats=True)
        self.bn64_128.training = False

        self.relu64_128 = nn.ReLU(inplace=True)

        self.conv128_128 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn128_128 = nn.BatchNorm2d(128, track_running_stats=True)
        self.bn128_128.training = False

        self.relu128_128 = nn.ReLU(inplace=True)
        self.pool_128 = nn.MaxPool2d(2, 2)

        self.conv128_256 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn128_256 = nn.BatchNorm2d(256, track_running_stats=True)
        self.bn128_256.training = False

        self.relu128_256 = nn.ReLU(inplace=True)

        self.conv256_256 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn256_256 = nn.BatchNorm2d(256, track_running_stats=True)
        self.bn256_256.training = False

        self.relu256_256 = nn.ReLU(inplace=True)
        self.pool_256 = nn.MaxPool2d(2, 2)

        self.conv256_512 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn256_512 = nn.BatchNorm2d(512, track_running_stats=True)
        self.bn256_512.training = False

        self.relu256_512 = nn.ReLU(inplace=True)

        self.conv512_512 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn512_512 = nn.BatchNorm2d(512, track_running_stats=True)
        self.bn512_512.training = False

        self.relu512_512 = nn.ReLU(inplace=True)
        self.pool_512 = nn.MaxPool2d(2, 2)

        self.conv512_1024 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv1024_512 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.up512 = nn.Upsample(scale_factor=2, mode='nearest')

        # self.concat1024 = torch.cat([self.up512, self.relu512_512], dim=1)
        #
        # self.conv1024_512_u = nn.Conv2d(1024, 512, 3, 1, 1)
        # self.bn1024_512_u = nn.BatchNorm2d(512, track_running_states=False)
        # self.relu1024_512_u = nn.ReLU(inplace=False)
        # self.conv1024_512_u = nn.Conv2d(512, 256, 3, 1, 1)
        # self.bn1024_512_u = nn.BatchNorm2d(256, track_running_states=False)
        # self.relu1024_256_u = nn.ReLU(inplace=True)
        # self.up256 = nn.Upsample(scale_factor=2, mode='nearest')
        #
        # self.concat512 = torch.cat([self.relu256_256, self.up256])
        #
        # self.conv512_256_u = nn.Conv2d(512, 256, 3, 1, 1)
        # self.bn512_256_u = nn.BatchNorm2d(256, track_running_states=False)
        # self.relu512_256_u = nn.ReLU(inplace=False)
        # self.conv512_256_u = nn.Conv2d(256, 128, 3, 1, 1)
        # self.bn512_512_u = nn.BatchNorm2d(128, track_running_states=False)
        # self.relu512_256_u = nn.ReLU(inplace=True)
        # self.up128 = nn.Upsample(scale_factor=2, mode='nearest')
        #
        # self.concat256 = torch.cat([self.relu128_128, self.up128])
        #
        # self.conv256_128_u = nn.Conv2d(256, 128, 3, 1, 1)
        # self.bn256_128_u = nn.BatchNorm2d(128, track_running_states=False)
        # self.relu256_128_u = nn.ReLU(inplace=False)
        # self.conv256_128_u = nn.Conv2d(128, 64, 3, 1, 1)
        # self.bn256_128_u = nn.BatchNorm2d(64, track_running_states=False)
        # self.relu256_128_u = nn.ReLU(inplace=True)
        # self.up64 = nn.Upsample(scale_factor=2, mode='nearest')
        #
        # self.concat128 = torch.cat([self.relu3_64, self.up64])
        #
        # self.conv256_128_u = nn.Conv2d(128, 64, 3, 1, 1)
        # self.bn256_128_u = nn.BatchNorm2d(64, track_running_states=False)
        # self.relu256_128_u = nn.ReLU(inplace=False)
        # self.conv256_128_u = nn.Conv2d(64, 64, 3, 1, 1)
        # self.bn256_128_u = nn.BatchNorm2d(64, track_running_states=False)
        # self.relu256_128_u = nn.ReLU(inplace=True)
        #
        # self.output = nn.Conv2d(64, 14, 3, 1, 1)


    def copy_bn_layer(self, pytorch_bn_layer, torch_bn_layer):

        pytorch_bn_layer.weight.data.copy_(torch_bn_layer.weight)
        pytorch_bn_layer.bias.data.copy_(torch_bn_layer.bias)
        pytorch_bn_layer.running_mean.data.copy_(torch_bn_layer.running_mean)
        pytorch_bn_layer.running_var.data.copy_(torch_bn_layer.running_var)

    def copy_conv_layer(self, pytorch_conv_layer, torch_conv_layer):

        pytorch_conv_layer.weight.data.copy_(torch_conv_layer.weight)
        pytorch_conv_layer.bias.data.copy_(torch_conv_layer.bias)

    def copy_weights(self, lua_model_t7):

        # SCENENET_RESULTS_FOLDER_RERUN/NYUv2_TABLE/SCENENET_RGB_EPOCH_15/converted_model.t7

        # block = unet:get(1)
        # save_conv_unet_block(block)
        #
        # block = unet:get(2):get(2):get(2)
        # save_conv_unet_block(block)
        #
        # block = unet:get(2):get(2):get(3):get(2):get(2)
        # save_conv_unet_block(block)
        #
        # block = unet:get(2):get(2):get(3):get(2):get(3):get(2):get(2)
        # save_conv_unet_block(block)
        #
        # block = unet:get(2):get(2):get(3):get(2):get(3):get(2):get(3):get(2)
        # save_conv(block:get(2))
        # save_conv(block:get(3))
        #
        # block = unet:get(2):get(2):get(3):get(2):get(3):get(2):get(5)
        # save_conv_unet_block(block)
        #
        # block = unet:get(2):get(2):get(3):get(2):get(5)
        # save_conv_unet_block(block)
        #
        # block = unet:get(2):get(2):get(5)
        # save_conv_unet_block(block)
        #
        # block = unet:get(4)
        # save_conv_unet_block(block)
        #
        # block = unet:get(5)
        # save_conv(block)

        self.lua_unet = load_lua(lua_model_t7)
        self.lua_unet.evaluate()

        self.first_block = self.lua_unet.get(0)

        self.copy_conv_layer(self.conv3_64, self.first_block.get(0))
        self.copy_bn_layer(self.bn3_64, self.first_block.get(1))
        self.copy_conv_layer(self.conv64_64, self.first_block.get(3))
        self.copy_bn_layer(self.bn64_64, self.first_block.get(4))

        self.second_block = self.lua_unet.get(1).get(1).get(1)

        self.copy_conv_layer(self.conv64_128, self.second_block.get(0))
        self.copy_bn_layer(self.bn64_128, self.second_block.get(1))
        self.copy_conv_layer(self.conv128_128, self.second_block.get(3))
        self.copy_bn_layer(self.bn128_128, self.second_block.get(4))

        self.third_block = self.lua_unet.get(1).get(1).get(2).get(1).get(1)

        self.copy_conv_layer(self.conv128_256, self.third_block.get(0))
        self.copy_bn_layer(self.bn128_256, self.third_block.get(1))
        self.copy_conv_layer(self.conv256_256, self.third_block.get(3))
        self.copy_bn_layer(self.bn256_256, self.third_block.get(4))

        # block = unet:get(2):get(2):get(3):get(2):get(3):get(2):get(2)
        # save_conv_unet_block(block)

        self.fourth_block = self.lua_unet.get(1).get(1).get(2).get(1).get(2).get(1).get(1)

        self.copy_conv_layer(self.conv256_512, self.fourth_block.get(0))
        self.copy_bn_layer(self.bn256_512, self.fourth_block.get(1))
        self.copy_conv_layer(self.conv512_512, self.fourth_block.get(3))
        self.copy_bn_layer(self.bn512_512, self.fourth_block.get(4))




        print('Have copied the weights of the first block')

    def run_torch_pytorch_import_test(self):

        # Batch should be in NCHW format
        input = np.zeros((1, 3, 128, 128))
        myImg = torch.tensor(input, dtype=torch.float32)

        yTorch = self.first_block.forward(myImg)
        yTorch = self.pool_64(yTorch)
        yTorch = self.second_block.forward(yTorch)

        ypyTorch = self.forward(myImg)

        print('DIFF {}'.format(np.sum(yTorch.detach().numpy() - ypyTorch.detach().numpy())))

    def forward(self, x):

        out = self.conv3_64(x)
        out = self.bn3_64(out)
        out = self.relu3_64(out)
        out = self.conv64_64(out)
        out = self.bn64_64(out)
        out = self.relu64_64(out)

        out = self.pool_64(out)

        out = self.conv64_128(out)
        out = self.bn64_128(out)
        out = self.relu64_128(out)
        out = self.conv128_128(out)
        out = self.bn128_128(out)
        out = self.relu128_128(out)

        out = self.pool_128(out)

        out = self.conv128_256(out)
        out = self.bn128_256(out)
        out = self.relu128_256(out)
        out = self.conv256_256(out)
        out = self.bn256_256(out)
        out = self.relu256_256(out)

        out = self.pool_256(out)

        out = self.conv256_512(out)
        out = self.bn256_512(out)
        out = self.relu256_512(out)
        out = self.conv512_512(out)
        out = self.bn512_512(out)
        out = self.relu256_512(out)
        

        return out