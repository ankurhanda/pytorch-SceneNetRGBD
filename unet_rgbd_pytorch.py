import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
#lua_unet = load_lua('../SCENENET_RESULTS_FOLDER_RERUN/NYUv2_TABLE/SCENENET_RGBD_EPOCH_10/converted_model.t7')

import numpy as np


class UNetRGBD(nn.Module):
    def __init__(self):
        super(UNetRGBD, self).__init__()

        ''' RGB '''

        '''first block'''
        self.conv_rgb_3_32 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn_rgb_3_32 = nn.BatchNorm2d(32, track_running_stats=True)
        self.bn_rgb_3_32.training = False

        self.relu_rgb_3_32 = nn.ReLU(inplace=True)

        self.conv_rgb_32_32 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn_rgb_32_32 = nn.BatchNorm2d(32, track_running_stats=True)
        self.bn_rgb_32_32.training = False

        self.relu_rgb_32_32 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)


        ''' second block '''
        self.conv_rgb_32_64 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn_rgb_32_64 = nn.BatchNorm2d(64, track_running_stats=True)
        self.bn_rgb_32_64.training = False

        self.relu_rgb_32_64 = nn.ReLU(inplace=True)

        self.conv_rgb_64_64 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn_rgb_64_64 = nn.BatchNorm2d(64, track_running_stats=True)
        self.bn_rgb_64_64.training = False

        self.relu_rgb_64_64 = nn.ReLU(inplace=True)


        ''' third block '''
        self.conv_rgb_64_128 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn_rgb_64_128 = nn.BatchNorm2d(128, track_running_stats=True)
        self.bn_rgb_64_128.training = False

        self.relu_rgb_64_128 = nn.ReLU(inplace=True)

        self.conv_rgb_128_128 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn_rgb_128_128 = nn.BatchNorm2d(128, track_running_stats=True)
        self.bn_rgb_128_128.training = False

        self.relu_rgb_128_128 = nn.ReLU(inplace=True)



        ''' fourth block '''
        self.conv_rgb_128_256 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn_rgb_128_256 = nn.BatchNorm2d(256, track_running_stats=True)
        self.bn_rgb_128_256.training = False

        self.relu_rgb_128_256 = nn.ReLU(inplace=True)

        self.conv_rgb_256_256 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn_rgb_256_256 = nn.BatchNorm2d(256, track_running_stats=True)
        self.bn_rgb_256_256.training = False

        self.relu_rgb_256_256 = nn.ReLU(inplace=True)


        ''' D '''

        '''first block'''

        self.conv_d_3_32 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn_d_3_32 = nn.BatchNorm2d(32, track_running_stats=True)
        self.bn_d_3_32.training = False

        self.relu_d_3_32 = nn.ReLU(inplace=True)

        self.conv_d_32_32 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn_d_32_32 = nn.BatchNorm2d(32, track_running_stats=True)
        self.bn_d_32_32.training = False

        self.relu_d_32_32 = nn.ReLU(inplace=True)


        ''' second block '''
        self.conv_d_32_64 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn_d_32_64 = nn.BatchNorm2d(64, track_running_stats=True)
        self.bn_d_32_64.training = False

        self.relu_d_32_64 = nn.ReLU(inplace=True)

        self.conv_d_64_64 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn_d_64_64 = nn.BatchNorm2d(64, track_running_stats=True)
        self.bn_d_64_64.training = False

        self.relu_d_64_64 = nn.ReLU(inplace=True)


        ''' third block '''

        self.conv_d_64_128 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn_d_64_128 = nn.BatchNorm2d(128, track_running_stats=True)
        self.bn_d_64_128.training = False

        self.relu_d_64_128 = nn.ReLU(inplace=True)

        self.conv_d_128_128 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn_d_128_128 = nn.BatchNorm2d(128, track_running_stats=True)
        self.bn_d_128_128.training = False

        self.relu_d_128_128 = nn.ReLU(inplace=True)

        ''' fourth block '''

        self.conv_d_128_256 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn_d_128_256 = nn.BatchNorm2d(256, track_running_stats=True)
        self.bn_d_128_256.training = False

        self.relu_d_128_256 = nn.ReLU(inplace=True)

        self.conv_d_256_256 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn_d_256_256 = nn.BatchNorm2d(256, track_running_stats=True)
        self.bn_d_256_256.training = False

        self.relu_d_256_256 = nn.ReLU(inplace=True)


        self.conv512_1024 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv1024_512 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')


        '''
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

        self.conv1024_512_u = nn.Conv2d(1024, 512, 3, 1, 1)
        self.bn1024_512_u = nn.BatchNorm2d(512, track_running_stats=True)
        self.bn1024_512_u.training = False

        self.relu1024_512_u = nn.ReLU(inplace=False)
        self.conv512_256_u = nn.Conv2d(512, 256, 3, 1, 1)

        self.bn512_256_u = nn.BatchNorm2d(256, track_running_stats=True)
        self.bn512_256_u.training = False

        self.relu512_256_u = nn.ReLU(inplace=True)
        self.up256 = nn.Upsample(scale_factor=2, mode='nearest')

        # self.concat512 = torch.cat([self.relu256_256, self.up256])

        self.conv512_256_catu = nn.Conv2d(512, 256, 3, 1, 1)
        self.bn512_256_catu = nn.BatchNorm2d(256, track_running_stats=True)
        self.bn512_256_catu.training = False

        self.relu512_256_catu = nn.ReLU(inplace=True)
        self.conv256_128_catu = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn256_128_catu = nn.BatchNorm2d(128, track_running_stats=True)
        self.bn256_128_catu.training = False

        self.relu256_128_catu = nn.ReLU(inplace=True)
        self.up128 = nn.Upsample(scale_factor=2, mode='nearest')




        self.conv256_128_u = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn256_128_u = nn.BatchNorm2d(128, track_running_stats=True)
        self.bn256_128_u.training = False

        self.relu256_128_u = nn.ReLU(inplace=True)
        self.conv128_64_u = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn128_64_u = nn.BatchNorm2d(64, track_running_stats=True)
        self.bn128_64_u.training = False

        self.relu128_64_u = nn.ReLU(inplace=True)
        self.up64 = nn.Upsample(scale_factor=2, mode='nearest')


        self.conv128_64_ucat = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn128_64_ucat = nn.BatchNorm2d(64, track_running_stats=True)
        self.bn128_64_ucat.training = False

        self.relu128_64_ucat = nn.ReLU(inplace=False)
        self.conv64_64_ucat = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn64_64_ucat = nn.BatchNorm2d(64, track_running_stats=True)
        self.bn64_64_ucat.training = False

        self.relu64_64_u = nn.ReLU(inplace=True)

        self.conv_out_64 = nn.Conv2d(64, 14, 1, 1, 0)
        
        '''


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

        self.lua_unet = load_lua(lua_model_t7)
        self.lua_unet.evaluate()

        self.first_block = self.lua_unet.get(0)

        self.first_rgb_block = self.first_block.get(0)
        self.first_d_block   = self.first_block.get(1)

        self.copy_conv_layer(self.conv_rgb_3_32, self.first_rgb_block.get(0))
        self.copy_bn_layer(self.bn_rgb_3_32, self.first_rgb_block.get(1))
        self.copy_conv_layer(self.conv_rgb_32_32, self.first_rgb_block.get(3))
        self.copy_bn_layer(self.bn_rgb_32_32, self.first_rgb_block.get(4))

        self.copy_conv_layer(self.conv_d_3_32, self.first_d_block.get(0))
        self.copy_bn_layer(self.bn_d_3_32, self.first_d_block.get(1))
        self.copy_conv_layer(self.conv_d_32_32, self.first_d_block.get(3))
        self.copy_bn_layer(self.bn_d_32_32, self.first_d_block.get(4))





        self.second_block = self.lua_unet.get(1).get(1).get(1)

        self.second_rgb_block = self.second_block.get(0)
        self.second_d_block = self.second_block.get(1)

        self.copy_conv_layer(self.conv_rgb_32_64, self.second_rgb_block.get(0))
        self.copy_bn_layer(self.bn_rgb_32_64, self.second_rgb_block.get(1))
        self.copy_conv_layer(self.conv_rgb_64_64, self.second_rgb_block.get(3))
        self.copy_bn_layer(self.bn_rgb_64_64, self.second_rgb_block.get(4))

        self.copy_conv_layer(self.conv_d_32_64, self.second_d_block.get(0))
        self.copy_bn_layer(self.bn_d_32_64, self.second_d_block.get(1))
        self.copy_conv_layer(self.conv_d_64_64, self.second_d_block.get(3))
        self.copy_bn_layer(self.bn_d_64_64, self.second_d_block.get(4))








        self.third_block = self.lua_unet.get(1).get(1).get(2).get(1).get(1)

        self.third_rgb_block = self.third_block.get(0)
        self.third_d_block = self.third_block.get(1)

        self.copy_conv_layer(self.conv_rgb_64_128, self.third_rgb_block.get(0))
        self.copy_bn_layer(self.bn_rgb_64_128, self.third_rgb_block.get(1))
        self.copy_conv_layer(self.conv_rgb_128_128, self.third_rgb_block.get(3))
        self.copy_bn_layer(self.bn_rgb_128_128, self.third_rgb_block.get(4))

        self.copy_conv_layer(self.conv_d_64_128, self.third_d_block.get(0))
        self.copy_bn_layer(self.bn_d_64_128, self.third_d_block.get(1))
        self.copy_conv_layer(self.conv_d_128_128, self.third_d_block.get(3))
        self.copy_bn_layer(self.bn_d_128_128, self.third_d_block.get(4))







        self.fourth_block = self.lua_unet.get(1).get(1).get(2).get(1).get(2).get(1).get(1)

        self.fourth_rgb_block = self.fourth_block.get(0)
        self.fourth_d_block = self.fourth_block.get(1)

        self.copy_conv_layer(self.conv_rgb_128_256, self.fourth_rgb_block.get(0))
        self.copy_bn_layer(self.bn_rgb_128_256, self.fourth_rgb_block.get(1))
        self.copy_conv_layer(self.conv_rgb_256_256, self.fourth_rgb_block.get(3))
        self.copy_bn_layer(self.bn_rgb_256_256, self.fourth_rgb_block.get(4))

        self.copy_conv_layer(self.conv_d_128_256, self.fourth_d_block.get(0))
        self.copy_bn_layer(self.bn_d_128_256, self.fourth_d_block.get(1))
        self.copy_conv_layer(self.conv_d_256_256, self.fourth_d_block.get(3))
        self.copy_bn_layer(self.bn_d_256_256, self.fourth_d_block.get(4))


        self.fifth_block = self.lua_unet.get(1).get(1).get(2).get(1).get(2).get(1).get(2)
        self.copy_conv_layer(self.conv512_1024, self.fifth_d_block.get(2))
        self.copy_conv_layer(self.conv1024_512, self.fifth_d_block.get(3))


        self.make_everything_identity()


        print(self.lua_unet)

        # self.sixth_block = self.lua_unet.get(1).get(1).get(2).get(1).get(2).get(1).get(3)
        #
        #
        # self.seventh_block = self.lua_unet.get(1).get(1).get(2).get(1).get(5)
        #
        #
        # self.eigth_block = self.lua_unet.get(1).get(1).get(5)
        #
        #
        # self.ninth_block = self.lua_unet.get(4)
        #
        #
        # self.tenth_block = self.lua_unet.get(5)


        '''
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

        self.fifth_block = self.lua_unet.get(1).get(1).get(2).get(1).get(2).get(1).get(2).get(1)
        self.copy_conv_layer(self.conv512_1024, self.fifth_block.get(1))
        self.copy_conv_layer(self.conv1024_512, self.fifth_block.get(2))

        self.sixth_block = self.lua_unet.get(1).get(1).get(2).get(1).get(2).get(1).get(4)
        self.copy_conv_layer(self.conv1024_512_u, self.sixth_block.get(0))
        self.copy_bn_layer(self.bn1024_512_u, self.sixth_block.get(1))
        self.copy_conv_layer(self.conv512_256_u, self.sixth_block.get(3))
        self.copy_bn_layer(self.bn512_256_u, self.sixth_block.get(4))

        self.seventh_block = self.lua_unet.get(1).get(1).get(2).get(1).get(4)
        print(self.seventh_block)
        self.copy_conv_layer(self.conv512_256_catu, self.seventh_block.get(0))
        self.copy_bn_layer(self.bn512_256_catu, self.seventh_block.get(1))
        self.copy_conv_layer(self.conv256_128_catu, self.seventh_block.get(3))
        self.copy_bn_layer(self.bn256_128_catu, self.seventh_block.get(4))


        # block = unet:get(2):get(2):get(5)
        # save_conv_unet_block(block)

        self.eigth_block = self.lua_unet.get(1).get(1).get(4)
        self.copy_conv_layer(self.conv256_128_u, self.eigth_block.get(0))
        self.copy_bn_layer(self.bn256_128_u, self.eigth_block.get(1))
        self.copy_conv_layer(self.conv128_64_u, self.eigth_block.get(3))
        self.copy_bn_layer(self.bn128_64_u, self.eigth_block.get(4))

        # block = unet:get(4)
        # save_conv_unet_block(block)

        self.ninth_block = self.lua_unet.get(3)
        self.copy_conv_layer(self.conv128_64_ucat, self.ninth_block.get(0))
        self.copy_bn_layer(self.bn128_64_ucat, self.ninth_block.get(1))
        self.copy_conv_layer(self.conv64_64_ucat, self.ninth_block.get(3))
        self.copy_bn_layer(self.bn64_64_ucat, self.ninth_block.get(4))

        # block = unet:get(5)
        # save_conv(block)

        self.tenth_block = self.lua_unet.get(4)
        self.copy_conv_layer(self.conv_out_64, self.tenth_block)
        '''


        print('Have copied the weights of the first block')


    def make_everything_identity(self):

        self.lua_unet.modules[1].modules[1].modules[2].modules[1].modules[2].modules[1].modules[3] = torch.legacy.nn.Identity()
        self.lua_unet.modules[1].modules[1].modules[2].modules[1].modules[2].modules[1].modules[4] = torch.legacy.nn.Identity()

        self.lua_unet.modules[1].modules[1].modules[2].modules[1].modules[5] = torch.legacy.nn.Identity()
        self.lua_unet.modules[1].modules[1].modules[2].modules[1].modules[3] = torch.legacy.nn.Identity()
        self.lua_unet.modules[1].modules[1].modules[2].modules[1].modules[4] = torch.legacy.nn.Identity()
        self.lua_unet.modules[1].modules[1].modules[2].modules[1].modules[6] = torch.legacy.nn.Identity()

        self.lua_unet.modules[1].modules[1].modules[5] = torch.legacy.nn.Identity()
        self.lua_unet.modules[1].modules[1].modules[3] = torch.legacy.nn.Identity()
        self.lua_unet.modules[1].modules[1].modules[4] = torch.legacy.nn.Identity()
        self.lua_unet.modules[1].modules[1].modules[6] = torch.legacy.nn.Identity()

        self.lua_unet.modules[2] = torch.legacy.nn.Identity()
        self.lua_unet.modules[3] = torch.legacy.nn.Identity()
        self.lua_unet.modules[4] = torch.legacy.nn.Identity()
        self.lua_unet.modules[5] = torch.legacy.nn.Identity()

    def run_torch_pytorch_import_test(self):

        # Batch should be in NCHW format
        # input = np.ones((1, 3, 128, 128))
        input_rgb = np.random.rand(1, 3, 128, 128)
        myrgbImg = torch.tensor(input_rgb, dtype=torch.float32)

        input_d = np.random.rand(1, 1, 128, 128)
        mydImg = torch.tensor(input_d, dtype=torch.float32)

        # yTorch = self.lua_unet.forward((myrgbImg, mydImg))

        # yTorch_rgb32, yTorch_d32 = self.first_block.forward((myrgbImg, mydImg))
        # yTorch_rgb32 = self.pool(yTorch_rgb32)
        # yTorch_d32   = self.pool(yTorch_d32)
        #
        # yTorch_rgb64, yTorch_d64 = self.second_block.forward((yTorch_rgb32, yTorch_d32))
        # yTorch_rgb64 = self.pool(yTorch_rgb64)
        # yTorch_d64 = self.pool(yTorch_d64)
        #
        # yTorch_rgb128, yTorch_d128 = self.third_block.forward((yTorch_rgb64, yTorch_d64))
        # yTorch_rgb128 = self.pool(yTorch_rgb128)
        # yTorch_d128 = self.pool(yTorch_d128)
        #
        # yTorch_rgb256, yTorch_d256 = self.fourth_block.forward((yTorch_rgb128, yTorch_d128))
        # yTorch_rgb256 = self.pool(yTorch_rgb256)
        # yTorch_d256 = self.pool(yTorch_d256)


        yTorch_out = self.lua_unet.forward((myrgbImg, mydImg))


        # yTorch = self.pool_64(yTorch64)
        # yTorch128 = self.second_block.forward(yTorch)
        # yTorch = self.pool_128(yTorch128)
        # yTorch256 = self.third_block.forward(yTorch)
        # yTorch = self.pool_256(yTorch256)
        # yTorch512 = self.fourth_block.forward(yTorch)
        # yTorch = self.fifth_block.forward(yTorch512)
        #
        # yTorch = torch.cat([yTorch512, yTorch], dim=1)
        # yTorch = self.sixth_block.forward(yTorch)
        # yTorch = self.up512(yTorch)
        #
        # yTorch = torch.cat([yTorch256, yTorch], dim=1)
        # yTorch = self.seventh_block.forward(yTorch)
        # yTorch = self.up256(yTorch)
        #
        # yTorch = torch.cat([yTorch128, yTorch], dim=1)
        # yTorch = self.eigth_block.forward(yTorch)
        # yTorch = self.up128(yTorch)
        #
        # yTorch = torch.cat([yTorch64, yTorch], dim=1)
        # yTorch = self.ninth_block.forward(yTorch)
        #
        # yTorch = self.tenth_block.forward(yTorch)

        print('yTorch shape = ', yTorch_out.detach().numpy().shape)

        # ypyTorch_rgb, ypyTorch_d = self.forward((myrgbImg, mydImg))
        ypyTorch_out = self.forward((myrgbImg, mydImg))

        # print('ypTorch_rgb shape = ', ypyTorch_rgb.detach().numpy().shape)
        # print('ypTorch_depth shape = ', ypyTorch_d.detach().numpy().shape)

        print('DIFF {}'.format(np.sum(yTorch_out.detach().numpy() - ypyTorch_out.detach().numpy())))
        # print('DIFF {}'.format(np.sum(yTorch_d256.detach().numpy() - ypyTorch_d.detach().numpy())))

    def forward(self, x):

        ''' RGB '''

        ''' first block '''

        out_rgb = self.conv_rgb_3_32(x[0])
        out_rgb = self.bn_rgb_3_32(out_rgb)
        out_rgb = self.relu_rgb_3_32(out_rgb)
        out_rgb = self.conv_rgb_32_32(out_rgb)
        out_rgb = self.bn_rgb_32_32(out_rgb)
        out_rgb_relu32 = self.relu_rgb_32_32(out_rgb)

        out_rgb = self.pool(out_rgb_relu32)

        ''' second block '''

        out_rgb = self.conv_rgb_32_64(out_rgb)
        out_rgb = self.bn_rgb_32_64(out_rgb)
        out_rgb = self.relu_rgb_32_64(out_rgb)
        out_rgb = self.conv_rgb_64_64(out_rgb)
        out_rgb = self.bn_rgb_64_64(out_rgb)
        out_rgb_relu64 = self.relu_rgb_64_64(out_rgb)

        out_rgb = self.pool(out_rgb_relu64)

        ''' third block '''

        out_rgb = self.conv_rgb_64_128(out_rgb)
        out_rgb = self.bn_rgb_64_128(out_rgb)
        out_rgb = self.relu_rgb_64_128(out_rgb)
        out_rgb = self.conv_rgb_128_128(out_rgb)
        out_rgb = self.bn_rgb_128_128(out_rgb)
        out_rgb_relu128 = self.relu_rgb_128_128(out_rgb)

        out_rgb = self.pool(out_rgb_relu128)


        ''' fourth block '''

        out_rgb = self.conv_rgb_128_256(out_rgb)
        out_rgb = self.bn_rgb_128_256(out_rgb)
        out_rgb = self.relu_rgb_128_256(out_rgb)
        out_rgb = self.conv_rgb_256_256(out_rgb)
        out_rgb = self.bn_rgb_256_256(out_rgb)
        out_rgb_relu256 = self.relu_rgb_256_256(out_rgb)

        out_rgb = out_rgb_relu256



        ''' D '''

        '''first block '''

        out_d = self.conv_d_3_32(x[1])
        out_d = self.bn_d_3_32(out_d)
        out_d = self.relu_d_3_32(out_d)
        out_d = self.conv_d_32_32(out_d)
        out_d = self.bn_d_32_32(out_d)
        out_d_relu32 = self.relu_d_32_32(out_d)

        out_d = self.pool(out_d_relu32)

        ''' second block '''

        out_d = self.conv_d_32_64(out_d)
        out_d = self.bn_d_32_64(out_d)
        out_d = self.relu_d_32_64(out_d)
        out_d = self.conv_d_64_64(out_d)
        out_d = self.bn_d_64_64(out_d)
        out_d_relu64 = self.relu_d_64_64(out_d)

        out_d = self.pool(out_d_relu64)

        ''' third block '''

        out_d = self.conv_d_64_128(out_d)
        out_d = self.bn_d_64_128(out_d)
        out_d = self.relu_d_64_128(out_d)
        out_d = self.conv_d_128_128(out_d)
        out_d = self.bn_d_128_128(out_d)
        out_d_relu128 = self.relu_d_128_128(out_d)

        out_d = self.pool(out_d_relu128)


        ''' fourth '''

        out_d = self.conv_d_128_256(out_d)
        out_d = self.bn_d_128_256(out_d)
        out_d = self.relu_d_128_256(out_d)
        out_d = self.conv_d_256_256(out_d)
        out_d = self.bn_d_256_256(out_d)
        out_d_relu256 = self.relu_d_256_256(out_d)

        out_d = out_d_relu256



        out = torch.cat([out_rgb, out_d], dim=1)

        out = self.pool(out)
        out = self.conv512_1024(out)
        out = self.conv1024_512(out)
        out = self.up(out)

        # out = self.conv_512_512(out)
        # out = self.bn_512_512(out)
        # out = self.relu_512_512(out)
        #
        # out = self.conv_512_256(out)
        # out = self.bn_512_256(out)
        # out = self.relu_512_256(out)
        #
        # out = self.up256(out)



        '''
        out = self.conv64_128(out)
        out = self.bn64_128(out)
        out = self.relu64_128(out)
        out = self.conv128_128(out)
        out = self.bn128_128(out)
        out_relu128 = self.relu128_128(out)

        out = self.pool_128(out_relu128)

        out = self.conv128_256(out)
        out = self.bn128_256(out)
        out = self.relu128_256(out)
        out = self.conv256_256(out)
        out = self.bn256_256(out)
        out_relu256 = self.relu256_256(out)

        out = self.pool_256(out_relu256)

        out = self.conv256_512(out)
        out = self.bn256_512(out)
        out = self.relu256_512(out)
        out = self.conv512_512(out)
        out = self.bn512_512(out)
        out_relu512 = self.relu512_512(out)

        out = self.pool_512(out_relu512)
        out = self.conv512_1024(out)
        out = self.conv1024_512(out)
        out = self.up512(out)

        out = torch.cat([out_relu512, out], dim=1)
        out = self.conv1024_512_u(out)
        out = self.bn1024_512_u(out)
        out = self.relu1024_512_u(out)
        out = self.conv512_256_u(out)
        out = self.bn512_256_u(out)
        out = self.relu512_256_u(out)
        out = self.up256(out)

        out = torch.cat([out_relu256, out], dim=1)
        out = self.conv512_256_catu(out)
        out = self.bn512_256_catu(out)
        out = self.relu512_256_catu(out)
        out = self.conv256_128_catu(out)
        out = self.bn256_128_catu(out)
        out = self.relu256_128_catu(out)
        out = self.up128(out)

        out = torch.cat([out_relu128, out], dim=1)
        out = self.conv256_128_u(out)
        out = self.bn256_128_u(out)
        out = self.relu256_128_u(out)
        out = self.conv128_64_u(out)
        out = self.bn128_64_u(out)
        out = self.relu128_64_u(out)
        out = self.up64(out)

        out = torch.cat([out_relu64, out], dim=1)
        out = self.conv128_64_ucat(out)
        out = self.bn128_64_ucat(out)
        out = self.relu128_64_ucat(out)
        out = self.conv64_64_ucat(out)
        out = self.bn64_64_ucat(out)
        out = self.relu64_64_u(out)

        out = self.conv_out_64(out)
        '''

        return out #(out_rgb, out_d)