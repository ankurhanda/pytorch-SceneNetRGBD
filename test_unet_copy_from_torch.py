from unet_pytorch import UNet

pytorch_unet = UNet()

pytorch_unet.copy_weights(lua_model_t7='../SCENENET_RESULTS_FOLDER_RERUN/NYUv2_TABLE/SCENENET_RGB_EPOCH_15/converted_model2.t7')
