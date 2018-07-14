import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
lua_unet = load_lua('../SCENENET_RESULTS_FOLDER_RERUN/NYUv2_TABLE/SCENENET_RGBD_EPOCH_10/converted_model.t7')
