import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from lut_layer_main import *
from binarization import *
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import warnings
import resnet
import json

def save_model_info_to_json(model, model_name, input_tensor, file_path):
    """Save model information to a JSON file in execution order."""
    execution_order = []

    def lut_kernel_w_preprocess(weights):
        # weight_size = out_num, self.lut_num, 2**lut_k
        weight_size = weights.size()
        out_num = weight_size[0]
        lut_num = weight_size[1]
        lut_k = weight_size[2]
        hex_list = []
        weights_np = (weights>=0).float().cpu().numpy()
        for i in range(out_num):
            for j in range(lut_num):
                bits = weights_np[i, j, ::-1]
                bit_str = ''.join(str(int(b)) for b in bits)
                hex_str = hex(int(bit_str, 2))[2:].zfill(16)
                hex_list.append(hex_str)
        return hex_list

    def quant_w_preprocess(layer):
        lut_kernels = layer.lut_quant_fc.lut_kernels
        lut_weights = []
        for lut_kernel in lut_kernels:
            weight = lut_kernel.w
            lut_weights.append(lut_kernel_w_preprocess(weight))
        return lut_weights

    def hook(module, input, output):
        module_info = {"type": module.__class__.__name__}
        
        is_6layer_model = "6layer" in model_name.lower()

        if isinstance(module, lut_quant):
            module_info.update({"in_channel": module.in_num,
                                "out_channel": module.out_num,
                                "lut_weights": quant_w_preprocess(module)
                                })
            execution_order.append(module_info)
        elif isinstance(module, lut_conv):
            layer_type = "lut_res" if is_6layer_model else module.__class__.__name__
            module_info.update({"type": layer_type,
                                "kernel_size": module.kernel_size,
                                "stride": module.stride,
                                "padding": module.padding,
                                "out_channel": module.out_channel,
                                "lut_weights": lut_kernel_w_preprocess(module.fc_conv.lut_kernel.w),
                                "row": input[0].size(2),
                                "col": input[0].size(3)
                                })
            execution_order.append(module_info)
        elif isinstance(module, lut_fc):
            module_info.update({"lut_num": module.out_num,
                                "lut_weights": lut_kernel_w_preprocess(module.lut_kernel.w)
                                })
            execution_order.append(module_info)
        elif isinstance(module, nn.BatchNorm2d):
            module_info.update({"threshold": torch.ceil(module.running_mean).cpu().tolist()})
            execution_order.append(module_info)

    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook))

    # Perform a forward pass to trace execution order
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Save to JSON
    with open(file_path, "w") as json_file:
        model_info = {}
        model_info['model_name'] = model_name
        model_info['layers'] = execution_order
        json.dump(model_info, json_file, indent=4)

