import math
from lut_tree import *

def generate_quant_layer(rtl_path, module_name, channel_in, channel_out, kernel_num_row, kernel_num_col, lut_size, lut_weights):
    """
    Generate a Verilog module for a quantization layer.
    Parameters:
        module_name (str): Name of the module.
        channel_in (int): Number of input channels. 
        channel_out (int): Number of output channels.
        kernel_num_row (int): Number of rows in the kernel.
        kernel_num_col (int): Number of columns in the kernel.
    """

    rtl_code = []
    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+module_name+"(\n")
    rtl_code.append("    input data_i_vld,\n")
    rtl_code.append("    input ["+str(channel_in*kernel_num_row*kernel_num_col-1)+":0] data_i,\n")
    rtl_code.append("    output data_o_vld,\n")
    rtl_code.append("    output ["+str(channel_out*kernel_num_row*kernel_num_col-1)+":0] data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    lut_weights_per_channel = [[] for _ in range(channel_out)]
    for layers in range(len(lut_weights)):
        lut_num_this_layer = len(lut_weights[layers])
        lut_num_each_channel = lut_num_this_layer//channel_out
        for channel_idx in range(channel_out):
            lut_weights_per_channel[channel_idx].append(lut_weights[layers][channel_idx*lut_num_each_channel:(channel_idx+1)*lut_num_each_channel])

    for row_idx in range(kernel_num_row):
        for col_idx in range(kernel_num_col):
            for channel_idx in range(channel_out):
                rtl_code.append(module_name+"_lut_tree_"+str(row_idx)+"_"+str(col_idx)+"_"+str(channel_idx)+" u_lut_tree_"+str(row_idx)+"_"+str(col_idx)+"_"+str(channel_idx)+" (\n")
                rtl_code.append("    .data_i(data_i["+str(row_idx*channel_in*kernel_num_col+col_idx*channel_in+channel_in-1)+":"+str(row_idx*channel_in*kernel_num_col+col_idx*channel_in)+"]),\n")
                rtl_code.append("    .data_o(data_o["+str(row_idx*channel_out*kernel_num_col+col_idx*channel_out+channel_idx)+"])\n")
                rtl_code.append(");\n")
                generate_lut_tree(rtl_path, module_name+"_lut_tree_"+str(row_idx)+"_"+str(col_idx)+"_"+str(channel_idx), channel_in, lut_size, lut_weights_per_channel[channel_idx])

    rtl_code.append("assign data_o_vld = data_i_vld;\n")
    rtl_code.append("\n")
    rtl_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)