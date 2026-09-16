import math
from lut_vector import *
from lut_vector_pair import *
from popcount_compare import *

def generate_conv_kernel(rtl_path, module_name, lut_num_per_channel, channel_num, lut_size, lut_weights, threshold, packing=False, lut_packing=None, lut_pin_active=None):
    """
    Generate a Verilog module for a convolution kernel with LUTs.
    Parameters:
        rtl_path (str): Path to the RTL directory.
        module_name (str): Name of the convolution kernel module.
        lut_num_per_channel (int): Number of LUTs per channel.
        channel_num (int): Number of output channels.
        lut_size (int): Size of each LUT.
        lut_weights (list of list of lists of string): Weights for each LUT in each channel.
        threshold (list of int): Threshold values for each channel.
        packing (bool): Whether to apply packing.
        lut_packing (list of list of string): Packing information for each LUT in each channel.
        lut_pin_active (list of list of string): Active pin information for each LUT in each channel.
    """
    rtl_code = []
    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append(f"module "+module_name+"(\n")
    rtl_code.append("    input       clk,\n")
    rtl_code.append("    input       rst_n,\n")
    rtl_code.append("    input       data_i_vld,\n")
    rtl_code.append("    input       ["+str(lut_num_per_channel*lut_size)+"-1:0] data_i,\n")
    rtl_code.append("    output      reg data_o_vld,\n")
    rtl_code.append("    output      ["+str(channel_num)+"-1:0] data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    if packing:
        pair_num = channel_num//2
        pair_num_remain = channel_num%2
        rtl_code.append("wire ["+str(lut_num_per_channel*2)+"-1:0] lut_out_pair [0:"+str(pair_num+pair_num_remain)+"-1];\n")
    rtl_code.append("wire ["+str(lut_num_per_channel)+"-1:0] lut_out [0:"+str(channel_num)+"-1];\n")
    rtl_code.append("\n")
    rtl_code.append("always @(posedge clk or negedge rst_n)\n")
    rtl_code.append("begin\n")
    rtl_code.append("    if (!rst_n) \n")
    rtl_code.append("        data_o_vld     <=      1'b0;\n")
    rtl_code.append("    else \n")
    rtl_code.append("        data_o_vld <= data_i_vld;\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")
    if packing:
        for pair_idx in range(pair_num):
            rtl_code.append(module_name+"_lut_vector_pair"+str(pair_idx)+" u_lut_vector_pair"+str(pair_idx)+" (\n")
            rtl_code.append("    .data_i     (data_i),\n")
            rtl_code.append("    .data_o     (lut_out_pair["+str(pair_idx)+"])\n")
            rtl_code.append(");\n")
            rtl_code.append("assign lut_out["+str(pair_idx*2)+"] = lut_out_pair["+str(pair_idx)+"]["+str(lut_num_per_channel-1)+":0];\n")
            rtl_code.append("assign lut_out["+str(pair_idx*2+1)+"] = lut_out_pair["+str(pair_idx)+"]["+str(lut_num_per_channel*2-1)+":"+str(lut_num_per_channel)+"];\n")
            rtl_code.append("\n")
            generate_lut_vector_pair(rtl_path, module_name+"_lut_vector_pair"+str(pair_idx), lut_num_per_channel, lut_size, lut_weights[pair_idx*lut_num_per_channel:(pair_idx+1)*lut_num_per_channel], lut_packing[pair_idx], lut_pin_active[lut_packing[pair_idx]['u']], tree_flg=False)
        if pair_num_remain==1:
            rtl_code.append(module_name+"_lut_vector_pair"+str(pair_num)+" u_lut_vector_pair"+str(pair_num)+" (\n")
            rtl_code.append("    .data_i     (data_i),\n")
            rtl_code.append("    .data_o     (lut_out_pair["+str(pair_num)+"]["+str(lut_num_per_channel-1)+":0])\n")
            rtl_code.append(");\n")
            rtl_code.append("assign lut_out["+str(pair_num*2)+"] = lut_out_pair["+str(pair_num)+"]["+str(lut_num_per_channel-1)+":0];\n")
            rtl_code.append("\n")
            generate_lut_vector(rtl_path, module_name+"_lut_vector_pair"+str(pair_num), lut_num_per_channel, lut_size, lut_weights[pair_num*lut_num_per_channel:(pair_num+1)*lut_num_per_channel])
    else:
        for channel_idx in range(channel_num):
            rtl_code.append(module_name+"_lut_vector_channel"+str(channel_idx)+" u_lut_vector_channel"+str(channel_idx)+" (\n")
            rtl_code.append("    .data_i     (data_i),\n")
            rtl_code.append("    .data_o     (lut_out["+str(channel_idx)+"])\n")
            rtl_code.append(");\n")
            rtl_code.append("\n")
            generate_lut_vector(rtl_path, module_name+"_lut_vector_channel"+str(channel_idx), lut_num_per_channel, lut_size, lut_weights[channel_idx*lut_num_per_channel:(channel_idx+1)*lut_num_per_channel])
    for channel_idx in range(channel_num):
        rtl_code.append("popcount_compare_bitnum_"+str(lut_num_per_channel)+"_thr_"+str(int(threshold[channel_idx]))+" u_popcount_compare_"+str(channel_idx)+"(\n")
        rtl_code.append("    .clk        (clk),\n")
        rtl_code.append("    .rst_n      (rst_n),\n")
        rtl_code.append("    .data_i_vld (data_i_vld),\n")
        rtl_code.append("    .data_i     (lut_out["+str(channel_idx)+"]),\n")
        rtl_code.append("    .data_o     (data_o["+str(channel_idx)+"])\n")
        rtl_code.append(");\n")
        rtl_code.append("\n")
        generate_popcount_compare(rtl_path, "popcount_compare_bitnum_"+str(lut_num_per_channel)+"_thr_"+str(int(threshold[channel_idx])), lut_num_per_channel, int(threshold[channel_idx]))
    rtl_code.append("\n")
    rtl_code.append("endmodule\n")
    
    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)