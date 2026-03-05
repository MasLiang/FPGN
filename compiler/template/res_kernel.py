import math
from lut_vector import *
from popcount_res import *
from res_max_pooling import *

def generate_res_kernel(rtl_path, module_name, lut_num_per_channel, in_bit_width, out_bit_width, stride_size, channel_in, channel_out, lut_size, lut_weights, threshold):
    """
    Generate a Verilog module for a residual kernel with LUTs.
    Parameters:
        rtl_path (str): Path to the RTL directory.
        module_name (str): Name of the residual kernel module.
        lut_num_per_channel (int): Number of LUTs per channel.
        in_bit_width (int): Bit width of the input data.
        out_bit_width (int): Bit width of the output data.
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        lut_size (int): Size of each LUT.
        lut_weights (list of list of lists of string): Weights for each LUT in each channel.
        threshold (list of int): Threshold values for each channel.
    """
    rtl_code = []
    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append(f"module "+module_name+"(\n")
    rtl_code.append("    input       clk,\n")
    rtl_code.append("    input       rst_n,\n")
    rtl_code.append("    input       data_i_vld,\n")
    rtl_code.append("    input       ["+str(lut_num_per_channel*lut_size)+"-1:0] data_i,\n")
    rtl_code.append("    input       ["+str(channel_in*in_bit_width*stride_size*stride_size)+"-1:0] res_i,\n")
    rtl_code.append("    output      reg data_o_vld,\n")
    rtl_code.append("    output      ["+str(channel_out)+"-1:0] data_o,\n")
    rtl_code.append("    output      ["+str(channel_out*out_bit_width)+"-1:0] res_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    rtl_code.append("wire ["+str(lut_num_per_channel)+"-1:0] lut_out [0:"+str(channel_out)+"-1];\n")
    rtl_code.append("wire ["+str(math.ceil(math.log(lut_num_per_channel, 2)))+"-1:0] data_o_popcount [0:"+str(channel_out)+"-1];\n")
    rtl_code.append("wire ["+str(in_bit_width*stride_size*stride_size)+"-1:0] max_o_channel [0:"+str(channel_out)+"-1];\n")
    rtl_code.append("wire ["+str(channel_out*in_bit_width*stride_size*stride_size)+"-1:0] res_i_padding;\n")
    rtl_code.append("\n")

    if channel_in==channel_out:
        rtl_code.append("assign res_i_padding = res_i;\n")
    else:
        rtl_code.append("assign res_i_padding = {")
        rtl_code.append("{" + str(((channel_out - channel_in)//2) * in_bit_width * stride_size * stride_size) + "{1'b0}}, res_i, {" + str((channel_out-channel_in-((channel_out - channel_in)//2)) * in_bit_width * stride_size * stride_size) + "{1'b0}}};\n")
    rtl_code.append("always @(posedge clk or negedge rst_n)\n")
    rtl_code.append("begin\n")
    rtl_code.append("    if (!rst_n) \n")
    rtl_code.append("        data_o_vld     <=      1'b0;\n")
    rtl_code.append("    else \n")
    rtl_code.append("        data_o_vld <= data_i_vld;\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")
    for channel_idx in range(channel_out):
        rtl_code.append(module_name+"_lut_vector_channel"+str(channel_idx)+" u_lut_vector_channel"+str(channel_idx)+" (\n")
        rtl_code.append("    .data_i     (data_i),\n")
        rtl_code.append("    .data_o     (lut_out["+str(channel_idx)+"])\n")
        rtl_code.append(");\n")
        rtl_code.append("\n")
        rtl_code.append("popcount_res_bitnum_"+str(lut_num_per_channel)+" u_popcount_res_"+str(channel_idx)+"(\n")
        rtl_code.append("    .clk        (clk),\n")
        rtl_code.append("    .rst_n      (rst_n),\n")
        rtl_code.append("    .data_i_vld (data_i_vld),\n")
        rtl_code.append("    .data_i     (lut_out["+str(channel_idx)+"]),\n")
        rtl_code.append("    .data_o     (data_o_popcount["+str(channel_idx)+"])\n")
        rtl_code.append(");\n")
        rtl_code.append("\n")
        rtl_code.append(module_name+"_res_max_pooling_"+str(channel_idx)+" u_res_max_pooling_"+str(channel_idx)+" (\n")
        rtl_code.append("    .clk        (clk),\n")
        rtl_code.append("    .rst_n      (rst_n),\n")
        rtl_code.append("    .data_i_vld (data_i_vld),\n")
        rtl_code.append("    .data_i     (res_i_padding["+str(channel_idx*in_bit_width*stride_size*stride_size)+"+: "+str(in_bit_width*stride_size*stride_size)+"]),\n")
        rtl_code.append("    .data_o_vld (),\n")
        rtl_code.append("    .data_o     (max_o_channel["+str(channel_idx)+"])\n")
        rtl_code.append(");\n")
        rtl_code.append("\n")
        generate_popcount_res(rtl_path, "popcount_res_bitnum_"+str(lut_num_per_channel), lut_num_per_channel)
        generate_lut_vector(rtl_path, module_name+"_lut_vector_channel"+str(channel_idx), lut_num_per_channel, lut_size, lut_weights[channel_idx*lut_num_per_channel:(channel_idx+1)*lut_num_per_channel])
        generate_res_max_pooling(rtl_path, module_name+"_res_max_pooling_"+str(channel_idx), stride_size, in_bit_width)
        # add & compare
        rtl_code.append("assign res_o["+str(channel_idx*out_bit_width)+"+: "+str(out_bit_width)+"] = max_o_channel["+str(channel_idx)+"] + data_o_popcount["+str(channel_idx)+"];\n")
        rtl_code.append("assign data_o["+str(channel_idx)+"] = (res_o["+str(channel_idx*out_bit_width)+"+: "+str(out_bit_width)+"] >= "+str(int(threshold[channel_idx]))+") ? 1'b1 : 1'b0;\n")
    rtl_code.append("\n")
    rtl_code.append("endmodule\n")
    
    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)
