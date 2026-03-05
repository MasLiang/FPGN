import math
from res_operation import *
from buffer_res_to_res import *

def generate_res_layer(rtl_path,
                        module_name, 
                        kernel_size, 
                        channel_in, 
                        channel_out, 
                        stride_size, 
                        padding_size, 
                        col, 
                        row, 
                        in_bit_width,
                        in_kernel_num_row, 
                        in_kernel_num_col, 
                        out_kernel_num_row, 
                        out_kernel_num_col, 
                        threshold, 
                        lut_size,
                        lut_weights):
    """
    Generate a Verilog module for a res layer.
    Parameters:
        rtl_path (str): Path to the RTL directory.
        module_name (str): Name of the res layer module.
        kernel_size (int): Size of the convolution kernel.             
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        stride_size (int): Stride size for the convolution.
        padding_size (int): Padding size for the convolution.
        col (int): Number of columns in the input feature map.
        row (int): Number of rows in the input feature map.
        in_bit_width (int): Bit width of the input data.
        in_kernel_num_row (int): Number of kernels in the row direction for input.
        in_kernel_num_col (int): Number of kernels in the column direction for input.  
        out_kernel_num_row (int): Number of kernels in the row direction for output.
        out_kernel_num_col (int): Number of kernels in the column direction for output.
        threshold (list of int): Threshold values for each output channel.
        lut_size (int): Size of the LUT.
        lut_weights (list of list of string): Weights for each LUT in each channel.
    """

    out_bit_width = max(int(math.ceil(math.log2((channel_in*kernel_size*kernel_size)//6 + 1))), in_bit_width)+1
    rtl_code = []
    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+module_name+"(\n")
    rtl_code.append("    input clk,\n")
    rtl_code.append("    input rst_n,\n")
    rtl_code.append("    input data_i_vld,\n")
    rtl_code.append("    input ["+str(channel_in*in_kernel_num_row*in_kernel_num_col)+"-1:0] data_i,\n")
    rtl_code.append("    input ["+str(channel_in*in_kernel_num_row*in_kernel_num_col*in_bit_width)+"-1:0] res_i,\n")
    rtl_code.append("    output data_o_vld,\n")
    rtl_code.append("    output["+str(channel_out*out_kernel_num_row*out_kernel_num_col)+"-1:0] data_o,\n")
    rtl_code.append("    output["+str(channel_out*out_bit_width*out_kernel_num_row*out_kernel_num_col)+"-1:0] res_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    rtl_code.append("wire ["+str(channel_in*kernel_size*kernel_size*out_kernel_num_row*out_kernel_num_col)+"-1:0] buffer2res_data_o;\n")
    rtl_code.append("wire ["+str(channel_in*stride_size*stride_size*out_kernel_num_row*out_kernel_num_col*in_bit_width)+"-1:0] buffer2res_res_o;\n")
    rtl_code.append("wire buffer2res_data_o_vld;\n")
    rtl_code.append("\n")
    rtl_code.append(module_name+"_buffer_to_res u_buffer(\n")
    rtl_code.append("    .clk            (clk),\n")
    rtl_code.append("    .rst_n          (rst_n),\n")
    rtl_code.append("    .data_i_vld     (data_i_vld),\n")
    rtl_code.append("    .data_i         (data_i),\n")
    rtl_code.append("    .res_i          (res_i),\n")
    rtl_code.append("    .data_o_vld     (buffer2res_data_o_vld),\n")
    rtl_code.append("    .data_o         (buffer2res_data_o),\n")
    rtl_code.append("    .res_o          (buffer2res_res_o)\n")
    rtl_code.append(");\n")
    generate_buffer_res_to_res(rtl_path, module_name+"_buffer_to_res", kernel_size, channel_in, stride_size,
                                 padding_size, col, row, in_bit_width, in_kernel_num_row, in_kernel_num_col,
                                 out_kernel_num_row, out_kernel_num_col)
    rtl_code.append("\n")
    rtl_code.append(module_name+"_res_operation u_res_operation(\n")
    rtl_code.append("    .clk           (clk),\n")
    rtl_code.append("    .rst_n         (rst_n),\n")
    rtl_code.append("    .data_i_vld    (buffer2res_data_o_vld),\n")
    rtl_code.append("    .data_i        (buffer2res_data_o),\n")
    rtl_code.append("    .res_i         (buffer2res_res_o),\n")
    rtl_code.append("    .data_o_vld    (data_o_vld),\n")
    rtl_code.append("    .data_o        (data_o),\n")
    rtl_code.append("    .res_o         (res_o)\n")
    rtl_code.append(");\n")
    generate_res_operation(rtl_path, module_name+"_res_operation", kernel_size, stride_size, in_bit_width, out_bit_width, channel_in, channel_out, 
                           out_kernel_num_row, out_kernel_num_col, lut_size, lut_weights, threshold)
    rtl_code.append("\n")
    rtl_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)
