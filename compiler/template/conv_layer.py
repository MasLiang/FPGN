import math
from conv_operation import *
from buffer_conv_to_conv import *

def generate_conv_layer(rtl_path,
                        module_name, 
                        kernel_size, 
                        channel_in, 
                        channel_out, 
                        stride_size, 
                        padding_size, 
                        col, 
                        row, 
                        in_kernel_num_row, 
                        in_kernel_num_col, 
                        out_kernel_num_row, 
                        out_kernel_num_col, 
                        threshold, 
                        lut_size,
                        lut_weights):
    """
    Generate a Verilog module for a convolution layer.
    Parameters:
        rtl_path (str): Path to the RTL directory.
        module_name (str): Name of the convolution layer module.
        kernel_size (int): Size of the convolution kernel.             
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        stride_size (int): Stride size for the convolution.
        padding_size (int): Padding size for the convolution.
        col (int): Number of columns in the input feature map.
        row (int): Number of rows in the input feature map.
        in_kernel_num_row (int): Number of kernels in the row direction for input.
        in_kernel_num_col (int): Number of kernels in the column direction for input.  
        out_kernel_num_row (int): Number of kernels in the row direction for output.
        out_kernel_num_col (int): Number of kernels in the column direction for output.
        threshold (list of int): Threshold values for each output channel.
        lut_size (int): Size of the LUT.
        lut_weights (list of list of string): Weights for each LUT in each channel.
    """

    rtl_code = []
    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+module_name+"(\n")
    rtl_code.append("    input clk,\n")
    rtl_code.append("    input rst_n,\n")
    rtl_code.append("    input data_i_vld,\n")
    rtl_code.append("    input ["+str(channel_in*in_kernel_num_row*in_kernel_num_col)+"-1:0] data_i,\n")
    rtl_code.append("    output data_o_vld,\n")
    rtl_code.append("    output["+str(channel_out*out_kernel_num_row*out_kernel_num_col)+"-1:0] data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    rtl_code.append("wire ["+str(channel_in*kernel_size*kernel_size*out_kernel_num_row*out_kernel_num_col)+"-1:0] buffer2conv_data_o;\n")
    rtl_code.append("wire buffer2conv_data_o_vld;\n")
    rtl_code.append("\n")
    rtl_code.append(module_name+"_buffer_to_conv u_buffer(\n")
    rtl_code.append("    .clk            (clk),\n")
    rtl_code.append("    .rst_n          (rst_n),\n")
    rtl_code.append("    .data_i_vld     (data_i_vld),\n")
    rtl_code.append("    .data_i         (data_i),\n")
    rtl_code.append("    .data_o_vld     (buffer2conv_data_o_vld),\n")
    rtl_code.append("    .data_o         (buffer2conv_data_o)\n")
    rtl_code.append(");\n")
    generate_buffer_conv_to_conv(rtl_path, module_name+"_buffer_to_conv", kernel_size, channel_in, stride_size, 
                                 padding_size, col, row, in_kernel_num_row, in_kernel_num_col, out_kernel_num_row, 
                                 out_kernel_num_col)
    rtl_code.append("\n")
    rtl_code.append(module_name+"_conv_operation u_conv_operation(\n")
    rtl_code.append("    .clk           (clk),\n")
    rtl_code.append("    .rst_n         (rst_n),\n")
    rtl_code.append("    .data_i_vld    (buffer2conv_data_o_vld),\n")
    rtl_code.append("    .data_i        (buffer2conv_data_o),\n")
    rtl_code.append("    .data_o_vld    (data_o_vld),\n")
    rtl_code.append("    .data_o        (data_o)\n")
    rtl_code.append(");\n")
    generate_conv_operation(rtl_path, module_name+"_conv_operation", kernel_size, channel_in, channel_out, 
                            out_kernel_num_row, out_kernel_num_col, lut_size, lut_weights, threshold)
    rtl_code.append("\n")
    rtl_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)