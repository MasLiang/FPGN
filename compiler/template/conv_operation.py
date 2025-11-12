import math
from conv_kernel import *

def generate_conv_operation(rtl_path,
                            module_name,
                            kernel_size,
                            channel_in,
                            channel_out,
                            kernel_num_row,
                            kernel_num_col,
                            lut_size,
                            lut_weights,
                            threshold):
    """
    Generate a Verilog module for a convolution operation in a conv-layer with multiple kernels.
    Parameters:
        module_name (str): Name of the convolution layer module.
        kernel_size (int): Size of the convolution kernel.
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        kernel_num_row (int): Number of kernels in the row direction.  
        kernel_num_col (int): Number of kernels in the column direction.
        lut_size (int): Size of each LUT.
        lut_weights (list of list of string): Weights for each LUT in each channel.
        threshold (list of int): Threshold values for each output channel.  
    """
    lut_num_per_channel = math.ceil(kernel_size*kernel_size*channel_in/lut_size)
    repeat_num = lut_num_per_channel*lut_size-kernel_size*kernel_size*channel_in

    rtl_code = []
    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+module_name+"(\n")
    rtl_code.append("    input clk,\n")
    rtl_code.append("    input rst_n,\n")
    rtl_code.append("    input data_i_vld,\n")
    rtl_code.append("    input ["+str(channel_in*kernel_size*kernel_size*kernel_num_row*kernel_num_col)+"-1:0] data_i,\n")
    rtl_code.append("    output data_o_vld,\n")
    rtl_code.append("    output["+str(channel_out*kernel_num_row*kernel_num_col)+"-1:0] data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    
    for idx_r in range(kernel_num_row):
        for idx_c in range(kernel_num_col):
            rtl_code.append("wire ["+str(channel_in*kernel_size*kernel_size)+"-1:0] conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_i;\n")
            rtl_code.append("wire ["+str(channel_out-1)+":0] conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_o;\n")
            rtl_code.append("wire conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_o_vld;\n")
            rtl_code.append("assign conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_i = data_i["+str((idx_r*kernel_num_col+idx_c)*(channel_in*kernel_size*kernel_size))+"+:"+str(channel_in*kernel_size*kernel_size)+"];\n")
            rtl_code.append(module_name+"_conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+" u_kernel_inst_r"+str(idx_r)+"_c"+str(idx_c)+" (\n")
            rtl_code.append("    .clk(clk),\n")
            rtl_code.append("    .rst_n(rst_n),\n")
            rtl_code.append("    .data_i_vld(data_i_vld),\n")
            if repeat_num==0:
                rtl_code.append("    .data_i(conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_i),\n")
            else:
                rtl_code.append("    .data_i({conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_i["+str(repeat_num-1)+":0], conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_i}),\n")
            rtl_code.append("    .data_o_vld(conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_o_vld),\n")
            rtl_code.append("    .data_o(conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_o)\n")
            rtl_code.append(");\n")
            rtl_code.append("\n")
            generate_conv_kernel(rtl_path, module_name+"_conv_kernel_r"+str(idx_r)+"_c"+str(idx_c), lut_num_per_channel, channel_out, lut_size, lut_weights, threshold)
    if kernel_num_col==1 and kernel_num_row==1:
        rtl_code.append("assign data_o = conv_kernel_r0_c0_data_o;\n")
    else:
        rtl_code.append("assign data_o = {conv_kernel_r0_c0_data_o, \n")
    for idx_r in range(kernel_num_row):
        for idx_c in range(kernel_num_col):
            if idx_r == 0 and idx_c == 0:
                continue
            if idx_r==kernel_num_row-1 and idx_c==kernel_num_col-1:
                rtl_code.append("                conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_o};\n")
            else:
                rtl_code.append("                conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_o, \n")
    if kernel_num_col==1 and kernel_num_row==1:
        rtl_code.append("assign data_o_vld = conv_kernel_r0_c0_data_o_vld;\n")
    else:
        rtl_code.append("assign data_o_vld = conv_kernel_r0_c0_data_o_vld & \n")
    for idx_r in range(kernel_num_row):
        for idx_c in range(kernel_num_col):
            if idx_r == 0 and idx_c == 0:
                continue
            if idx_r==kernel_num_row-1 and idx_c==kernel_num_col-1:
                rtl_code.append("                conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_o_vld;\n")
            else:
                rtl_code.append("                conv_kernel_r"+str(idx_r)+"_c"+str(idx_c)+"_data_o_vld & \n")

    rtl_code.append("\n")
    rtl_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)