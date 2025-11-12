import math
from quant_layer import *
from conv_layer import *
from res_layer import *
from fc_layer import *
from group_sum import *
from buffer_to_fc import *
from fc_in_conn import *

import json
import math

def generate_rtl(rtl_path, cfg=None):
    head_code = []
    def_code = []
    logic_code = []

    lut_size = cfg['lut_size']
    bit_width = cfg['bit_width']
    col = cfg['col']
    row = cfg['row']
    channel_in = cfg['channel_in']
    in_kernel_num_row = cfg['layers'][0]['opt_h']
    in_kernel_num_col = cfg['layers'][0]['opt_w']
    out_class = cfg['out_class']
    module_name = cfg['model_name']

    head_code.append("`timescale 1ns/1ps\n")
    head_code.append("module "+module_name+"(\n")
    head_code.append("    input clk,\n")
    head_code.append("    input rst_n,\n")
    head_code.append("    input data_i_vld,\n")
    head_code.append("    input ["+str(bit_width*in_kernel_num_row*in_kernel_num_col*channel_in)+"-1:0] data_i,\n")
    head_code.append("    output data_o_vld,\n")
    head_code.append("    output ["+str(math.ceil(math.log2(out_class+1)))+"-1:0] data_o\n")
    head_code.append(");\n")
    head_code.append("\n")

    conv_idx = 0
    fc_idx = 0
    res_idx = 0
    res_bit_width = 1
    layer_type = ''
    for layer in (cfg['layers'] if cfg is not None else []):
        if layer['type']=='lut_quant':
            layer_type = layer['type']
            quant_channel_out = layer['out_channel']
            quant_out_kernel_num_row = layer['opt_h']
            quant_out_kernel_num_col = layer['opt_w']
            quant_channels = layer['quant_channels']
            def_code.append("wire ["+str(quant_channel_out*channel_in*quant_out_kernel_num_row*quant_out_kernel_num_col)+"-1:0] data_out_quant;\n")
            for quant_channel_idx in range(quant_channels):
                quant_lut_weights = layer['lut_weights'][quant_channel_idx]
                def_code.append("wire ["+str(quant_channel_out*quant_out_kernel_num_row*quant_out_kernel_num_col)+"-1:0] data_out_quant_"+str(quant_channel_idx)+";\n")
                def_code.append("wire data_o_vld_quant_"+str(quant_channel_idx)+";\n")
                logic_code.append("quant_layer_"+str(quant_channel_idx)+" u_quant_layer_"+str(quant_channel_idx)+"(\n")
                logic_code.append("    .data_i_vld   (data_i_vld),\n")
                logic_code.append("    .data_i       (data_i["+str(bit_width*in_kernel_num_row*in_kernel_num_col*(quant_channel_idx+1)-1)+":"+str(bit_width*in_kernel_num_row*in_kernel_num_col*quant_channel_idx)+"]),\n")
                logic_code.append("    .data_o_vld   (data_o_vld_quant_"+str(quant_channel_idx)+"),\n")
                logic_code.append("    .data_o       (data_out_quant_"+str(quant_channel_idx)+")\n")
                logic_code.append(");\n")
                generate_quant_layer(rtl_path, "quant_layer_"+str(quant_channel_idx), 
                                     bit_width, 
                                     quant_channel_out, 
                                     quant_out_kernel_num_row, 
                                     quant_out_kernel_num_col, 
                                     lut_size, 
                                     quant_lut_weights)
            quant_bits_per_channel = quant_channel_out*quant_out_kernel_num_row*quant_out_kernel_num_col
            logic_code.append("assign data_out_quant = {")
            for bits_idx in range(quant_bits_per_channel-1,-1,0-quant_channel_out):
                for channel_idx in range(quant_channels-1, -1, -1):
                    if bits_idx==quant_bits_per_channel-1 and channel_idx==quant_channels-1:
                        logic_code.append("data_out_quant_"+str(channel_idx)+"["+str(bits_idx)+":"+str(bits_idx-quant_channel_out+1)+"],\n")
                    elif bits_idx-quant_channel_idx<=0 and channel_idx==0:
                        logic_code.append("                         data_out_quant_"+str(channel_idx)+"["+str(bits_idx)+":"+str(bits_idx-quant_channel_out+1)+"]};\n")
                    else:
                        logic_code.append("                         data_out_quant_"+str(channel_idx)+"["+str(bits_idx)+":"+str(bits_idx-quant_channel_out+1)+"],\n")
            logic_code.append("\n")
            channel_in = quant_channels*quant_channel_out
            in_kernel_num_row = quant_out_kernel_num_row
            in_kernel_num_col = quant_out_kernel_num_col
        elif layer['type'] in ['lut_conv']:
            layer_type = layer['type']
            conv_channel_out = layer['out_channel']
            conv_out_kernel_num_row = layer['opt_h']
            conv_out_kernel_num_col = layer['opt_w']
            def_code.append("wire ["+str(channel_in*in_kernel_num_row*in_kernel_num_col)+"-1:0] data_i_conv_"+str(conv_idx)+";\n")
            def_code.append("wire data_i_vld_conv_"+str(conv_idx)+";\n")
            def_code.append("wire data_o_vld_conv_"+str(conv_idx)+";\n")
            def_code.append("wire ["+str(conv_channel_out*conv_out_kernel_num_row*conv_out_kernel_num_col)+"-1:0] data_o_conv_"+str(conv_idx)+";\n")
            logic_code.append("\n")
            if conv_idx==0:
                logic_code.append("assign data_i_conv_"+str(conv_idx)+" = data_out_quant;\n")
                logic_code.append("assign data_i_vld_conv_"+str(conv_idx)+" = data_o_vld_quant_0;\n")
            else:
                logic_code.append("assign data_i_conv_"+str(conv_idx)+" = data_o_conv_"+str(conv_idx-1)+";\n")
                logic_code.append("assign data_i_vld_conv_"+str(conv_idx)+" = data_o_vld_conv_"+str(conv_idx-1)+";\n")
            logic_code.append("\n")
            logic_code.append("conv_layer_"+str(conv_idx)+" u_conv_layer_"+str(conv_idx)+"(\n")
            logic_code.append("    .clk            (clk),\n")
            logic_code.append("    .rst_n          (rst_n),\n")
            logic_code.append("    .data_i_vld     (data_i_vld_conv_"+str(conv_idx)+"),\n")
            logic_code.append("    .data_i         (data_i_conv_"+str(conv_idx)+"),\n")
            logic_code.append("    .data_o_vld     (data_o_vld_conv_"+str(conv_idx)+"),\n")
            logic_code.append("    .data_o         (data_o_conv_"+str(conv_idx)+")\n")
            logic_code.append(");\n")
            logic_code.append("\n")
            generate_conv_layer(rtl_path, "conv_layer_"+str(conv_idx),
                                layer['kernel_size'],
                                channel_in,
                                conv_channel_out,
                                layer['stride'],
                                layer['padding'][0]+layer['padding'][1],
                                layer['col'],
                                layer['row'],
                                in_kernel_num_row,
                                in_kernel_num_col,
                                conv_out_kernel_num_row,
                                conv_out_kernel_num_col,
                                layer['threshold'],
                                lut_size,
                                layer['lut_weights'])
            conv_idx += 1
            channel_in = conv_channel_out
            in_kernel_num_row = conv_out_kernel_num_row
            in_kernel_num_col = conv_out_kernel_num_col
            col = layer['col']//layer['stride']
            row = layer['row']//layer['stride']
        elif layer['type'] in ['lut_res']:
            layer_type = layer['type']
            res_channel_out = layer['out_channel']
            res_channel_in = layer['in_channel']
            res_out_kernel_num_row = layer['opt_h']
            res_out_kernel_num_col = layer['opt_w']
            def_code.append("wire ["+str(channel_in*in_kernel_num_row*in_kernel_num_col)+"-1:0] data_i_res_"+str(res_idx)+";\n")
            def_code.append("wire ["+str(channel_in*in_kernel_num_row*in_kernel_num_col*res_bit_width)+"-1:0] res_i_res_"+str(res_idx)+";\n")
            def_code.append("wire data_i_vld_res_"+str(res_idx)+";\n")
            def_code.append("wire data_o_vld_res_"+str(res_idx)+";\n")
            def_code.append("wire ["+str(res_channel_out*res_out_kernel_num_row*res_out_kernel_num_col)+"-1:0] data_o_res_"+str(res_idx)+";\n")
            logic_code.append("\n")
            if res_idx==0:
                logic_code.append("assign data_i_res_"+str(res_idx)+" = data_out_quant;\n")
                logic_code.append("assign data_i_vld_res_"+str(res_idx)+" = data_o_vld_quant_0;\n")
                logic_code.append("assign res_i_res_"+str(res_idx)+" = 0;\n")
            else:
                logic_code.append("assign data_i_res_"+str(res_idx)+" = data_o_res_"+str(res_idx-1)+";\n")
                logic_code.append("assign data_i_vld_res_"+str(res_idx)+" = data_o_vld_res_"+str(res_idx-1)+";\n")
                logic_code.append("assign res_i_res_"+str(res_idx)+" = res_o_res_"+str(res_idx-1)+";\n")
            logic_code.append("\n")
            logic_code.append("res_layer_"+str(res_idx)+" u_res_layer_"+str(res_idx)+"(\n")
            logic_code.append("    .clk            (clk),\n")
            logic_code.append("    .rst_n          (rst_n),\n")
            logic_code.append("    .data_i_vld     (data_i_vld_res_"+str(res_idx)+"),\n")
            logic_code.append("    .data_i         (data_i_res_"+str(res_idx)+"),\n")
            if res_idx==0:
                logic_code.append("    .res_i          (0),\n")
            else:
                logic_code.append("    .res_i          (res_i_res_"+str(res_idx)+"),\n")
            logic_code.append("    .data_o_vld     (data_o_vld_res_"+str(res_idx)+"),\n")
            logic_code.append("    .data_o         (data_o_res_"+str(res_idx)+"),\n")
            logic_code.append("    .res_o          (res_o_res_"+str(res_idx)+")\n")
            logic_code.append(");\n")
            logic_code.append("\n")
            generate_res_layer(rtl_path, "res_layer_"+str(res_idx),
                                layer['kernel_size'],
                                channel_in,
                                res_channel_out,
                                layer['stride'],
                                layer['padding'][0]+layer['padding'][1],
                                layer['col'],
                                layer['row'],
                                res_bit_width,
                                in_kernel_num_row,
                                in_kernel_num_col,
                                res_out_kernel_num_row,
                                res_out_kernel_num_col,
                                layer['threshold'],
                                lut_size,
                                layer['lut_weights'])
            res_bit_width = max(math.ceil(math.log((layer['kernel_size']*layer['kernel_size']*channel_in/cfg['lut_size']),2)), res_bit_width)+1
            def_code.append("wire ["+str(res_channel_out*res_out_kernel_num_row*res_out_kernel_num_col*res_bit_width)+"-1:0] res_o_res_"+str(res_idx)+";\n")
            res_idx += 1
            channel_in = res_channel_out
            in_kernel_num_row = res_out_kernel_num_row
            in_kernel_num_col = res_out_kernel_num_col
            col = layer['col']//layer['stride']
            row = layer['row']//layer['stride']
        elif layer['type']=='lut_fc':
            if fc_idx==0:
                lut_num = layer['lut_num']
                def_code.append("wire ["+str(channel_in*in_kernel_num_row*in_kernel_num_col)+"-1:0] data_i_buffer_to_fc;\n")
                def_code.append("wire data_i_vld_buffer_to_fc;\n")
                def_code.append("wire ["+str(lut_size*lut_num)+"-1:0] data_o_buffer_to_fc;\n")
                def_code.append("wire data_o_vld_buffer_to_fc;\n")
                logic_code.append("\n")
                if layer_type=='lut_conv':
                    logic_code.append("assign data_i_buffer_to_fc = data_o_conv_"+str(conv_idx-1)+";\n")
                    logic_code.append("assign data_i_vld_buffer_to_fc = data_o_vld_conv_"+str(conv_idx-1)+";\n")
                elif layer_type=='lut_res':
                    logic_code.append("assign data_i_buffer_to_fc = data_o_res_"+str(res_idx-1)+";\n")
                    logic_code.append("assign data_i_vld_buffer_to_fc = data_o_vld_res_"+str(res_idx-1)+";\n")
                logic_code.append("\n")
                logic_code.append("buffer_to_fc u_buffer_to_fc(\n")
                logic_code.append("    .clk            (clk),\n")
                logic_code.append("    .rst_n          (rst_n),\n")
                logic_code.append("    .data_i_vld     (data_i_vld_buffer_to_fc),\n")
                logic_code.append("    .data_i         (data_i_buffer_to_fc),\n")
                logic_code.append("    .data_o_vld     (data_o_vld_buffer_to_fc),\n")
                logic_code.append("    .data_o         (data_o_buffer_to_fc)\n")
                logic_code.append(");\n")
                logic_code.append("\n")
                generate_buffer_to_fc(rtl_path,
                                           'buffer_to_fc',
                                           channel_in,
                                           col,
                                           row,
                                           in_kernel_num_row,
                                           in_kernel_num_col,
                                           lut_size,
                                           lut_num)
                def_code.append("wire ["+str(lut_num)+"-1:0] data_o_fc_"+str(fc_idx)+";\n")
                logic_code.append("fc_layer_0 u_fc_layer_0(\n")
                logic_code.append("    .data_i         (data_o_buffer_to_fc),\n")
                logic_code.append("    .data_o         (data_o_fc_0)\n")
                logic_code.append(");\n")
                logic_code.append("\n")      
                generate_fc_layer(rtl_path, "fc_layer_0", lut_num, lut_size, layer['lut_weights'])
            else:
                in_num = lut_num
                lut_num = layer['lut_num']
                def_code.append("wire ["+str(lut_num*lut_size)+"-1:0] data_o_fc_out_conn_"+str(fc_idx)+";\n")
                logic_code.append("\n")
                logic_code.append("fc_in_conn_"+str(fc_idx)+" u_fc_in_conn_"+str(fc_idx)+"(\n")
                logic_code.append("    .data_i         (data_o_fc_"+str(fc_idx-1)+"),\n")
                logic_code.append("    .data_o         (data_o_fc_out_conn_"+str(fc_idx)+")\n")
                logic_code.append(");\n")
                logic_code.append("\n")
                generate_fc_in_conn(rtl_path, "fc_in_conn_"+str(fc_idx), in_num, lut_num*lut_size)

                def_code.append("wire ["+str(lut_num)+"-1:0] data_o_fc_"+str(fc_idx)+";\n")
                logic_code.append("fc_layer_"+str(fc_idx)+" u_fc_layer_"+str(fc_idx)+"(\n")
                logic_code.append("    .data_i         (data_o_fc_out_conn_"+str(fc_idx)+"),\n")
                logic_code.append("    .data_o         (data_o_fc_"+str(fc_idx)+")\n")
                logic_code.append(");\n")
                logic_code.append("\n")
                generate_fc_layer(rtl_path, "fc_layer_"+str(fc_idx), lut_num, lut_size, layer['lut_weights'])
            fc_idx += 1

    # temp
    lut_num = 2000
    generate_group_sum(rtl_path, "group_sum", lut_num, out_class)
    def_code.append("reg data_i_vld_group_sum;\n")
    def_code.append("reg ["+str(lut_num)+"-1:0] data_i_group_sum;\n")

    logic_code.append("always @(posedge clk or negedge rst_n)\n")
    logic_code.append("begin\n")
    logic_code.append("    if (!rst_n)\n")
    logic_code.append("    begin\n")
    logic_code.append("        data_i_vld_group_sum <= 1'b0;\n")
    logic_code.append("        data_i_group_sum <= "+str(lut_num)+"'b0;\n")
    logic_code.append("    end\n")
    logic_code.append("    else\n")
    logic_code.append("    begin\n")
    logic_code.append("        data_i_vld_group_sum <= data_o_vld_buffer_to_fc;\n")
    logic_code.append("        data_i_group_sum <= data_o_fc_"+str(fc_idx-1)+";\n")
    logic_code.append("    end\n")
    logic_code.append("end\n")
    logic_code.append("\n")
    logic_code.append("group_sum u_group_sum(\n")
    logic_code.append("    .clk            (clk),\n")
    logic_code.append("    .rst_n          (rst_n),\n")
    logic_code.append("    .data_i_vld     (data_i_vld_group_sum),\n")
    logic_code.append("    .data_i         (data_i_group_sum),\n")
    logic_code.append("    .data_o_vld     (data_o_vld),\n")
    logic_code.append("    .data_o         (data_o)   \n")
    logic_code.append(");\n")
    logic_code.append("\n")
    logic_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in head_code+['\n']+def_code+['\n']+logic_code:
            f.write(i)

if __name__ == "__main__":
    cfg = json_parser("../src/model_execution_info.json")
    generate_rtl("./verilog", cfg)
