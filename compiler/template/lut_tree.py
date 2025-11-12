import math
from lut_vector import *

def generate_lut_tree(rtl_path, module_name, input_num, lut_size, lut_weights):
    """
    Generate a Verilog module for a LUT tree.
    Parameters:  
        module_name (str): Name of the LUT tree module.
        input_num (int): Number of inputs to the LUT tree.
        lut_size (int): Size of each LUT.  
        lut_weights (list of list of string): Weights for each LUT in each layer.
    """
    lut_num_temp = math.ceil(input_num / lut_size)
    lut_num_per_layer = [lut_num_temp]
    lut_layer = 1
    while lut_num_temp > 1:
        lut_num_temp  = math.ceil(lut_num_temp / lut_size)
        lut_layer += 1
        lut_num_per_layer.append(lut_num_temp)

    rtl_code = []

    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+module_name+"(\n")
    rtl_code.append("    input  ["+str(input_num)+"-1:0] data_i,\n")
    rtl_code.append("    output data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")

    for layer_idx in range(lut_layer):
        rtl_code.append("wire ["+str(lut_num_per_layer[layer_idx])+"-1:0] vector_o_"+str(layer_idx)+";\n")
        rtl_code.append(module_name+"_lut_vector_"+str(layer_idx)+" u_lut_vector_"+str(layer_idx)+" (\n")
        if layer_idx == 0:
            rtl_code.append("    .data_i     ({"+str(lut_num_per_layer[layer_idx]*lut_size-input_num)+"'b0, data_i}),\n")
        else:
            rtl_code.append("    .data_i     ({"+str(lut_num_per_layer[layer_idx]*lut_size-lut_num_per_layer[layer_idx-1])+"'b0, vector_o_"+str(layer_idx-1)+"}),\n")
        rtl_code.append("    .data_o     (vector_o_"+str(layer_idx)+")\n")
        rtl_code.append(");\n")
        rtl_code.append("\n")
        generate_lut_vector(rtl_path, module_name+"_lut_vector_"+str(layer_idx), lut_num_per_layer[layer_idx], lut_size, lut_weights[layer_idx], True)

    rtl_code.append("assign data_o = vector_o_"+str(layer_idx)+";\n")
    rtl_code.append("endmodule\n")
    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)