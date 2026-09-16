import math
import random

def generate_lut_vector_pair(rtl_path, module_name, lut_num, lut_size, lut_weights, lut_packing, lut_pin_active, tree_flg=False):
    """
    Generate a Verilog module for a LUT vector.
    Parameters:
        rtl_path (str): Path to the RTL directory.
        module_name (str): Name of the LUT vector module.
        lut_num (int): Number of LUTs.
        lut_size (int): Number of input of each LUT.
        lut_weights (list of string): Weights for each LUT.
        lut_packing (list of string): Packing information for each LUT.
        lut_pin_active (list of string): Active pin information for each LUT.
        tree_flg (bool): Whether to generate a tree structure.
    """

    rtl_code = []

    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+module_name+"(\n")
    rtl_code.append("    input       ["+str(lut_num*lut_size)+"-1:0] data_i,\n")
    rtl_code.append("    output      ["+str(lut_num*2)+"-1:0] data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    for lut_idx in range(lut_num):
        packing_info = lut_packing['types'][lut_idx]
        pin_active_info = lut_pin_active[lut_idx]
        if packing_info=="SPLIT":
            rtl_code.append("LUT6 #(\n")
            rtl_code.append("   .INIT(64'h"+lut_weights[lut_idx][0]+")\n")
            rtl_code.append(") LUT6_inst_"+str(lut_idx)+"_u (\n")
            rtl_code.append("   .O(data_o["+str(lut_idx)+"]),\n")
            rtl_code.append("   .I0(data_i["+str(lut_idx)+"*6+0]),\n")
            rtl_code.append("   .I1(data_i["+str(lut_idx)+"*6+1]),\n")
            rtl_code.append("   .I2(data_i["+str(lut_idx)+"*6+2]),\n")
            rtl_code.append("   .I3(data_i["+str(lut_idx)+"*6+3]),\n")
            rtl_code.append("   .I4(data_i["+str(lut_idx)+"*6+4]),\n")
            rtl_code.append("   .I5(data_i["+str(lut_idx)+"*6+5])\n")
            rtl_code.append(");\n")
            rtl_code.append("LUT6 #(\n")
            rtl_code.append("   .INIT(64'h"+lut_weights[lut_idx][1]+")\n")
            rtl_code.append(") LUT6_inst_"+str(lut_idx)+"_v (\n")
            rtl_code.append("   .O(data_o["+str(lut_idx+lut_num)+"]),\n")
            rtl_code.append("   .I0(data_i["+str(lut_idx)+"*6+0]),\n")
            rtl_code.append("   .I1(data_i["+str(lut_idx)+"*6+1]),\n")
            rtl_code.append("   .I2(data_i["+str(lut_idx)+"*6+2]),\n")
            rtl_code.append("   .I3(data_i["+str(lut_idx)+"*6+3]),\n")
            rtl_code.append("   .I4(data_i["+str(lut_idx)+"*6+4]),\n")
            rtl_code.append("   .I5(data_i["+str(lut_idx)+"*6+5])\n")
            rtl_code.append(");\n")
        elif packing_info=="PACKED":
            rtl_code.append("LUT6_2 #(\n")
            rtl_code.append("   .INIT(64'h"+lut_weights[lut_idx]+")\n")
            rtl_code.append(") LUT6_inst_"+str(lut_idx)+" (\n")
            rtl_code.append("   .O6(data_o["+str(lut_idx)+"]),\n")
            rtl_code.append("   .O5(data_o["+str(lut_idx+lut_num)+"]),\n")
            rtl_code.append("   .I0(data_i["+str(lut_idx)+"*6+0]),\n")
            rtl_code.append("   .I1(data_i["+str(lut_idx)+"*6+1]),\n")
            rtl_code.append("   .I2(data_i["+str(lut_idx)+"*6+2]),\n")
            rtl_code.append("   .I3(data_i["+str(lut_idx)+"*6+3]),\n")
            rtl_code.append("   .I4(data_i["+str(lut_idx)+"*6+4]),\n")
            rtl_code.append("   .I5(1'b1)\n")
            rtl_code.append(");\n")
        elif packing_info=="SINGLE_V" or packing_info=="SINGLE_U":
            rtl_code.append("LUT6 #(\n")
            rtl_code.append("   .INIT(64'h"+lut_weights[lut_idx]+")\n")
            rtl_code.append(") LUT6_inst_"+str(lut_idx)+" (\n")
            if packing_info=="SINGLE_V":
                rtl_code.append("   .O(data_o["+str(lut_idx+lut_num)+"]),\n")
            else:
                rtl_code.append("   .O(data_o["+str(lut_idx)+"]),\n")
            rtl_code.append("   .I0(data_i["+str(lut_idx)+"*6+0]),\n")
            rtl_code.append("   .I1(data_i["+str(lut_idx)+"*6+1]),\n")
            rtl_code.append("   .I2(data_i["+str(lut_idx)+"*6+2]),\n")
            rtl_code.append("   .I3(data_i["+str(lut_idx)+"*6+3]),\n")
            rtl_code.append("   .I4(data_i["+str(lut_idx)+"*6+4]),\n")
            rtl_code.append("   .I5(data_i["+str(lut_idx)+"*6+5])\n")
            rtl_code.append(");\n")
            if packing_info=="SINGLE_V":
                rtl_code.append("assign data_o["+str(lut_idx)+"] = 1'b0;\n")
            else:
                rtl_code.append("assign data_o["+str(lut_idx+lut_num)+"] = 1'b0;\n")
        else:
            rtl_code.append("assign data_o["+str(lut_idx)+"] = 1'b0;\n")
            rtl_code.append("assign data_o["+str(lut_idx+lut_num)+"] = 1'b0;\n")
        #rtl_code.append("localparam LUT_INIT_"+str(lut_idx)+"=64'h"+lut_weights[lut_idx]+";\n")
        #rtl_code.append("wire ["+str(lut_size-1)+":0] lut_inputs_"+str(lut_idx)+";\n")
        #rtl_code.append("assign lut_inputs_"+str(lut_idx)+" = data_i["+str((lut_idx+1)*lut_size-1)+":"+str(lut_idx*lut_size)+"];\n")
        #rtl_code.append("assign data_o["+str(lut_idx)+"] = LUT_INIT_"+str(lut_idx)+"[lut_inputs_"+str(lut_idx)+"];\n")
        #rtl_code.append("\n")
            
    rtl_code.append("endmodule\n")
    
    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)
