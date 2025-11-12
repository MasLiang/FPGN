import math

def generate_lut_vector(rtl_path, module_name, lut_num, lut_size, lut_weights, tree_flg=False):
    """
    Generate a Verilog module for a LUT vector.
    Parameters:
        rtl_path (str): Path to the RTL directory.
        module_name (str): Name of the LUT vector module.
        lut_num (int): Number of LUTs.
        lut_size (int): Number of input of each LUT.
        lut_weights (list of string): Weights for each LUT.
    """

    rtl_code = []

    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+module_name+"(\n")
    rtl_code.append("    input       ["+str(lut_num*lut_size)+"-1:0] data_i,\n")
    rtl_code.append("    output      ["+str(lut_num)+"-1:0] data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    for lut_idx in range(lut_num):
        rtl_code.append("LUT6 #(\n")
        rtl_code.append("   .INIT(64'h"+lut_weights[lut_idx]+")\n")
        rtl_code.append(") LUT6_inst_"+str(lut_idx)+" (\n")
        rtl_code.append("   .O(data_o["+str(lut_idx)+"]),\n")
        rtl_code.append("   .I0(data_i["+str(lut_idx)+"*6+0]),\n")
        rtl_code.append("   .I1(data_i["+str(lut_idx)+"*6+1]),\n")
        rtl_code.append("   .I2(data_i["+str(lut_idx)+"*6+2]),\n")
        rtl_code.append("   .I3(data_i["+str(lut_idx)+"*6+3]),\n")
        rtl_code.append("   .I4(data_i["+str(lut_idx)+"*6+4]),\n")
        rtl_code.append("   .I5(data_i["+str(lut_idx)+"*6+5])\n")
        rtl_code.append(");\n")
            
    rtl_code.append("endmodule\n")
    
    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)