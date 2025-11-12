import math
from popcount_2lut import *
from popcount_3lut import *

def generate_popcount_compare(rtl_path, module_name, width, threshold):
    lut6_count = math.ceil(width / 6)
    count_width = (width + 1).bit_length()
    width_bits = math.ceil(math.log2(width+1))
    
    rtl_code = []
    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+str(module_name)+"(\n")
    rtl_code.append("    input  clk,                        \n")
    rtl_code.append("    input  rst_n,                      \n")
    rtl_code.append("    input  data_i_vld,                 \n")
    rtl_code.append("    input  ["+str(width)+"-1:0] data_i,         \n")
    rtl_code.append("    output data_o  \n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    rtl_code.append("wire [2:0] lut6_results [0:"+str(lut6_count)+"-1];\n\n")
    rtl_code.append("reg [2:0] lut6_results_reg [0:"+str(lut6_count)+"-1];\n\n")
    generate_popcount_2lut(rtl_path)
    generate_popcount_3lut(rtl_path)
    for i in range(lut6_count):
        if (i == lut6_count-1 and width % 6 == 1):
            rtl_code.append("assign lut6_results["+str(i)+"] = {{1'b0, data_i["+str(6*i)+"]}};\n")
        elif (i == lut6_count-1 and width % 6 == 2):
            rtl_code.append("popcount_2lut u_2lut_"+str(i)+"(.bits({{1'b0, data_i["+str(6*i+1)+":"+str(6*i)+"]}}), .count(lut6_results["+str(i)+"][1:0]));\n")
            rtl_code.append("assign lut6_results["+str(i)+"][2] = 1'b0;\n")
        elif (i == lut6_count-1 and width % 6 == 3):
            rtl_code.append("popcount_2lut u_2lut_"+str(i)+"(.bits(data_i["+str(6*i+2)+":"+str(6*i)+"]), .count(lut6_results["+str(i)+"][1:0]));\n")
            rtl_code.append("assign lut6_results["+str(i)+"][2] = 1'b0;\n")
        elif (i == lut6_count-1 and width % 6 == 4):
            rtl_code.append("popcount_3lut u_3lut_"+str(i)+"(.bits({{2'b0, data_i["+str(6*i+3)+":"+str(6*i)+"]}}), .count(lut6_results["+str(i)+"]));\n")
        elif (i == lut6_count-1 and width % 6 == 5):
            rtl_code.append("popcount_3lut u_3lut_"+str(i)+"(.bits({{1'b0, data_i["+str(6*i+4)+":"+str(6*i)+"]}}), .count(lut6_results["+str(i)+"]));\n")
        else:
            rtl_code.append("popcount_3lut u_3lut_"+str(i)+"(.bits(data_i["+str(6*i+5)+":"+str(6*i)+"]), .count(lut6_results["+str(i)+"]));\n")
    for i in range(lut6_count):
        rtl_code.append("always @(posedge clk or negedge rst_n) begin\n")
        rtl_code.append("    if (!rst_n) \n")
        rtl_code.append("        lut6_results_reg["+str(i)+"] <= 3'b0;\n")
        rtl_code.append("    else \n")
        rtl_code.append("        lut6_results_reg["+str(i)+"] <= lut6_results["+str(i)+"];\n")
        rtl_code.append("end\n\n")
    
    def generate_adder_tree(elements, level=0, width=64, width_bits=7):
        if elements <= 1:
            return elements, []
        
        next_level_elements = (elements + 1) // 2
        code = []
        
        if level == 0:
            code.append("    wire ["+str(width_bits)+"-1:0] stage"+str(level+1)+" [0:"+str(next_level_elements-1)+"];\n")
            for i in range(elements//2):
                code.append("assign stage"+str(level+1)+"["+str(i)+"] = {"+str(width_bits-3)+"'b0, lut6_results_reg["+str(2*i)+"]} + ")
                code.append("{"+str(width_bits-3)+"'b0, lut6_results_reg["+str(2*i+1)+"]};\n")
            if elements % 2 == 1:
                code.append("assign stage"+str(level+1)+"["+str(elements//2)+"] = {"+str(width_bits-2)+"'b0, lut6_results_reg["+str(elements-1)+"]};\n")
            code.append("\n")
        else:
            code.append("    wire ["+str(width_bits)+"-1:0] stage"+str(level+1)+" [0:"+str(next_level_elements-1)+"];\n")
            for i in range(elements//2):
                code.append("assign stage"+str(level+1)+"["+str(i)+"] = stage"+str(level)+"["+str(2*i)+"] + stage"+str(level)+"["+str(2*i+1)+"];\n")
            if elements % 2 == 1:
                code.append("assign stage"+str(level+1)+"["+str(elements//2)+"] = stage"+str(level)+"["+str(elements-1)+"];\n")
            code.append("\n")

        return next_level_elements, code
    
    all_tree_code = []
    current_elements = lut6_count
    level = 0
    
    while current_elements > 1:
        current_elements, code = generate_adder_tree(current_elements, level, width, width_bits)
        all_tree_code.extend(code)
        level += 1
    rtl_code.extend(all_tree_code)
    rtl_code.append("assign data_o = stage"+str(level)+"[0]>="+str(threshold)+";\n\n")
    rtl_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)
    
