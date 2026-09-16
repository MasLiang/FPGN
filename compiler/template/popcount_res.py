import math
from popcount_2lut import *
from popcount_3lut import *

def lut_reduction(width, level, idx=0):
    lut6_count = math.ceil(width/6)
    width_bits = math.ceil(math.log2(width+1))
    def_code = []
    rtl_code = []
    def_code.append("wire [5:0] lut6_data_i_level"+str(level)+"_idx"+str(idx)+"[0:"+str(lut6_count)+"-1];\n")
    def_code.append("wire [2:0] lut6_data_o_level"+str(level)+"_idx"+str(idx)+"[0:"+str(lut6_count)+"-1];\n")
    for i in range(lut6_count):
        if level==0:
            if i<lut6_count-1:
                rtl_code.append("assign lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"] = data_i["+str(i*6+5)+":"+str(i*6)+"];\n")
            else:                                                     
                rtl_code.append("assign lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"] = data_i["+str(width-1)+":"+str(i*6)+"];\n")
        else:
            if i<lut6_count-1:
                rtl_code.append("assign lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"] = data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i*6+5)+":"+str(i*6)+"];\n")
            else:
                rtl_code.append("assign lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"] = data_i_level"+str(level)+"_idx"+str(idx)+"["+str(width-1)+":"+str(i*6)+"];\n")

    for i in range(lut6_count):
        if (i == lut6_count-1 and width % 6 == 1):
            rtl_code.append("assign lut6_data_o_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"] = {2'b0, lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"][0]};\n")
        elif (i == lut6_count-1 and width % 6 == 2):
            rtl_code.append("popcount_2lut u_2lut_level"+str(level)+"_idx"+str(idx)+"_"+str(i)+"(.bits({1'b0, lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"][1:0]}), .count(lut6_data_o_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"][1:0]));\n")
            rtl_code.append("assign lut6_data_o_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"][2] = 1'b0;\n")
        elif (i == lut6_count-1 and width % 6 == 3):
            rtl_code.append("popcount_2lut u_2lut_level"+str(level)+"_idx"+str(idx)+"_"+str(i)+"(.bits(lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"][2:0]), .count(lut6_data_o_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"][1:0]));\n")
            rtl_code.append("assign lut6_data_o_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"][2] = 1'b0;\n")
        elif (i == lut6_count-1 and width % 6 == 4):
            rtl_code.append("popcount_3lut u_3lut_level"+str(level)+"_idx"+str(idx)+"_"+str(i)+"(.bits({2'b0, lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"][3:0]}), .count(lut6_data_o_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"]));\n")
        elif (i == lut6_count-1 and width % 6 == 5):
            rtl_code.append("popcount_3lut u_3lut_level"+str(level)+"_idx"+str(idx)+"_"+str(i)+"(.bits({1'b0, lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"][4:0]}), .count(lut6_data_o_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"]));\n")
        else:
            rtl_code.append("popcount_3lut u_3lut_level"+str(level)+"_idx"+str(idx)+"_"+str(i)+"(.bits(lut6_data_i_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"]), .count(lut6_data_o_level"+str(level)+"_idx"+str(idx)+"["+str(i)+"]));\n")

    return def_code, rtl_code, lut6_count

def generate_popcount_res(rtl_path, module_name, width):
    width_bits = math.ceil(math.log2(width+1))
    
    rtl_code = []
    def_code = []
    def_code.append("`timescale 1ns/1ps\n")
    def_code.append("module "+str(module_name)+"(\n")
    def_code.append("    input  clk                              ,\n")
    def_code.append("    input  rst_n                            ,\n")
    def_code.append("    input  data_i_vld                       ,\n")
    def_code.append("    input  ["+str(width)+"-1:0] data_i      ,\n")
    def_code.append("    output ["+str(width_bits)+"-1:0] data_o  \n")
    def_code.append(");\n")
    def_code.append("\n")
    d_code, r_code, lut6_count_level0 = lut_reduction(width, 0)
    def_code = def_code + d_code
    def_code.append("\n")
    rtl_code = rtl_code + r_code
    rtl_code.append("\n")

    generate_popcount_2lut(rtl_path)
    generate_popcount_3lut(rtl_path)

    for i in range(3):
        def_code.append("wire   ["+str(lut6_count_level0-1)+":0] data_i_level1_idx"+str(i)+";\n")
        for j in range(lut6_count_level0):
            rtl_code.append("assign data_i_level1_idx"+str(i)+"["+str(j)+"] = lut6_data_o_level0_idx0["+str(j)+"]["+str(i)+"];\n")
        d_code, r_code, lut6_count_level1 = lut_reduction(lut6_count_level0, 1, i)
        def_code = def_code + d_code
        def_code.append("\n")
        rtl_code = rtl_code + r_code
        rtl_code.append("\n")
        def_code.append("reg [2:0] lut6_results_reg_idx"+str(i)+"[0:"+str(lut6_count_level1-1)+"];\n")
        for j in range(lut6_count_level1):
            rtl_code.append("always @(posedge clk or negedge rst_n) begin\n")
            rtl_code.append("    if (!rst_n) \n")
            rtl_code.append("        lut6_results_reg_idx"+str(i)+"["+str(j)+"] <= 3'b0;\n")
            rtl_code.append("    else if (data_i_vld) \n")
            rtl_code.append("        lut6_results_reg_idx"+str(i)+"["+str(j)+"] <= lut6_data_o_level1_idx"+str(i)+"["+str(j)+"];\n")
            rtl_code.append("end\n\n")
            if i==0:
                rtl_code.append("assign adder_tree_data_i["+str(i*lut6_count_level1+j)+"] = {2'b0, lut6_results_reg_idx"+str(i)+"["+str(j)+"]};\n")
            elif i==1:
                rtl_code.append("assign adder_tree_data_i["+str(i*lut6_count_level1+j)+"] = {1'b0, lut6_results_reg_idx"+str(i)+"["+str(j)+"], 1'b0};\n")
            elif i==2:
                rtl_code.append("assign adder_tree_data_i["+str(i*lut6_count_level1+j)+"] = {lut6_results_reg_idx"+str(i)+"["+str(j)+"], 2'b0};\n")
    
    def_code.append("wire [4:0] adder_tree_data_i[0:"+str(lut6_count_level1*3-1)+"];\n")

    def generate_adder_tree(elements, level=0, width_bits=7):
        if elements <= 1:
            return elements, []
        
        next_level_elements = (elements + 1) // 2
        code = []
        
        if level == 0:
            code.append("    wire ["+str(width_bits)+"-1:0] stage"+str(level+1)+" [0:"+str(next_level_elements-1)+"];\n")
            for i in range(elements//2):
                code.append("assign stage"+str(level+1)+"["+str(i)+"] = adder_tree_data_i["+str(2*i)+"] + adder_tree_data_i["+str(2*i+1)+"];\n")
            if elements % 2 == 1:
                code.append("assign stage"+str(level+1)+"["+str(elements//2)+"] = {"+str(width_bits-4)+"'b0, adder_tree_data_i["+str(elements-1)+"]};\n")
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
    current_elements = lut6_count_level1 * 3
    level = 0
    
    while current_elements > 1:
        current_elements, code = generate_adder_tree(current_elements, level, width_bits)
        all_tree_code.extend(code)
        level += 1
    rtl_code.extend(all_tree_code)
    rtl_code.append("assign data_o = stage"+str(level)+"[0];\n")
    rtl_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in def_code+["\n"]+rtl_code:
            f.write(i)
    
    # return level-reg_level-1
