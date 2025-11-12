import math

def generate_fc_in_conn(rtl_path,
                        module_name, 
                        in_num, 
                        out_num):
    """
    Generate the RTL code for the fully connected layer input connection.
    Parameter:
        rtl_path: The path to the RTL output directory.
        module_name: The name of the module.
        in_num: The number of input bits.
        out_num: The number of output bits.
    """
    rtl_code = []
    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+module_name+"(\n")
    rtl_code.append("    input ["+str(in_num)+"-1:0] data_i,\n")
    rtl_code.append("    output ["+str(out_num)+"-1:0] data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    repeat_num = math.ceil(out_num/in_num)
    left_num = out_num
    for i in range(repeat_num):
        if repeat_num==1:
            rtl_code.append("assign data_o = data_i["+str(out_num)+"-1:0];\n")
        else:
            if i==0:
                rtl_code.append("assign data_o = {data_i,\n")
            elif i<repeat_num-1:
                rtl_code.append("                 data_i,\n")
            else:
                rtl_code.append("                 data_i["+str(left_num-1)+":0]};\n")
        left_num -= in_num
    rtl_code.append("\n")
    rtl_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)