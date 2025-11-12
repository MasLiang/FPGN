import math
def generate_res_max_pooling(rtl_path, module_name, stride_size, in_bit_width):
    rtl_code = []
    rtl_code.append("module "+module_name+" (\n")
    rtl_code.append("    input       clk,\n")
    rtl_code.append("    input       rst_n,\n")
    rtl_code.append("    input       data_i_vld,\n")
    rtl_code.append("    input       ["+str(stride_size*stride_size*in_bit_width)+"-1:0] data_i,\n")
    rtl_code.append("    output      reg data_o_vld,\n")
    rtl_code.append("    output      reg ["+str(in_bit_width)+"-1:0] data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    rtl_code.append("always @(posedge clk or negedge rst_n)\n")
    rtl_code.append("begin\n")
    rtl_code.append("    if (!rst_n) \n")
    rtl_code.append("        data_o_vld     <=      1'b0;\n")
    rtl_code.append("    else \n")
    rtl_code.append("        data_o_vld     <=      data_i_vld;\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")
    rtl_code.append("wire ["+str(in_bit_width)+"-1:0] data_i_sep[0:"+str(stride_size*stride_size-1)+"];\n")
    rtl_code.append("wire ["+str(in_bit_width+math.ceil(math.log2(stride_size*stride_size)))+"-1:0] data_i_sum;\n")
    for j in range(stride_size):
        for k in range(stride_size):
            rtl_code.append("assign data_i_sep["+str(j*stride_size+k)+"] = data_i["+str((j*stride_size+k+1)*in_bit_width-1)+":"+str((j*stride_size+k)*in_bit_width)+"];\n")
    for j in range(stride_size*stride_size):
        if j==0 and stride_size*stride_size==1:
            rtl_code.append("assign data_i_sum = data_i_sep["+str(j)+"];\n")
        elif j==0:
            rtl_code.append("assign data_i_sum = data_i_sep["+str(j)+"]+\n")
        elif j==stride_size*stride_size-1:
            rtl_code.append("                        data_i_sep[" + str(j) + "];\n")
        else:
            rtl_code.append("                        data_i_sep[" + str(j) + "]+\n")
    rtl_code.append("always @(posedge clk or negedge rst_n)\n")
    rtl_code.append("begin\n")
    rtl_code.append("    if (!rst_n) \n")
    rtl_code.append("        data_o     <=      {"+str(in_bit_width)+"{1'b0}};\n")
    rtl_code.append("    else if (data_i_vld) \n")
    if in_bit_width==1:
        rtl_code.append("        data_o     <=      data_i_sum["+str(in_bit_width+math.ceil(math.log2(stride_size*stride_size))-1)+"];\n")
    else:
        rtl_code.append("        data_o     <=      data_i_sum["+str(in_bit_width+math.ceil(math.log2(stride_size*stride_size)))+"-1:"+str(math.ceil(math.log2(stride_size*stride_size)))+"];\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")

    rtl_code.append("endmodule\n")  

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)
    