import math

def generate_buffer_to_fc(rtl_path,
                               module_name, 
                               channel, 
                               col, 
                               row, 
                               in_kernel_num_row, 
                               in_kernel_num_col, 
                               lut_size, 
                               lut_num):
    """
    Generate a Verilog module for a buffer that converts convolution data to fully connected layer data.
    Parameters:
        module_name (str): Name of the module.
        channel (int): Number of channels.
        col (int): Number of columns in the input data.
        row (int): Number of rows in the input data.
        in_kernel_num_row (int): Number of rows in the input kernel.
        in_kernel_num_col (int): Number of columns in the input kernel.
        lut_size (int): Size of the LUT.
        lut_num (int): Number of LUTs.
    """
    rtl_code = []

    rtl_code.append("`timescale 1ns/1ps\n")
    rtl_code.append("module "+module_name+"(\n")
    rtl_code.append("    input clk,\n")
    rtl_code.append("    input rst_n,\n")
    rtl_code.append("    input data_i_vld,\n")
    rtl_code.append("    input["+str(channel*in_kernel_num_row*in_kernel_num_col)+"-1:0] data_i,\n")
    rtl_code.append("    output data_o_vld,\n")
    rtl_code.append("    output ["+str(lut_size*lut_num)+"-1:0] data_o\n")
    rtl_code.append(");\n")
    rtl_code.append("\n")
    rtl_code.append("reg ["+str(channel)+"-1:0] mem_blocks [0:"+str(row)+"-1][0:"+str(col)+"-1];\n")
    rtl_code.append("reg ["+str(math.ceil(math.log2(row+1)))+"-1:0] write_row_addr;\n")
    rtl_code.append("reg ["+str(math.ceil(math.log2(col+1)))+"-1:0] write_col_addr;\n")
    rtl_code.append("wire ["+str(col*row*channel)+"-1:0] data_o_pre;\n")
    rtl_code.append("\n")
    rtl_code.append("always @(posedge clk or negedge rst_n) \n")
    rtl_code.append("begin\n")
    rtl_code.append("    if (!rst_n) \n")
    rtl_code.append("    begin\n")
    rtl_code.append("        write_row_addr <= {"+str(math.ceil(math.log2(row+1)))+"{1'b0}};\n")
    rtl_code.append("        write_col_addr <= {"+str(math.ceil(math.log2(col+1)))+"{1'b0}};\n")
    rtl_code.append("    end \n")
    rtl_code.append("    else \n")
    rtl_code.append("    if (data_i_vld) \n")
    rtl_code.append("    begin\n")
    rtl_code.append("        if (write_col_addr >= "+str(col)+" - "+str(in_kernel_num_col)+") \n")
    rtl_code.append("        begin\n")
    rtl_code.append("            write_col_addr <= {"+str(math.ceil(math.log(col,2)))+"{1'b0}};\n")
    rtl_code.append("            if (write_row_addr >= "+str(row)+" - "+str(in_kernel_num_row)+") \n")
    rtl_code.append("            begin\n")
    rtl_code.append("                write_row_addr <= {"+str(math.ceil(math.log(row,2)))+"{1'b0}};\n")
    rtl_code.append("            end \n")
    rtl_code.append("            else \n")
    rtl_code.append("            begin\n")
    rtl_code.append("                write_row_addr <= write_row_addr + "+str(in_kernel_num_row)+";\n")
    rtl_code.append("            end\n")
    rtl_code.append("        end \n")
    rtl_code.append("        else \n")
    rtl_code.append("        begin\n")
    rtl_code.append("            write_col_addr <= write_col_addr + "+str(in_kernel_num_col)+";\n")
    rtl_code.append("        end\n")
    rtl_code.append("    end\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")
    rtl_code.append("integer wr_r, wr_c;\n")
    rtl_code.append("always @(posedge clk) \n")
    rtl_code.append("begin\n")
    rtl_code.append("    if (data_i_vld) \n")
    rtl_code.append("    begin\n")
    rtl_code.append("        for (wr_r = 0; wr_r < "+str(in_kernel_num_row)+"; wr_r = wr_r + 1) \n")
    rtl_code.append("        begin : write_rows\n")
    rtl_code.append("            for (wr_c = 0; wr_c < "+str(in_kernel_num_col)+"; wr_c = wr_c + 1) \n")
    rtl_code.append("            begin : write_cols\n")
    rtl_code.append("                mem_blocks[write_row_addr+wr_r][write_col_addr+wr_c] <= data_i[(wr_r*"+str(in_kernel_num_col)+"+wr_c)*"+str(channel)+"+:"+str(channel)+"];\n")
    rtl_code.append("            end\n")
    rtl_code.append("        end\n")
    rtl_code.append("    end\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")
    rtl_code.append("reg can_read;\n")
    rtl_code.append("wire can_read_pre = data_i_vld && (write_col_addr >= "+str(col-in_kernel_num_col)+") && (write_row_addr >= "+str(row-in_kernel_num_row)+");\n")
    rtl_code.append("always @(posedge clk or negedge rst_n) \n")
    rtl_code.append("begin\n")
    rtl_code.append("    if (!rst_n) \n")
    rtl_code.append("    begin\n")
    rtl_code.append("        can_read <= 1'b0;\n")
    rtl_code.append("    end \n")
    rtl_code.append("    else if(data_o_vld)\n")
    rtl_code.append("    begin\n")
    rtl_code.append("        can_read <= 1'b0;\n")
    rtl_code.append("    end\n")
    rtl_code.append("    else if(can_read_pre)\n")
    rtl_code.append("    begin\n")
    rtl_code.append("        can_read <= 1'b1;\n")
    rtl_code.append("    end\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")
    rtl_code.append("assign data_o_vld = can_read;\n")
    rtl_code.append("\n")
    for channel_idx in range (channel):
        for col_idx in range(col):
            for row_idx in range(row):
                if channel_idx== 0 and col_idx == 0 and row_idx == 0:
                    rtl_code.append("assign data_o_pre = {mem_blocks["+str(row_idx)+"]["+str(col_idx)+"]["+str(channel_idx)+"],\n")
                elif channel_idx == channel-1 and col_idx == col-1 and row_idx == row-1:
                    rtl_code.append("                     mem_blocks["+str(row_idx)+"]["+str(col_idx)+"]["+str(channel_idx)+"]};\n")
                else:
                    rtl_code.append("                     mem_blocks["+str(row_idx)+"]["+str(col_idx)+"]["+str(channel_idx)+"],\n")
    repeat_num = (lut_size*lut_num) // (row*col*channel)
    last_num = (lut_size*lut_num) % (row*col*channel)
    rtl_code.append("assign data_o = {")
    for i in range(repeat_num):
        if i==repeat_num and last_num==0:
            rtl_code.append("                data_o_pre};\n")
        else:
            rtl_code.append("                data_o_pre, \n")
    if last_num>0:
        rtl_code.append("                data_o_pre["+str(last_num-1)+":0]};\n")

    rtl_code.append("\n")
    rtl_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in rtl_code:
            f.write(i)