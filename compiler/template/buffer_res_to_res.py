import math

def generate_buffer_res_to_res(rtl_path,
                               module_name, 
                               kernel_size, 
                               channel, 
                               stride_size, 
                               padding_size, 
                               col, 
                               row, 
                               in_bit_width,
                               in_kernel_num_row, 
                               in_kernel_num_col, 
                               out_kernel_num_row, 
                               out_kernel_num_col):
    """    
    Generate a Verilog module for a buffer to convolution layer.
    This module handles the conversion of buffered input data to a format suitable for convolution operations.
    Parameters:
        rtl_path (str): Path to the RTL directory.
        module_name (str): Name of the buffer to convolution module.
        kernel_size (int): Size of the convolution kernel.
        channel (int): Number of input channels.
        stride_size (int): Stride size for the convolution operation.
        padding_size (int): Padding size applied to the input data.
        col (int): Number of columns in the input data.
        row (int): Number of rows in the input data.
        in_bit_width (int): Bit width of the input data.
        in_kernel_num_row (int): Number of kernels in row in the input.
        in_kernel_num_col (int): Number of kernels in columns in the input.
        out_kernel_num_row (int): Number of kernels in row in the output.
        out_kernel_num_col (int): Number of kernels in columns in the output.
    """  
    if stride_size < kernel_size:
        safety_margin = stride_size 
    else:
        safety_margin = 0
    mem_row = min(max(in_kernel_num_row, (kernel_size+(out_kernel_num_row-1)*stride_size))+in_kernel_num_row, row)
    mem_col = col
    mem_row_bit = math.ceil(math.log(mem_row, 2))
    if mem_row_bit==0:
        mem_row_bit = 1
    mem_col_bit = math.ceil(math.log(mem_col, 2))
    if mem_col_bit==0:
        mem_col_bit = 1
    col_bit = math.ceil(math.log(col+padding_size, 2))
    if col_bit==0:
        col_bit = 1
    row_bit = math.ceil(math.log(row+padding_size, 2))
    if row_bit==0:
        row_bit = 1
    required_read_row = (out_kernel_num_row-1) * stride_size + kernel_size
    required_read_col = (out_kernel_num_col-1) * stride_size + kernel_size
    in_kernel_num_col_bit = math.ceil(math.log(in_kernel_num_col, 2))
    if in_kernel_num_col_bit==0:
        in_kernel_num_col_bit = 1

    shift_per_time = out_kernel_num_col*stride_size
    shift_times = col//shift_per_time

    rtl_code = []
    port_code = []
    decl_code = []
    assign_code = []
    port_code.append("`timescale 1ns/1ps\n")
    port_code.append("module "+module_name+"(\n")
    port_code.append("    input clk,\n")
    port_code.append("    input rst_n,\n")
    port_code.append("    input data_i_vld,\n")
    port_code.append("    input ["+str(channel*in_kernel_num_row*in_kernel_num_col)+"-1:0] data_i,\n")
    port_code.append("    input ["+str(channel*in_kernel_num_row*in_kernel_num_col*in_bit_width)+"-1:0] res_i,\n")
    port_code.append("    output data_o_vld,\n")
    port_code.append("    output wire ["+str(channel*kernel_size*kernel_size*out_kernel_num_row*out_kernel_num_col)+"-1:0] data_o,\n")
    port_code.append("    output wire ["+str(channel*stride_size*stride_size*out_kernel_num_row*out_kernel_num_col*in_bit_width)+"-1:0] res_o\n")
    port_code.append(");\n")
    port_code.append("\n")
    decl_code.append('(* DONT_TOUCH = "yes" *)\n')
    decl_code.append("reg ["+str(channel*(1+in_bit_width))+"-1:0] mem_blocks [0:"+str(mem_row)+"-1][0:"+str(mem_col)+"-1];\n")
    decl_code.append("reg ["+str(channel*(1+in_bit_width))+"-1:0] mem_blocks_padding [0:"+str(mem_row)+"-1][0:1];\n")
    if shift_times>1:
        decl_code.append("wire shift_en [0:"+str(kernel_size+(out_kernel_num_row-1)*stride_size)+"-1];\n")
    decl_code.append("reg [("+str(mem_row_bit)+")-1:0] write_row_addr;\n")
    decl_code.append("reg ["+str(col_bit)+"-1:0] write_col_addr;\n")
    decl_code.append("(* MAX_FANOUT = 100 *)\n")
    decl_code.append("reg ["+str(row_bit)+"-1:0] total_write_row;\n")
    decl_code.append("(* MAX_FANOUT = 100 *)\n")
    decl_code.append("reg ["+str(row_bit)+"-1:0] current_read_row;\n")
    decl_code.append("(* MAX_FANOUT = 100 *)\n")
    decl_code.append("reg ["+str(col_bit)+"-1:0] current_read_col;\n")
    # decl_code.append("wire start_padding_row_flg;\n")
    decl_code.append("wire end_padding_row_flg;\n")
    decl_code.append("wire ["+str(row_bit)+"-1:0] valid_read_row;\n")
    rtl_code.append("\n")
    rtl_code.append("always @(posedge clk or negedge rst_n) \n")
    rtl_code.append("begin\n")
    rtl_code.append("    if (!rst_n) \n")
    rtl_code.append("    begin\n")
    rtl_code.append("        write_row_addr <= "+str(padding_size//2)+";\n")
    rtl_code.append("        write_col_addr <= "+str(mem_col_bit)+"'b0;\n")
    rtl_code.append("    end \n")
    rtl_code.append("    else if (data_i_vld) \n")
    rtl_code.append("    begin\n")
    rtl_code.append("        if (write_col_addr >= "+str(mem_col-in_kernel_num_col)+") \n")
    rtl_code.append("        begin\n")
    rtl_code.append("            write_col_addr <= write_col_addr - "+str(mem_col-in_kernel_num_col)+";\n")
    rtl_code.append("            if (total_write_row == "+str(row-in_kernel_num_row+padding_size//2)+")\n")
    rtl_code.append("            begin\n")
    rtl_code.append("                write_row_addr <= "+str(padding_size//2)+"'b0;\n")
    rtl_code.append("            end\n")
    rtl_code.append("            else if (((write_row_addr >= "+str(mem_row-in_kernel_num_row)+")&(!read_row_add_en)) | ((write_row_addr >= "+str(mem_row-in_kernel_num_row-stride_size)+")&(read_row_add_en)))\n")
    rtl_code.append("            begin\n")
    rtl_code.append("                if (read_row_add_en)\n")
    rtl_code.append("                   write_row_addr <= write_row_addr - "+str(mem_row-in_kernel_num_row-stride_size)+";\n")
    rtl_code.append("                else\n")
    rtl_code.append("                   write_row_addr <= write_row_addr - "+str(mem_row-in_kernel_num_row)+";\n")
    rtl_code.append("            end \n")
    rtl_code.append("            else \n")
    rtl_code.append("            begin\n")
    rtl_code.append("                if (read_row_add_en)\n")
    rtl_code.append("                   write_row_addr <= write_row_addr + "+str(in_kernel_num_row-stride_size)+";\n")
    rtl_code.append("                else\n")
    rtl_code.append("                   write_row_addr <= write_row_addr + "+str(in_kernel_num_row)+";\n")
    rtl_code.append("            end\n")
    rtl_code.append("        end \n")
    rtl_code.append("        else \n")
    rtl_code.append("        begin\n")
    rtl_code.append("            write_col_addr <= write_col_addr + "+str(in_kernel_num_col)+";\n")
    rtl_code.append("            if(read_row_add_en)\n")
    rtl_code.append("                write_row_addr <= write_row_addr - "+str(stride_size)+";\n")
    rtl_code.append("        end\n")
    rtl_code.append("    end\n")
    rtl_code.append("    else \n")
    rtl_code.append("    begin\n")
    rtl_code.append("        if(read_row_add_en)\n")
    rtl_code.append("            write_row_addr <= write_row_addr - "+str(stride_size)+";\n")
    rtl_code.append("    end\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")
    row_addr_start_lst = []
    row_addr_end_lst = []
    col_addr_start_lst = []
    col_addr_end_lst = []
    for mr in range(0, row, in_kernel_num_row):
        start_addr = mr%mem_row
        row_addr_start_lst.append(start_addr)
    row_addr_start_lst = list(set(row_addr_start_lst))
    for start_addr in row_addr_start_lst:
        end_addr = start_addr + in_kernel_num_row-1
        if end_addr>=mem_row:
            end_addr = end_addr - mem_row
        row_addr_end_lst.append(end_addr)
    for mc in range(0, col, in_kernel_num_col):
        start_addr = mc
        end_addr = mc + in_kernel_num_col -1
        col_addr_start_lst.append(start_addr)
        col_addr_end_lst.append(end_addr)

    for mr in range(mem_row):
        for mc in range(mem_col):
            wrt_en_addr_lst = []
            for row_addr_idx in range(len(row_addr_start_lst)):
                row_start_addr = row_addr_start_lst[row_addr_idx]
                row_end_addr = row_addr_end_lst[row_addr_idx]
                if row_start_addr<=row_end_addr:
                    if mr>=row_start_addr and mr<=row_end_addr:
                        for col_addr_idx in range(len(col_addr_start_lst)):
                            col_start_addr = col_addr_start_lst[col_addr_idx]
                            col_end_addr = col_addr_end_lst[col_addr_idx]
                            if mc>=col_start_addr and mc<=col_end_addr:
                                row_bias = mr - row_start_addr
                                col_bias = mc - col_start_addr
                                wrt_en_addr_lst.append((row_start_addr, col_start_addr, row_bias, col_bias))
                            else:
                                continue
                    else:
                        continue
                else:
                    if mr>=row_start_addr or mr<=row_end_addr:
                        for col_addr_idx in range(len(col_addr_start_lst)):
                            col_start_addr = col_addr_start_lst[col_addr_idx]
                            col_end_addr = col_addr_end_lst[col_addr_idx]
                            if mc>=col_start_addr and mc<=col_end_addr:
                                row_bias = mr - row_start_addr
                                col_bias = mc - col_start_addr
                                if row_bias<0:
                                    row_bias = row_bias + mem_row
                                if col_bias<0:
                                    col_bias = col_bias + mem_col
                                wrt_en_addr_lst.append((row_start_addr, col_start_addr, row_bias, col_bias))
                            else:
                                continue
                    else:
                        continue
 
                    
            if mr<padding_size//2:
                rtl_code.append("always @(posedge clk or negedge rst_n) \n")
            else:
                rtl_code.append("always @(posedge clk) \n")
            rtl_code.append("begin\n")
            if mr<padding_size//2:
                rtl_code.append("    if (rst_n==1'b0) \n")
                rtl_code.append("    begin\n")
                rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= "+str(channel)+"'b0; \n")
                rtl_code.append("    end\n")
            for wrt_en_idx in range(len(wrt_en_addr_lst)):
                row_start_addr, col_start_addr, row_bias, col_bias = wrt_en_addr_lst[wrt_en_idx]
                decl_code.append("wire mem_blocks_wen_"+str(mr)+"_"+str(mc)+"_"+str(row_bias)+"_"+str(col_bias)+";\n")
                assign_code.append("assign mem_blocks_wen_"+str(mr)+"_"+str(mc)+"_"+str(row_bias)+"_"+str(col_bias)+" = write_row_addr=="+str(row_start_addr)+" && write_col_addr=="+str(col_start_addr)+";\n")
                if wrt_en_idx==0:
                    if mr<padding_size//2:
                        rtl_code.append("    else if (data_i_vld && mem_blocks_wen_"+str(mr)+"_"+str(mc)+"_"+str(row_bias)+"_"+str(col_bias)+") \n")
                    else:
                        rtl_code.append("    if (data_i_vld && mem_blocks_wen_"+str(mr)+"_"+str(mc)+"_"+str(row_bias)+"_"+str(col_bias)+") \n")
                else:
                    rtl_code.append("    else if (data_i_vld && mem_blocks_wen_"+str(mr)+"_"+str(mc)+"_"+str(row_bias)+"_"+str(col_bias)+") \n")
                rtl_code.append("    begin\n")
                rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= {res_i["+str((row_bias*in_kernel_num_col+col_bias+1)*channel*in_bit_width)+"-1:"+str((row_bias*in_kernel_num_col+col_bias)*channel*in_bit_width)+"], data_i["+str((row_bias*in_kernel_num_col+col_bias+1)*channel)+"-1:"+str((row_bias*in_kernel_num_col+col_bias)*channel)+"]}; \n")
                rtl_code.append("    end\n")
            if shift_times>1:
                if mr>kernel_size+(out_kernel_num_row-1)*stride_size-1-padding_size//2:
                    rtl_code.append("    else if (shift_en["+str(mr)+"]&read_row_add_en) \n")
                else:
                    rtl_code.append("    else if (read_row_add_en) \n")
                rtl_code.append("    begin\n")
                final_shift = shift_per_time*(shift_times-1)
                new_col = mc - final_shift
                if new_col==0:
                    if mr+stride_size>=mem_row:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= 0;\n")
                    else:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= mem_blocks_padding["+str(mr+stride_size)+"][0];\n")
                elif new_col==-1:
                    if mr+stride_size>=mem_row:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= 0;\n")
                    else:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= mem_blocks_padding["+str(mr+stride_size)+"][1];\n")
                elif new_col< 0:
                    new_col = new_col + col+1
                    if mr+stride_size>=mem_row:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= 0;\n")
                    else:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= mem_blocks["+str(mr+stride_size)+"]["+str(new_col)+"];\n")
                else:
                    if mr+stride_size>=mem_row:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= 0;\n")
                    else:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= mem_blocks["+str(mr+stride_size)+"]["+str(new_col)+"];\n")
                rtl_code.append("    end\n")
                if mr>kernel_size+(out_kernel_num_row-1)*stride_size-1-padding_size//2:
                    rtl_code.append("    else if(shift_en["+str(mr)+"])\n")
                    rtl_code.append("    begin\n")
                    new_col = mc+shift_per_time
                    if new_col==col:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= mem_blocks_padding["+str(mr)+"][1];\n")
                    elif new_col==col+1:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= mem_blocks_padding["+str(mr)+"][0];\n")
                    elif new_col> col:
                        new_col = new_col - col-2
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= mem_blocks["+str(mr)+"]["+str(new_col)+"];\n")
                    else:
                        rtl_code.append("        mem_blocks["+str(mr)+"]["+str(mc)+"] <= mem_blocks["+str(mr)+"]["+str(new_col)+"];\n")
                    rtl_code.append("    end\n")
            rtl_code.append("end\n")
            rtl_code.append("\n")
    for mr in range(mem_row):
        for mc in range(2):
            rtl_code.append("always @(posedge clk or negedge rst_n) \n")
            rtl_code.append("begin\n")
            rtl_code.append("    if (rst_n==1'b0) \n")
            rtl_code.append("    begin\n")
            rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= "+str(channel)+"'b0; \n")
            rtl_code.append("    end\n")
            rtl_code.append("    else if (write_row_addr == "+str(mr)+") \n")
            rtl_code.append("    begin\n")
            rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= "+str(channel)+"'b0; \n")
            rtl_code.append("    end\n")
            rtl_code.append("    else if (end_padding_row_flg) \n")
            rtl_code.append("    begin\n")
            rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= "+str(channel)+"'b0;\n")
            rtl_code.append("    end\n")
            if shift_times>1:
                rtl_code.append("    else if (shift_en["+str(mr)+"]&read_row_add_en) \n")
                rtl_code.append("    begin\n")
                rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= "+str(channel)+"'b0;\n")
                rtl_code.append("    end\n")
                rtl_code.append("    else if (shift_en["+str(mr)+"]) \n")
                rtl_code.append("    begin\n")
                shift_col = stride_size*out_kernel_num_col
                if mc==0:
                    if shift_col==col:
                        rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= mem_blocks_padding["+str(mr)+"][1]; \n")
                    elif shift_col==col+1:
                        rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= mem_blocks_padding["+str(mr)+"][0]; \n")
                    elif shift_col> col:
                        new_col = shift_col - col -2
                        rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= mem_blocks["+str(mr)+"]["+str(new_col)+"];\n")
                    else:
                        rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= mem_blocks["+str(mr)+"]["+str(shift_col)+"];\n")
                else:
                    if shift_col==1:
                        rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= mem_blocks_padding["+str(mr)+"][0]; \n")
                    elif shift_col==col+2:
                        rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= mem_blocks_padding["+str(mr)+"][1]; \n")
                    else:
                        new_col = shift_col -2
                        rtl_code.append("        mem_blocks_padding["+str(mr)+"]["+str(mc)+"] <= mem_blocks["+str(mr)+"]["+str(new_col)+"];\n")
                rtl_code.append("    end\n")
            rtl_code.append("end\n")
            rtl_code.append("\n")
    mem_blocks_wire = [[[[] for _ in range(channel*(1+in_bit_width))] for _ in range(col+padding_size)] for _ in range(row+padding_size)]
    for r in range(row):
        for c in range(col):
            for ch in range(channel*2):
                if ch<channel:
                    mem_blocks_wire[r+(padding_size//2)][c+(padding_size//2)][ch] = "mem_blocks["+str(r%mem_row)+"]["+str(c)+"]["+str(ch)+"]"
                else:
                    mem_blocks_wire[r+(padding_size//2)][c+(padding_size//2)][ch] = "mem_blocks["+str(r%mem_row)+"]["+str(c)+"]["+str(channel+(ch-channel)*in_bit_width)+":+"+str(in_bit_width)+"]"
    for c in range(col+padding_size):
        for ch in range(channel*2):
            if ch<channel:
                mem_blocks_wire[0][c][ch] = "1'b0"
                mem_blocks_wire[row+1][c][ch] = "1'b0"
            else:
                mem_blocks_wire[0][c][ch] = str(in_bit_width)+"'b0"
                mem_blocks_wire[row+1][c][ch] = str(in_bit_width)+"'b0"
    for r in range(1,row+padding_size-1):
        for ch in range(channel*2):
            if ch<channel:
                mem_blocks_wire[r][0][ch] = "mem_blocks_padding["+str(r%(mem_row))+"][0]["+str(ch)+"]"
                mem_blocks_wire[r][col+1][ch] = "mem_blocks_padding["+str(r%(mem_row))+"][1]["+str(ch)+"]"
            else:
                mem_blocks_wire[r][0][ch] = "mem_blocks_padding["+str(r%(mem_row))+"][0]["+str(channel+(ch-channel)*in_bit_width)+":+"+str(in_bit_width)+"]"
                mem_blocks_wire[r][col+1][ch] = "mem_blocks_padding["+str(r%(mem_row))+"][1]["+str(channel+(ch-channel)*in_bit_width)+":+"+str(in_bit_width)+"]"
    decl_code.append("wire write_col_last_flg;\n")
    rtl_code.append("\n")
    rtl_code.append("assign write_col_last_flg = (write_col_addr >= "+str(mem_col-in_kernel_num_col)+") ? 1'b1 : 1'b0;\n")
    rtl_code.append("always @(posedge clk or negedge rst_n) \n")
    rtl_code.append("begin\n")
    rtl_code.append("    if (!rst_n) begin\n")
    rtl_code.append("        total_write_row <= "+str(padding_size//2)+";\n")
    rtl_code.append("    end \n")
    rtl_code.append("    else if (data_i_vld) \n")
    rtl_code.append("    begin\n")
    rtl_code.append("        if (write_col_last_flg) \n")
    rtl_code.append("        begin\n")
    rtl_code.append("            if (total_write_row > "+str(row-in_kernel_num_row+padding_size//2)+")\n")
    rtl_code.append("            begin\n")
    rtl_code.append("                total_write_row <= "+str(padding_size//2+in_kernel_num_row)+";\n")
    rtl_code.append("            end \n")
    rtl_code.append("            else \n")
    rtl_code.append("            begin\n")
    rtl_code.append("                total_write_row <= total_write_row + "+str(in_kernel_num_row)+";\n")
    rtl_code.append("            end\n")
    rtl_code.append("        end\n")
    rtl_code.append("        else if (read_frame_done) \n")
    rtl_code.append("        begin\n")
    rtl_code.append("            total_write_row <= "+str(padding_size//2)+";\n")
    rtl_code.append("        end\n")
    rtl_code.append("    end\n")
    rtl_code.append("    else if (read_frame_done) \n")
    rtl_code.append("    begin\n")
    rtl_code.append("        total_write_row <= "+str(padding_size//2)+";\n")
    rtl_code.append("    end\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")
    for rd_r in range(out_kernel_num_row):
        for rd_c in range(out_kernel_num_col):
            for channel_idx in range(channel):
                for k_c in range(kernel_size):
                    for k_r in range(kernel_size):
                        read_row = rd_r+k_c
                        read_col = rd_c+k_r
                        if read_col==0:
                            rtl_code.append("assign data_o["+str((((rd_r*out_kernel_num_col+rd_c)*channel+channel_idx)*kernel_size+k_c)*kernel_size+k_r)+"] = mem_blocks_padding["+str(read_row)+"][0];\n")
                        elif read_col==col+1:
                            rtl_code.append("assign data_o["+str((((rd_r*out_kernel_num_col+rd_c)*channel+channel_idx)*kernel_size+k_c)*kernel_size+k_r)+"] = mem_blocks_padding["+str(read_row)+"][1];\n")
                        else:
                            rtl_code.append("assign data_o["+str((((rd_r*out_kernel_num_col+rd_c)*channel+channel_idx)*kernel_size+k_c)*kernel_size+k_r)+"] = mem_blocks["+str(read_row)+"]["+str(read_col)+"]["+str(channel_idx)+"];\n")
    for rd_r in range(out_kernel_num_row):
        for rd_c in range(out_kernel_num_col):
            for channel_idx in range(channel):
                for s_c in range(stride_size):
                    for s_r in range(stride_size):
                        read_row = rd_r+k_c
                        read_col = rd_c+k_r
                        if read_col==0:
                            rtl_code.append("assign res_o["+str(in_bit_width*((((rd_r*out_kernel_num_col+rd_c)*channel+channel_idx)*stride_size+s_c)*stride_size+s_r)+in_bit_width-1)+":"+str(in_bit_width*((((rd_r*out_kernel_num_col+rd_c)*channel+channel_idx)*stride_size+s_c)*stride_size+s_r))+"] = mem_blocks_padding["+str(read_row)+"][0]["+str(channel_idx+channel)+"];\n")
                        elif read_col==col+1:
                            rtl_code.append("assign res_o["+str(in_bit_width*((((rd_r*out_kernel_num_col+rd_c)*channel+channel_idx)*stride_size+s_c)*stride_size+s_r)+in_bit_width-1)+":"+str(in_bit_width*((((rd_r*out_kernel_num_col+rd_c)*channel+channel_idx)*stride_size+s_c)*stride_size+s_r))+"] = mem_blocks_padding["+str(read_row)+"][1]["+str(channel_idx+channel)+"];\n")
                        else:
                            rtl_code.append("assign res_o["+str(in_bit_width*((((rd_r*out_kernel_num_col+rd_c)*channel+channel_idx)*stride_size+s_c)*stride_size+s_r)+in_bit_width-1)+":"+str(in_bit_width*((((rd_r*out_kernel_num_col+rd_c)*channel+channel_idx)*stride_size+s_c)*stride_size+s_r))+"] = mem_blocks["+str(read_row)+"]["+str(read_col)+"]["+str(channel_idx+channel)+"];\n")

    rtl_code.append("\n")
    decl_code.append("wire read_row_add_en;\n")
    decl_code.append("wire read_frame_done;\n")
    rtl_code.append("assign read_row_add_en = current_read_col >= "+str(padding_size+col-out_kernel_num_col*stride_size-(kernel_size-1))+";\n")
    rtl_code.append("assign read_frame_done = read_row_add_en && (current_read_row >= "+str(padding_size+row-out_kernel_num_row*stride_size-(kernel_size-1))+");\n")
    rtl_code.append("always @(posedge clk or negedge rst_n)\n")
    rtl_code.append("begin\n")
    rtl_code.append("    if(!rst_n)\n")
    rtl_code.append("    begin\n")
    rtl_code.append("        current_read_col <= 0;\n")
    rtl_code.append("        current_read_row <= 0;\n")
    rtl_code.append("    end\n")
    rtl_code.append("    else if(data_o_vld)\n")
    rtl_code.append("    begin\n")
    rtl_code.append("        if(read_row_add_en)\n")
    rtl_code.append("        begin\n")
    rtl_code.append("            current_read_col <= 0;\n")
    rtl_code.append("            if (current_read_row >= "+str(padding_size+row-out_kernel_num_row*stride_size-(kernel_size-1))+") \n")
    rtl_code.append("            begin\n")
    rtl_code.append("                current_read_row <= 0;\n")
    rtl_code.append("            end \n")
    rtl_code.append("            else \n")
    rtl_code.append("            begin\n")
    rtl_code.append("                current_read_row <= current_read_row + "+str(out_kernel_num_row*stride_size)+";\n")
    rtl_code.append("            end\n")
    rtl_code.append("        end\n")
    rtl_code.append("        else\n")
    rtl_code.append("        begin\n")
    rtl_code.append("            current_read_col <= current_read_col + "+str(out_kernel_num_col*stride_size)+";\n")
    rtl_code.append("        end\n")
    rtl_code.append("    end\n")
    rtl_code.append("end\n")
    rtl_code.append("\n")
    cor_rows = [[] for _ in range(mem_row)]
    all_read_row = set()
    for kernel_idx in range(out_kernel_num_row):
        start_row = kernel_idx*stride_size
        for k_r in range(kernel_size):
            row_idx = start_row + k_r
            all_read_row.add(row_idx)
    all_read_row = list(all_read_row)
    all_read_row.sort()

    map_read_to_shift = [set() for _ in range(row+padding_size)]
    for start_row in range(0, row+padding_size-len(all_read_row), stride_size*out_kernel_num_row):
        for read_row_bias in all_read_row:
            read_row = start_row + read_row_bias
            if read_row<row+padding_size:
                map_read_to_shift[read_row].add(start_row)
            else:
                map_read_to_shift[read_row-(row+padding_size)].add(start_row)

    if shift_times>1:
        for m_r in range(mem_row):
            if m_r<kernel_size+(out_kernel_num_row-1)*stride_size:
                if m_r>kernel_size+(out_kernel_num_row-1)*stride_size-1-padding_size//2:
                    rtl_code.append("assign shift_en["+str(m_r)+"] = data_o_vld & (current_read_row!=0);\n")
                else:
                    rtl_code.append("assign shift_en["+str(m_r)+"] = data_o_vld;\n")

    # rtl_code.append("assign start_padding_row_flg = (current_read_row < "+str(padding_size)+") ? 1'b1 : 1'b0;\n")
    rtl_code.append("assign end_padding_row_flg = (total_write_row > "+str(row)+") ? 1'b1 : 1'b0;\n")
    rtl_code.append("assign valid_read_row = total_write_row-current_read_row;\n")
    # rtl_code.append("assign valid_read_col = (write_col_addr<<"+str(math.ceil(math.log2(in_kernel_num_col)))+") > (current_read_col+"+str(required_read_col-1)+") ? 1'b1 : 1'b0;\n")
    rtl_code.append("assign data_o_vld = (valid_read_row+end_padding_row_flg)>= "+str(required_read_row)+" ? 1'b1 : 1'b0;\n")
    rtl_code.append("\n")
    rtl_code.append("endmodule\n")

    with open(rtl_path+"/"+module_name + ".v", "w") as f:
        for i in port_code+["\n"]+decl_code+["\n"]+assign_code+["\n"]+rtl_code:
            f.write(i)