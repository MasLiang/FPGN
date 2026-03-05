import json
import math
import cvxpy as cp
import numpy as np
import copy
import random
from json_parser import json_parser
import time

def get_true_mux_row_input(total_rows, buffer_depth, write_parallelism):
    buffer_sources = [[] for _ in range(buffer_depth)]
    write_total_rows = 0
    while (write_total_rows<total_rows):
        for i in range(write_total_rows, write_total_rows+write_parallelism, 1):
            buffer_sources[i%buffer_depth].append(i-write_total_rows)
        write_total_rows += write_parallelism

    lens = [len(list(set(i))) for i in buffer_sources]
    return sum(lens)/buffer_depth

def get_true_mux_row_output(total_rows, buffer_depth, read_parallelism, stride, kernel_size, padding):
    out_rows = kernel_size * read_parallelism
    chunk_nums = ((total_rows+padding-kernel_size)//stride+1)//read_parallelism
    start_rows = 0
    chunk_idx = 0
    read_source = [[] for _ in range(out_rows)]
    for _ in range(chunk_nums):
        idx = 0
        for p in range(read_parallelism):
            start_row = stride*p+start_rows
            for k in range(kernel_size):
                read_source[idx].append((start_row+k)%buffer_depth)
                idx += 1
        start_rows += kernel_size+(read_parallelism-1)*stride
    lens = [len(list(set(i))) for i in read_source]
    return sum(lens)/out_rows

def divisors(n):
    if n <= 0:
        return [1]
    ds = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            ds.append(i)
            j = n // i
            if j != i:
                ds.append(j)
    return sorted(ds)

def solve(data, limit, h_config):
    LAYER_NUM = len(data['layers'])
    
    M = 1e6 
    

    T_comp_vals_per_w = [] 
    N_chunks_const_per_layer = [] 
    S_0_const_per_layer = []      
    K_const_per_layer = []        
    StrideRatio_const_per_layer = []     
    pairs_list = []
    lut_vals_per_w = []
    t_vals_per_w = []
    w_vals_per_w = []
    
    for i, layer in enumerate(data['layers']):
        h_i = h_config[i] 
        
        
        
        if i == 0 and layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            kernel_size_0 = layer.get('kernel_size', 3)
            stride_0 = layer.get('stride', 1)
            padding_0 = layer.get('padding', [0, 0])[0]
            required_rows_0 = kernel_size_0 + (h_i - 1) * stride_0 - padding_0
            required_rows_0 = max(1.0, required_rows_0)
            initial_data_0 = (layer['col'] // stride_0) * required_rows_0 * layer.get('in_channel', 1)
            S_0 = math.ceil(initial_data_0 / float(data.get('BW', 1)))
            S_0_const_per_layer.append(S_0)
        else:
            S_0_const_per_layer.append(0) 

        if i > 0 and layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            h_prev = h_config[i-1] 
            
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            padding_i = layer.get('padding', [0, 0])[0]
            
            required_rows_i = kernel_size_i + (h_i - 1) * stride_i - padding_i
            required_rows_i = max(1.0, required_rows_i)
            
            K_i = 1.0
            if h_prev > 0:
                K_i = math.ceil(required_rows_i / h_prev)
            K_i = max(1.0, K_i)
            K_const_per_layer.append(K_i)
            
            Ratio_i = 1.0
            if h_prev > 0:
                Ratio_i = (h_i * stride_i) / h_prev
            StrideRatio_const_per_layer.append(Ratio_i)
        else:
            K_const_per_layer.append(1.0) 
            StrideRatio_const_per_layer.append(1.0) 

        l_T_comp = []
        l_lut = []
        l_t = []
        l_w = []
        current_pairs = []

        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            N_i = (layer['row'] // layer['stride']) / float(h_i) if h_i > 0 else 1.0
            N_chunks_const_per_layer.append(N_i)
            
            w_divs = divisors(layer['col']//layer['stride'])
            
            for w_ in w_divs:
                if w_ <= 0: continue
                current_pairs.append((w_, h_i)) 
                
                T_calc = (layer['col'] // layer['stride']) / float(w_)
                T_comp = T_calc
                if i == 0: 
                    data_per_chunk = (layer['col'] // layer['stride']) * h_i * layer.get('in_channel', 1)
                    T_input = data_per_chunk / float(data.get('BW', 1))
                    T_comp = max(T_calc, T_input)
                
                l_T_comp.append(T_comp)
                
                t = w_ * h_i
                l_lut.append(layer['lut_num'] * t)
                l_t.append(t)
                l_w.append(w_)

            if not current_pairs: 
                current_pairs.append((1, h_i))
                l_T_comp.append(1e6) 
                l_lut.append(layer['lut_num'] * h_i)
                l_t.append(h_i)
                l_w.append(1)
        else:
            N_chunks_const_per_layer.append(1.0) 

            current_pairs.append((1, 1))
            l_T_comp.append(0.0)
            l_lut.append(layer.get('lut_num', 0)+layer.get('grp_sum_lut_num', 0)) 
            l_t.append(1)
            l_w.append(1)

        pairs_list.append(current_pairs)
        T_comp_vals_per_w.append(np.array(l_T_comp, dtype=float))
        lut_vals_per_w.append(np.array(l_lut, dtype=float))
        t_vals_per_w.append(np.array(l_t, dtype=float))
        w_vals_per_w.append(np.array(l_w, dtype=float))

    
    y = [cp.Variable(len(pairs_list[i]), boolean=True) for i in range(LAYER_NUM)]

    constraints = []
    for i in range(LAYER_NUM):
        constraints.append(cp.sum(y[i]) == 1)

    L_vars = [cp.Variable(nonneg=True) for _ in range(LAYER_NUM)] 
    C_vars = [cp.Variable(nonneg=True) for _ in range(LAYER_NUM)] 
    W_vars = [cp.Variable(nonneg=True) for _ in range(LAYER_NUM)] 
    S_vars = [cp.Variable(integer=True) for _ in range(LAYER_NUM)] 
    T_comp_scalar_vars = [cp.Variable(nonneg=True) for _ in range(LAYER_NUM)] 
    
    for i in range(LAYER_NUM):
        constraints.append(S_vars[i] >= 0)

    lut_total_expr = 0
    reg_total_expr = 0
    for i, layer in enumerate(data['layers']):
        h_i = h_config[i] 
        lut_total_expr += cp.sum(cp.multiply(lut_vals_per_w[i], y[i]))
        
        if layer.get('has_variable_reg', False):
            assert i < LAYER_NUM - 1, "last layer cannot have variable reg"
            h_next = h_config[i+1] 
            
            next_layer = data['layers'][i+1]
            kernel_size_next = next_layer.get('kernel_size', 3)
            stride_next = next_layer.get('stride', 1)
            required_rows = max(h_i, (h_next - 1) * stride_next + kernel_size_next)+h_i
            row_count = layer['row'] // layer['stride']  
            
            if required_rows <= row_count:
                expr1 = layer['reg_num'] + layer['reg_bias'] * h_next
                
                expr2 = 0
                if layer['type'] == 'lut_quant':
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                    expr2 = h_i * reg_per_row
                else: # lut_conv / lut_res
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel']
                    expr2 = h_i * reg_per_row
                
                r_i = cp.Variable(nonneg=True)
                constraints += [r_i >= expr1, r_i >= expr2]
                reg_base = r_i + expr2
                reg_base_const = max(expr1, expr2) + expr2
                padding_term = sum(next_layer.get('padding',[0,0])) * required_rows * next_layer.get("in_channel", 1)
            else:
                if layer['type'] == 'lut_quant':
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                    reg_base = row_count * reg_per_row
                else:
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel']
                    reg_base = row_count * reg_per_row
                reg_base_const = reg_base
                padding_term = sum(next_layer.get('padding',[0,0])) * row_count * next_layer.get("in_channel", 1)
                
            # [!!!] 需求 1: 输入 LUT [!!!]
            # 数量等于 (max(expr1, expr2) + expr2) * 系数，即 (r_i + expr2) * 系数
            # 系数 = max((log2(col) + log2(kernel_size + (h_i - 1) * stride) + 3) / 6, 1)
            col_i = layer['col']
            next_col_i = next_layer['col']
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            next_stride_i = next_layer.get('stride', 1)
            row_term = reg_base_const//reg_per_row
            #mux_w_row = h_i // math.gcd(h_i, row_term)
            mux_w_row = get_true_mux_row_input(next_layer['row'], row_term, h_i)
            mux_r_sel = min(math.ceil(next_layer['row']/h_config[i+1]/next_stride_i*1.5), math.ceil(math.log2(next_layer['row']+1)))
            input_lut_list = []
            for w_val in w_vals_per_w[i+1]:
                mux_r_col = 0 if w_val >= next_col_i//next_stride_i else 2
                mux_w_col = col_i//w_val
                total_mux_inputs = mux_w_row  + mux_r_col + (0 if mux_r_col==0 else mux_r_sel)
                if total_mux_inputs <= 1:
                    lut_per_reg = 0
                else:
                    lut_per_reg = math.ceil((total_mux_inputs-1)/5)
                input_lut_list.append(reg_base_const * lut_per_reg)
            input_lut_count = cp.sum(cp.multiply(np.array(input_lut_list), y[i+1]))
            lut_total_expr += input_lut_count

            reg_base += padding_term
            
            
            # [!!!] 需求 2: 输出 LUT [!!!]
            # m_row = max(kernel_size + h_{i+1} * (stride_{i+1} - 1), h_i) + h_i
            # 输出 LUT = ceil(ceil(ceil(log2(m_row)) + m_row) / 6) * 2 / 3) * w_{i+1} * h_{i+1} * k_{i+1}^2 * out_channel
            # [!!!] Bug Fix: 如果 h_i+1 == row 或 极限情况，则 output_lut = 0 NO MUX[!!!]
            h_ip1 = h_config[i+1]
            if h_ip1 != layer['row']//layer['stride']: #and required_rows <= row_count:
                next_layer = data['layers'][i+1]
                kernel_size_next = next_layer.get('kernel_size', 3)
                stride_next = next_layer.get('stride', 1)
                padding_next = next_layer.get('padding', [1,1])
                
                # 基础输出 LUT 计算（每个输出bit）
                #mux_row = row_term // math.gcd(h_ip1*stride_next, row_term)
                mux_row = get_true_mux_row_output(next_layer['row'], row_term, h_ip1, stride_next, kernel_size_next, sum(padding_next))
                #mux_sel = math.ceil(math.log2(next_layer['row']+1))
                mux_sel = math.ceil(min(mux_row, math.ceil(math.log2(next_layer['row']+1))))
                total_mux_input = mux_row + (mux_sel if mux_row>1 else 0)
                if mux_row <= 1:
                    output_lut_per_bit = 0
                else:
                    output_lut_per_bit = math.ceil((total_mux_input-1)/5)
                
                # [!!!] Bug Fix: 乘以输出总bit数 = w_{i+1} * h_{i+1} * k_{i+1}^2 * out_channel * [quant_channels] [!!!]
                # w_{i+1} 是变量，需要用 cvxpy 表达式
                out_channel = layer.get('out_channel', 1)
                # 如果是 lut_quant 层，还需要乘以 quant_channels
                if layer['type'] == 'lut_quant':
                    quant_channels = layer.get('quant_channels', 1)
                    #output_total_bits = h_next * (kernel_size_next ** 2) * out_channel * quant_channels * cp.sum(cp.multiply(w_vals_per_w[i+1], y[i+1]))
                    output_total_bits = ((cp.sum(cp.multiply(w_vals_per_w[i+1], y[i+1]))-1)*stride_next+kernel_size_next) * ((h_next-1)*stride_next+kernel_size_next) * out_channel * quant_channels
                else:
                    #output_total_bits = h_next * (kernel_size_next ** 2) * out_channel * cp.sum(cp.multiply(w_vals_per_w[i+1], y[i+1]))
                    output_total_bits = ((cp.sum(cp.multiply(w_vals_per_w[i+1], y[i+1]))-1)*stride_next+kernel_size_next) * ((h_next-1)*stride_next+kernel_size_next) * out_channel
                output_lut_count = output_lut_per_bit * output_total_bits
                lut_total_expr += output_lut_count
            
            pct_reg_term = layer.get('pct_reg_num', 0) * cp.sum(cp.multiply(t_vals_per_w[i], y[i]))
            
            if required_rows <= row_count:
                # 情况1：行数足够，计算 log
                log_term = math.ceil(math.log2(next_layer['col']+1)) * 2 + math.ceil(math.ceil(math.log2(next_layer['row']+1)) * 2.5)
                reg_total_expr += reg_base + pct_reg_term + log_term
            else:
                reg_total_expr += reg_base + pct_reg_term

        else: 
            base_reg = layer.get('reg_num', 0)
            pct_reg_term = layer.get('pct_reg_num', 0) * cp.sum(cp.multiply(t_vals_per_w[i], y[i]))
            
            if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
                log_term = 0
                if layer['col'] > 0 and layer['row'] > 0:
                    log_term = math.ceil(math.log2(next_layer['col']+1)) * 2 + math.ceil(math.ceil(math.log2(next_layer['row']+1)) * 2.5)
                reg_total_expr += base_reg + pct_reg_term + log_term
            else: 
                reg_total_expr += base_reg + pct_reg_term
                if layer['type'] == 'lut_fc' and i == LAYER_NUM - 1:
                    grp_sum = math.ceil(math.log2(layer['lut_num']//10))*(10+4+1)
                    grp_sum += math.ceil(math.log2(10))
                    reg_total_expr += grp_sum

    
    prev_layer_type = 'init' 

    for i, layer in enumerate(data['layers']):
        
        T_comp_expr_i = cp.sum(cp.multiply(T_comp_vals_per_w[i], y[i]))
        constraints.append(T_comp_scalar_vars[i] == T_comp_expr_i)

        if i > 0:
            if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant'] and \
               prev_layer_type in ['lut_conv', 'lut_res', 'lut_quant']:
                
                N_i = N_chunks_const_per_layer[i]
                N_prev = N_chunks_const_per_layer[i-1]
                
                constraints.append(N_i * T_comp_scalar_vars[i] <= N_prev * T_comp_scalar_vars[i-1])
        
        D_i = layer.get('additional_latency', 0) 
        D_prev = data['layers'][i-1].get('additional_latency', 0) if i > 0 else 0

        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            constraints.append(W_vars[i] >= T_comp_scalar_vars[i])
            
            if i > 0:
                Ratio_i = StrideRatio_const_per_layer[i]
                if prev_layer_type in ['lut_conv', 'lut_res', 'lut_quant']: 
                    constraints.append(W_vars[i] >= W_vars[i-1] * Ratio_i)
            else:
                constraints.append(W_vars[i] == T_comp_scalar_vars[i]) 

            N_i = N_chunks_const_per_layer[i]
            N_minus_1 = N_i - 1
            
            if i < LAYER_NUM - 1 and data['layers'][i+1]['type'] == 'lut_fc':
                padding = layer.get('padding', [0, 0])
                kernel_size = layer.get('kernel_size', 3)
                stride = layer.get('stride', 1)
                top_padding = padding[0] if len(padding) > 1 else 0  
                bottom_padding = padding[1] if len(padding) > 1 else 0  # 只取底部padding
                # Fix Bug: the bottom padding is not always needed
                # calculate how many rows are needed, then calculate how many padding rows are needed
                bottom_padding_required = (layer['row']+top_padding+bottom_padding-kernel_size)//stride*stride+kernel_size-layer['row'] - top_padding
                h_i = h_config[i]
                bottom_padding_blocks = bottom_padding_required / float(h_i*layer['stride']) if h_i > 0 else 0
                # 前面的块需要等待：(N-1-padding_blocks) * W + T_comp
                # padding块只需计算：padding_blocks * T_comp
                # 总时间 = (N-1-padding_blocks) * W + (1+padding_blocks) * T_comp
                effective_N_minus_1 = max(0, N_minus_1 - bottom_padding_blocks)
                constraints.append(C_vars[i] == effective_N_minus_1 * W_vars[i] + (1 + bottom_padding_blocks) * T_comp_scalar_vars[i])
            else:
                constraints.append(C_vars[i] == N_minus_1 * W_vars[i] + T_comp_scalar_vars[i])

            if i == 0:
                S_0 = S_0_const_per_layer[i]
                constraints.append(S_vars[i] == S_0)
            elif prev_layer_type in ['lut_fc']:
                constraints.append(S_vars[i] >= S_vars[i-1] + C_vars[i-1] + D_prev)
            else:
                K_i = K_const_per_layer[i]
                K_minus_1 = K_i - 1
                TotalDataReadyDuration = K_minus_1 * W_vars[i-1] + T_comp_scalar_vars[i-1]
                constraints.append(S_vars[i] >= S_vars[i-1] + TotalDataReadyDuration + D_prev)

            constraints.append(L_vars[i] == C_vars[i] + D_i)

        elif layer['type'] in ['lut_fc']:
            constraints.append(W_vars[i] == 0)
            constraints.append(T_comp_scalar_vars[i] == 0)
            
            if i == LAYER_NUM - 1:
                constraints.append(C_vars[i] == 4)
            else: 
                constraints.append(C_vars[i] == 0)
            
            constraints.append(L_vars[i] == C_vars[i] + D_i) 
            
            if i > 0:
                constraints.append(S_vars[i] >= S_vars[i-1] + C_vars[i-1] + D_prev)
            else:
                constraints.append(S_vars[i] == 0)
        
        else:
            if i > 0:
                constraints += [C_vars[i] == C_vars[i-1], L_vars[i] == L_vars[i-1], S_vars[i] == S_vars[i-1], W_vars[i] == W_vars[i-1]]
            else:
                constraints += [C_vars[i] == 0, L_vars[i] == 0, S_vars[i] == 0, W_vars[i] == 0]

        prev_layer_type = layer['type']

    constraints += [
        lut_total_expr <= limit['LUT'],
        reg_total_expr <= limit['FF']
    ]

    total_latency = S_vars[-1] + L_vars[-1]
    
    prob_stage1 = cp.Problem(cp.Minimize(total_latency), constraints)
    prob_stage1.solve(solver=cp.GUROBI, verbose=False, reoptimize=True)
    
    if prob_stage1.status not in ['optimal', 'optimal_inaccurate']:
        print(f"Stage 1 failed with status: {prob_stage1.status}")
        data['opt_solver_status'] = f'stage1_failed_{prob_stage1.status}'
        return data 
    
    optimal_latency = prob_stage1.value
    print(f"Optimal latency found: {optimal_latency:.8f}")
    
    print("Stage 2: Minimizing resources with fixed latency...")
    latency_tolerance = 1e-5 
    latency_constraint = total_latency <= optimal_latency + latency_tolerance
    stage2_constraints = constraints + [latency_constraint]
    
    normalized_lut = lut_total_expr / limit['LUT']
    normalized_reg = reg_total_expr / limit['FF']
    resource_objective = normalized_lut + normalized_reg
    
    prob = cp.Problem(cp.Minimize(resource_objective), stage2_constraints)
    prob.solve(solver=cp.GUROBI, verbose=False, NumericFocus=3, reoptimize=True) 

    print("status:", prob.status)
    if prob.value is not None:
        print(f"obj: {prob.value:.5f}")
    else:
        print("obj: (not found)")

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print(f"Warning: Solver status is {prob.status}")
        data['opt_solver_status'] = f'stage2_failed_{prob.status}'
        return data

    total_lut_used = 0
    total_reg_used = 0
    
    w_choices = []
    
    for i in range(LAYER_NUM):
        layer = data['layers'][i]
        h_choice = h_config[i] 
        
        ys = y[i].value
        if ys is None:
            print(f"Warning: No solution found for layer {i}")
            idx = 0 
            w_choice = 1
            if w_vals_per_w[i].size > 0:
                 w_choice = w_vals_per_w[i][0]
        else:
            idx = int(np.argmax(ys))
            if idx >= len(w_vals_per_w[i]): 
                print(f"Warning: Index {idx} out of bounds for w_vals layer {i}")
                idx = 0
            w_choice = int(w_vals_per_w[i][idx])
            
        w_choices.append(w_choice)

        layer['opt_w'] = w_choice
        layer['opt_h'] = h_choice
        
        if L_vars[i].value is not None:
            layer['opt_total_latency'] = float(L_vars[i].value)
        if C_vars[i].value is not None:
            layer['opt_computation_time'] = float(C_vars[i].value)
        if S_vars[i].value is not None:
            layer['opt_start_time'] = int(round(S_vars[i].value))
        if W_vars[i].value is not None:
            layer['opt_wait_time'] = float(W_vars[i].value)
        if T_comp_scalar_vars[i].value is not None:
            layer['opt_chunk_compute_time'] = float(T_comp_scalar_vars[i].value)

        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            if idx >= len(lut_vals_per_w[i]): 
                idx = 0
            layer['opt_lut_usage'] = int(lut_vals_per_w[i][idx])
        else:
            layer['opt_lut_usage'] = layer.get('lut_num', 0)+layer.get('grp_sum_lut_num', 0)
        total_lut_used += layer['opt_lut_usage']

    for i in range(LAYER_NUM):
        layer = data['layers'][i]
        w_choice = w_choices[i]
        h_choice = h_config[i]
        t_choice = w_choice * h_choice
        
        idx = 0
        if w_vals_per_w[i].size > 0:
            w_list = w_vals_per_w[i].tolist()
            if w_choice in w_list:
                idx = w_list.index(w_choice)
        
        if layer.get('has_variable_reg', False) and i < LAYER_NUM - 1:
            next_h = h_config[i+1]
            
            next_layer = data['layers'][i+1]
            kernel_size_next = next_layer.get('kernel_size', 3)
            stride_next = next_layer.get('stride', 1)
            required_rows = max(h_choice, (next_h - 1) * stride_next + kernel_size_next)+h_choice
            row_count = layer['row']// layer['stride']

            if required_rows <= row_count:
                expr2 = 0
                if layer['type'] == 'lut_quant':
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                    expr2 = h_choice * reg_per_row
                else: # lut_conv / lut_res
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel']
                    expr2 = h_choice * reg_per_row
                expr1 = layer['reg_num'] + layer['reg_bias'] * next_h
                base_reg = max(expr1, expr2) + expr2
                padding_term = sum(next_layer.get('padding',[0,0])) * required_rows * next_layer['in_channel']
            else:
                if layer['type'] == 'lut_quant':
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                    base_reg = row_count * reg_per_row
                else:
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel']
                    base_reg = row_count * reg_per_row
                padding_term = sum(next_layer.get('padding',[0,0])) * row_count * next_layer['in_channel']
                
            # 计算 input_lut 系数
            col_i = layer['col']
            next_col_i = next_layer['col']
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            next_stride_i = next_layer.get('stride', 1)
            row_term = base_reg // reg_per_row
            #mux_w_row = h_choice // math.gcd(h_choice, row_term)
            mux_w_row = get_true_mux_row_input(next_layer['row'], row_term, h_choice)
            mux_r_sel = min(math.ceil(next_layer['row']/h_config[i+1]/next_stride_i*1.5), math.ceil(math.log2(next_layer['row']+1)))
            mux_r_col = 0 if (w_choices[i+1] >= (next_col_i//next_stride_i)) else 2
            total_mux_inputs = mux_w_row + mux_r_col + (0 if mux_r_col==0 else mux_r_sel)
            print("h_choice: "+str(h_choice)+" row_term: "+str(row_term))
            print("w_choices[i+1]: "+str(w_choices[i+1])+" col_i: "+str(next_col_i)+" stride_i: "+str(next_stride_i))
            print("layer type: "+layer['type']+" next_layer_type: "+next_layer['type'])
            if total_mux_inputs <= 1:
                lut_per_reg = 0
            elif total_mux_inputs <= 6:
                lut_per_reg = 1
            else:
                lut_per_reg = math.ceil((total_mux_inputs-1)/5)
            print("next_row: "+str(next_layer['row'])+" h_i+1: "+str(h_config[i+1])+" nxt_stride: "+str(next_stride_i)+" nxt row log: "+str(math.ceil(math.log(next_layer['row']+1))))
            print("mux_w_row: "+str(mux_w_row)+" mux_w_col: "+str(mux_w_col)+" mux_r_col: "+str(mux_r_col)+" mux_r_sel: "+str(mux_r_sel)+" lut_per_reg: "+str(lut_per_reg)+" base reg: "+str(base_reg))
            input_lut = lut_per_reg * base_reg
            print("input lut: "+str(input_lut))
            base_reg += padding_term
            
            layer['opt_input_lut_usage'] = int(input_lut)
            total_lut_used += int(input_lut)
            
            # [!!!] 需求 2: 输出 LUT (结果记录) [!!!]
            # [!!!] Bug Fix: 如果 h_next == row 或 极限情况，则 output_lut = 0（不需要缓存）[!!!]
            if next_h != layer['row']//layer['stride']: #and required_rows <= row_count:
                next_layer = data['layers'][i+1]
                kernel_size_next = next_layer.get('kernel_size', 3)
                stride_next = next_layer.get('stride', 1)
                padding_next = next_layer.get('padding', [1,1])
                w_next = w_choices[i+1]
                
                #mux_row = row_term // math.gcd(next_h*stride_next, row_term)
                print("get true mux row output: "+str(next_layer['row'])+str(row_term)+str(next_h)+str(stride_next)+str(kernel_size_next)+str(sum(padding_next)))
                mux_row = get_true_mux_row_output(next_layer['row'], row_term, next_h, stride_next, kernel_size_next, sum(padding_next))
                #mux_sel = math.ceil(math.log2(next_layer['row']+1))
                mux_sel = math.ceil(min(mux_row, math.ceil(math.log2(next_layer['row']+1))))
                total_mux_input = mux_row + (mux_sel if mux_row>1 else 0)
                print("***out*** row_term: "+str(row_term)+" mux_row: "+str(mux_row)+" mux_sel: "+str(mux_sel)+" total_mux_input: "+str(total_mux_input))
                if total_mux_input <= 1:
                    output_lut_per_bit = 0
                else:
                    output_lut_per_bit = math.ceil((total_mux_input-1)/5)
                
                # [!!!] Bug Fix: 乘以输出总bit数 = w_{i+1} * h_{i+1} * k_{i+1}^2 * out_channel * [quant_channels] [!!!]
                out_channel = layer.get('out_channel', 1)
                # 如果是 lut_quant 层，还需要乘以 quant_channels
                if layer['type'] == 'lut_quant':
                    quant_channels = layer.get('quant_channels', 1)
                    #output_total_bits = w_next * next_h * (kernel_size_next ** 2) * out_channel * quant_channels
                    output_total_bits = ((w_next-1)*stride_next+kernel_size_next) * ((next_h-1)*stride_next+kernel_size_next) * out_channel * quant_channels
                else:
                    #output_total_bits = w_next * next_h * (kernel_size_next ** 2) * out_channel
                    output_total_bits = ((w_next-1)*stride_next+kernel_size_next) * ((next_h-1)*stride_next+kernel_size_next) * out_channel
                output_lut = output_lut_per_bit * output_total_bits
                print("output_lut_count: "+str(output_lut)+" output_lut_per_bit: "+str(output_lut_per_bit)+" output_total_bits: "+str(output_total_bits))
                layer['opt_output_lut_usage'] = int(output_lut)
                total_lut_used += int(output_lut)
            else:
                layer['opt_output_lut_usage'] = 0
            
            pct_reg = layer.get('pct_reg_num', 0) * t_vals_per_w[i][idx]
            
            if required_rows <= row_count:
                log_reg = 0
                log_reg = math.ceil(math.log2(next_layer['col']+1)) * 2 + math.ceil(math.ceil(math.log2(next_layer['row']+1)) * 2.5) # addrs. 2 is col, 2.5 is row
                print("base_reg: "+str(base_reg)+" pct_reg: "+str(pct_reg)+" log_reg: "+str(log_reg)+" col: "+str(next_layer['col'])+" row: "+str(next_layer['row']))
                base_reg = base_reg + log_reg
            
            base_reg = base_reg + pct_reg 

        else: 
            layer['opt_input_lut_usage'] = 0
            layer['opt_output_lut_usage'] = 0
            
            base_reg = layer.get('reg_num', 0)
            pct_reg = layer.get('pct_reg_num', 0) * t_vals_per_w[i][idx]
            #print("base_reg: "+str(base_reg)+" pct_reg: "+str(pct_reg)+" log_reg: "+str(0))
            base_reg = base_reg + pct_reg
            
            if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
                if layer['col'] > 0 and layer['row'] > 0:
                    log_term = math.ceil(math.log2(next_layer['col']+1)) * 2 + math.ceil(math.ceil(math.log2(next_layer['row']+1)) * 2.5)
                    base_reg += log_term

        if layer['type'] == 'lut_fc' and i == LAYER_NUM - 1:
            grp_sum = math.ceil(math.log2(layer['lut_num']//10))*(10+4+1)
            grp_sum += math.ceil(math.log2(10))
            base_reg += grp_sum

        layer['opt_reg_usage'] = int(base_reg)
        total_reg_used += layer['opt_reg_usage']

    data['opt_final_computation_time'] = float(C_vars[-1].value) if C_vars[-1].value is not None else 0.0
    data['opt_final_total_latency'] = float(L_vars[-1].value) if L_vars[-1].value is not None else 0.0
    data['opt_final_start_time'] = int(round(S_vars[-1].value)) if S_vars[-1].value is not None else 0
    data['opt_final_wait_time'] = float(W_vars[-1].value) if W_vars[-1].value is not None else 0.0
    data['opt_total_latency'] = data['opt_final_total_latency'] + data['opt_final_start_time']
    data['opt_total_lut_used'] = total_lut_used
    data['opt_total_reg_used'] = total_reg_used
    data['opt_lut_utilization'] = total_lut_used / limit['LUT']
    data['opt_reg_utilization'] = total_reg_used / limit['FF']
    data['opt_solver_status'] = prob.status
    
    return data

def check_resource_pruning(base_data, limit, h_config):
    """
    计算 (w=1, h=h_config) 时的资源消耗下限。
    如果下限超过限制，返回 False (应剪枝)。
    """
    total_lut_base = 0
    total_reg_base = 0
    LAYER_NUM = len(base_data['layers'])
    
    for i, layer in enumerate(base_data['layers']):
        h_choice = h_config[i]
        w_choice = 1 
        t_choice = w_choice * h_choice

        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            total_lut_base += layer['lut_num'] * t_choice
        else:
            total_lut_base += layer.get('lut_num', 0)+layer.get('grp_sum_lut_num', 0) 
            
        if layer.get('has_variable_reg', False) and i < LAYER_NUM - 1:
            next_h = h_config[i+1]
            
            next_layer = base_data['layers'][i+1]
            kernel_size_next = next_layer.get('kernel_size', 3)
            stride_next = next_layer.get('stride', 1)
            required_rows = max(h_choice, (next_h - 1) * stride_next + kernel_size_next)+h_choice
            row_count = layer['row'] // layer['stride']
            
            expr2 = 0
            if required_rows <= row_count:
                # 情况1：行数足够
                if layer['type'] == 'lut_quant':
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                    expr2 = h_choice * reg_per_row
                else: # lut_conv / lut_res
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel']
                    expr2 = h_choice * reg_per_row
                expr1 = layer['reg_num'] + layer['reg_bias'] * next_h
                base_reg = max(expr1, expr2) + expr2
                padding_term = sum(next_layer.get('padding',[0,0])) * required_rows * next_layer['in_channel']
            else:
                if layer['type'] == 'lut_quant':
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                    base_reg = row_count * reg_per_row
                else:
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel']
                    base_reg = row_count * reg_per_row
                padding_term = sum(next_layer.get('padding',[0,0])) * row_count * next_layer['in_channel']
                
            # 计算 input_lut 系数
            col_i = layer['col']
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            row_term = base_reg // reg_per_row
            #mux_w_row = h_choice // math.gcd(h_choice, row_term)
            mux_w_row = get_true_mux_row_input(next_layer['row'], row_term, h_choice)
            mux_w_col = col_i // w_choice
            mux_r_sel = min(math.ceil(next_layer['row']/next_h/stride_next*1.5), math.ceil(math.log2(next_layer['row']+1)))
            mux_r_col = 0 if w_choice >= col_i//stride_i else 2
            total_mux_inputs = mux_w_row + mux_r_col + (0 if mux_r_col==0 else mux_r_sel)
            if total_mux_inputs <= 1:
                lut_per_reg = 0
            elif total_mux_inputs <= 6:
                lut_per_reg = 1
            else:
                lut_per_reg = math.ceil((total_mux_inputs-1)/5)
            input_lut = lut_per_reg * base_reg
            base_reg = base_reg + padding_term
            
            total_lut_base += input_lut
            
            # [!!!] 需求 2: 输出 LUT (剪枝版本) [!!!]
            # [!!!] Bug Fix: 如果 h_choice == row 或 极限情况，则 output_lut = 0（不需要缓存）[!!!]
            h_next = h_config[i+1]
            if h_next != layer['row']//layer['stride']: #and required_rows <= row_count:
                next_layer = base_data['layers'][i+1]
                kernel_size_next = next_layer.get('kernel_size', 3)
                stride_next = next_layer.get('stride', 1)
                padding_next = next_layer.get('padding', [1,1])
                w_next = 1  # 假设下一层 w 最小
                
                #mux_row = row_term // math.gcd(next_h*stride_next, row_term)
                mux_row = get_true_mux_row_output(next_layer['row'], row_term, next_h, stride_next, kernel_size_next, sum(padding_next))
                mux_sel = math.ceil(min(mux_row, math.ceil(math.log2(next_layer['row']+1))))
                total_mux_input = mux_row + (mux_sel if mux_row>1 else 0)
                if total_mux_input <= 1:
                    output_lut_per_bit = 0
                else:
                    output_lut_per_bit = math.ceil((total_mux_input-1)/5)
                
                # [!!!] Bug Fix: 乘以输出总bit数 = w_{i+1} * h_{i+1} * k_{i+1}^2 * out_channel * [quant_channels] [!!!]
                out_channel = layer.get('out_channel', 1)
                # 如果是 lut_quant 层，还需要乘以 quant_channels
                if layer['type'] == 'lut_quant':
                    quant_channels = layer.get('quant_channels', 1)
                    #output_total_bits = w_next * next_h * (kernel_size_next ** 2) * out_channel * quant_channels
                    output_total_bits = ((w_next-1)*stride_next+kernel_size_next) * ((next_h-1)*stride_next+kernel_size_next) * out_channel * quant_channels
                else:
                    #output_total_bits = w_next * next_h * (kernel_size_next ** 2) * out_channel
                    output_total_bits = ((w_next-1)*stride_next+kernel_size_next) * ((next_h-1)*stride_next+kernel_size_next) * out_channel
                output_lut = output_lut_per_bit * output_total_bits
                total_lut_base += output_lut
            
            pct_reg = layer.get('pct_reg_num', 0) * t_choice
            
            if required_rows <= row_count:
                log_reg = 0
                log_reg = math.ceil(math.log2(next_layer['col']+1)) * 2 + math.ceil(math.ceil(math.log2(next_layer['row']+1)) * 2.5)
                base_reg = base_reg + log_reg
            
            base_reg = base_reg + pct_reg

        else: 
            base_reg = layer.get('reg_num', 0)
            pct_reg = layer.get('pct_reg_num', 0) * t_choice
            base_reg = base_reg + pct_reg
            if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
                if layer['col'] > 0 and layer['row'] > 0:
                    log_term = math.ceil(math.log2(next_layer['col']+1)) * 2 + math.ceil(math.ceil(math.log2(next_layer['row']+1)) * 2.5)
                    base_reg += log_term
            
        if layer['type'] == 'lut_fc' and i == LAYER_NUM - 1:
            grp_sum = math.ceil(math.log2(layer['lut_num']//10))*(10+4+1)
            grp_sum += math.ceil(math.log2(10))
            base_reg += grp_sum

        total_reg_base += int(base_reg)

    return True 

def check_latency_pruning(base_data, h_config):
    LAYER_NUM = len(base_data['layers'])
    
    T_comp_min = [0.0] * LAYER_NUM
    N_const = [0.0] * LAYER_NUM
    S_const = [0.0] * LAYER_NUM
    K_const = [0.0] * LAYER_NUM
    Ratio_const = [0.0] * LAYER_NUM
    
    W_min = [0.0] * LAYER_NUM
    C_min = [0.0] * LAYER_NUM
    S_min = [0] * LAYER_NUM
    
    for i, layer in enumerate(base_data['layers']):
        h_i = h_config[i]
        
        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            N_const[i] = (layer['row'] // layer['stride']) / float(h_i) if h_i > 0 else 1.0
            
            T_calc_min = 1.0 
            T_comp_min[i] = T_calc_min
            if i == 0: 
                data_per_chunk = (layer['col'] // layer['stride']) * h_i * layer.get('in_channel', 1)
                T_input_0 = data_per_chunk / float(base_data.get('BW', 1))
                T_comp_min[i] = max(T_calc_min, T_input_0)
            
            if i == 0:
                kernel_size_0 = layer.get('kernel_size', 3)
                stride_0 = layer.get('stride', 1)
                padding_0 = layer.get('padding', [0, 0])[0]
                required_rows_0 = kernel_size_0 + (h_i - 1) * stride_0 - padding_0
                required_rows_0 = max(1.0, required_rows_0)
                initial_data_0 = (layer['col'] // stride_0) * required_rows_0 * layer.get('in_channel', 1)
                S_const[i] = math.ceil(initial_data_0 / float(base_data.get('BW', 1)))
            
            if i > 0:
                h_prev = h_config[i-1]
                kernel_size_i = layer.get('kernel_size', 3)
                stride_i = layer.get('stride', 1)
                padding_i = layer.get('padding', [0, 0])[0]
                required_rows_i = kernel_size_i + (h_i - 1) * stride_i - padding_i
                required_rows_i = max(1.0, required_rows_i)
                
                K_const[i] = max(1.0, math.ceil(required_rows_i / h_prev)) if h_prev > 0 else 1.0
                Ratio_const[i] = (h_i * stride_i) / h_prev if h_prev > 0 else 1.0
            else:
                K_const[i] = 1.0
                Ratio_const[i] = 1.0
        

    prev_layer_type = 'init'
    for i, layer in enumerate(base_data['layers']):
        D_i = layer.get('additional_latency', 0) 
        D_prev = base_data['layers'][i-1].get('additional_latency', 0) if i > 0 else 0

        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            W_min[i] = T_comp_min[i]
            if i > 0:
                if prev_layer_type in ['lut_conv', 'lut_res', 'lut_quant']:
                    W_min[i] = max(T_comp_min[i], W_min[i-1] * Ratio_const[i])
            
            N_minus_1 = N_const[i] - 1
            
            if i < LAYER_NUM - 1 and base_data['layers'][i+1]['type'] == 'lut_fc':
                padding = layer.get('padding', [0, 0])
                kernel_size = layer.get('kernel_size', 3)
                top_padding = padding[0] if len(padding) > 1 else 0  
                bottom_padding = padding[1] if len(padding) > 1 else 0  # 只取底部padding
                h_i = h_config[i]
                bottom_padding_required = (layer['row']+top_padding+bottom_padding-kernel_size)//layer['stride']*layer['stride']+kernel_size-layer['row'] - top_padding
                
                bottom_padding_blocks = bottom_padding_required / float(h_i*layer['stride']) if h_i > 0 else 0
                effective_N_minus_1 = max(0, N_minus_1 - bottom_padding_blocks)
                C_min[i] = effective_N_minus_1 * W_min[i] + (1 + bottom_padding_blocks) * T_comp_min[i]
            else:
                C_min[i] = N_minus_1 * W_min[i] + T_comp_min[i]
            
            if i == 0:
                S_min[i] = S_const[i]
            elif prev_layer_type in ['lut_fc']:
                S_min[i] = S_min[i-1] + C_min[i-1] + D_prev
            else:
                K_minus_1 = K_const[i] - 1
                DataReadyTime = K_minus_1 * W_min[i-1] + T_comp_min[i-1]
                S_min[i] = S_min[i-1] + DataReadyTime + D_prev
                
        elif layer['type'] in ['lut_fc']:
            if i == LAYER_NUM - 1:
                C_min[i] = 4
            else: 
                C_min[i] = 0
            
            if i > 0:
                S_min[i] = S_min[i-1] + C_min[i-1] + D_prev
            else:
                S_min[i] = 0
        
        prev_layer_type = layer['type']

    L_last = C_min[LAYER_NUM - 1] + base_data['layers'][LAYER_NUM - 1].get('additional_latency', 0)
    L_lower_bound = S_min[LAYER_NUM - 1] + L_last
    return L_lower_bound

def heuristic_h_search_best(json_path, limit, mode='steepest', random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    start_time = time.time()
    base_data = json_parser(json_path)
    LAYER_NUM = len(base_data['layers'])
    
    h_options_per_layer = []
    for i, layer in enumerate(base_data['layers']):
        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            row_stride = layer['row'] // layer['stride']
            if row_stride <= 0:
                print(f"Warning: Layer {i} row/stride is <= 0. Using [1].")
                h_divs = [1]
            else:
                h_divs = divisors(row_stride)
            print(f"Layer {i} ({layer['type']}) h options: {h_divs}")
            h_options_per_layer.append([h for h in h_divs if h > 0])
        else:
            h_options_per_layer.append([1]) 

    h_current_best = [random.choice(options) for options in h_options_per_layer]
    
    data = json_parser(json_path) 
    solution_data = solve(copy.deepcopy(data), limit, h_current_best)
    
    if (solution_data and 
        solution_data.get('opt_solver_status') in ['optimal', 'optimal_inaccurate']):
        
        latency_current_best = solution_data['opt_total_latency']
    else:
        print(f"!!! 错误: 基准 H_CONFIG {h_current_best} 求解失败。")
        print("请检查模型或资源约束。")
        exit()
        
    best_solution_data = solution_data

    # 4. 开始迭代搜索
    iteration = 0
    improved_last_iteration = True 
    
    while improved_last_iteration:
        iteration += 1
        print(f"\n--- [ 迭代 {iteration} ] ---")
        
        improved_in_this_iteration = False 
        
        # 我们仍然跟踪本轮的最好解，但只用于比较
        iteration_best_h = list(h_current_best) 
        iteration_best_latency = latency_current_best
        iteration_best_data = best_solution_data

        # 遍历每一层
        for i in range(LAYER_NUM):
            # 遍历该层的每一个 h 选项
            for h_test_option in h_options_per_layer[i]:
                
                if h_test_option == h_current_best[i]:
                    continue
                    
                h_test_config = list(h_current_best) # 关键: 总是从*迭代开始时*的解出发
                h_test_config[i] = h_test_option
                
                print(f"正在考虑 H_CONFIG: {h_test_config}")
                
                # 剪枝 1: 资源剪枝 [!!!]
                if not check_resource_pruning(base_data, limit, h_test_config):
                    print(f"--- 结果: (跳过 - 资源剪枝)\n")
                    continue 
                    
                # 剪枝 2: 延迟剪枝 [!!!]
                L_lower_bound = check_latency_pruning(base_data, h_test_config)
                # 关键: 必须严格小于 (<), 而不是 <=
                if L_lower_bound > (latency_current_best - 1e-5):
                    print(f"--- 剪枝 (延迟下限 {L_lower_bound:.2f} > 当前最优 {latency_current_best:.2f}) ---")
                    print(f"--- 结果: (跳过 - 延迟剪枝)\n")
                    continue 
                
                print(f"--- (未剪枝, 正在运行求解器...) ---")
                data = json_parser(json_path) # 重新加载
                solution_data = solve(copy.deepcopy(data), limit, h_test_config)
                
                if (solution_data and 
                    solution_data.get('opt_solver_status') in ['optimal', 'optimal_inaccurate']):
                    
                    current_latency = solution_data['opt_total_latency']
                    
                    if current_latency < (iteration_best_latency - 1e-5):
                         print(f"*** 找到本轮更优解! ***")
                         print(f"*** 延迟: {current_latency:.4f} (优于 {iteration_best_latency:.4f})")
                         print(f"*** 新 H_CONFIG: {h_test_config}\n")

                         iteration_best_latency = current_latency
                         iteration_best_h = h_test_config
                         iteration_best_data = solution_data 
                         improved_in_this_iteration = True # 标记我们找到了一个真正更好的山谷
                    
                    # (我们不再关心 "横向移动")
                    else:
                        print(f"测试结果: {current_latency:.4f} (未改进)\n")
                else:
                    status = "failed (invalid)"
                    if solution_data:
                         status = solution_data.get('opt_solver_status', 'failed')
                    print(f"H_CONFIG {h_test_config} 求解失败. 状态: {status}\n")

        # (一轮迭代 *所有邻居* 结束)
        
        if improved_in_this_iteration:
             # 我们在这一轮找到了一个*最陡峭*的下降
             # 更新我们的基准
             print(f"--- [ 迭代 {iteration} 结束 ] ---")
             print(f"选择最优邻居: {iteration_best_h} (延迟: {iteration_best_latency:.4f})")
             h_current_best = iteration_best_h
             latency_current_best = iteration_best_latency
             best_solution_data = iteration_best_data
             # improved_last_iteration 保持 True, 循环将继续
        else:
            # 我们测试了所有邻居, 没有一个*严格*比当前的更好
            print(f"--- [ 迭代 {iteration} 结束 ] ---")
            print("未找到更多改进。搜索收敛。")
            improved_last_iteration = False # [!!!] 退出 while 循环 [!!!]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总耗时: {elapsed_time:.2f} 秒")

    # (while 循环结束, 搜索收敛)

    # --- 5. 打印最终的最优解 (heuristic_h_search_fast) ---
    
    if best_solution_data:
        print(f"\n\n==========================================")
        print(f"=== 启发式搜索完成 (Fast) ===")
        print(f"==========================================")
        print(f"最优 H Config: {h_current_best}\n")
        
        for i, layer in enumerate(best_solution_data['layers']):
            lut_info = f"LUT={layer.get('opt_lut_usage', 0)}"
            if layer.get('opt_input_lut_usage', 0) > 0:
                lut_info += f"+input_lut={layer['opt_input_lut_usage']}"
            if layer.get('opt_output_lut_usage', 0) > 0:
                lut_info += f"+output_lut={layer['opt_output_lut_usage']}"
            
            print(f"layer {i}: type={layer['type']}, w={layer['opt_w']}, h={layer['opt_h']}, "
                  f"S={layer['opt_start_time']}, C={layer['opt_computation_time']:.3f}, "
                  f"W={layer['opt_wait_time']:.3f}, T_comp={layer['opt_chunk_compute_time']:.3f}, {lut_info}")

        print(f"\n=== 优化结果 ===")
        print(f"LUT Usage: {best_solution_data['opt_total_lut_used']}/{limit['LUT']} ({best_solution_data['opt_lut_utilization']:.2%})")
        print(f"FF Usage: {best_solution_data['opt_total_reg_used']}/{limit['FF']} ({best_solution_data['opt_reg_utilization']:.2%})")
        print(f"Total Latency: {best_solution_data['opt_total_latency']:.8f}")

        output_path = "model_execution_info_optimized.json"
        with open(output_path, "w") as f:
            json.dump(best_solution_data, f, indent=2)
        print(f"\nOptimized data saved to {output_path}")
        return best_solution_data

    else:
        print("\n\n==========================================")
        print(f"=== 优化失败 ===")
        print("未找到任何可行解。")
        print("==========================================")
        return None

def heuristic_h_search_best(json_path, limit, mode='steepest', random_seed=None):
    """
    启发式搜索 - 带横向移动的爬山算法，使用随机起始点
    
    Args:
        json_path: JSON配置文件路径
        limit: 资源限制字典
        mode: 搜索模式（保留参数，当前未使用）
        random_seed: 随机种子（可选），用于重现实验结果
    """
    if random_seed is not None:
        random.seed(random_seed)
        print(f"使用随机种子: {random_seed}")
    
    start_time = time.time()
    # 1. 加载一次数据，以获取层数和 h 的选项
    base_data = json_parser(json_path)
    LAYER_NUM = len(base_data['layers'])
    
    # 2. 为每一层生成 h 的所有可能选项
    h_options_per_layer = []
    for i, layer in enumerate(base_data['layers']):
        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            # 确保 row/stride 大于 0
            row_stride = layer['row'] // layer['stride']
            if row_stride <= 0:
                print(f"警告: Layer {i} row/stride is <= 0. Using [1].")
                h_divs = [1]
            else:
                h_divs = divisors(row_stride)
            print(f"Layer {i} ({layer['type']}) h options: {h_divs}")
            h_options_per_layer.append([h for h in h_divs if h > 0])
        else:
            h_options_per_layer.append([1]) # FC 层 h 必须为 1

    # 3. 初始化爬山算法 - [!!!] 随机选择起始点 [!!!]
    #h_current_best = [random.choice(options) for options in h_options_per_layer]
    h_current_best = [options[0] for options in h_options_per_layer] 
    #h_current_best = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    print(f"=== 启发式搜索启动 (模式: {mode}, 随机起始点) ===")
    print(f"随机选择的起始 H_CONFIG: {h_current_best}")
    data = json_parser(json_path) 
    solution_data = solve(copy.deepcopy(data), limit, h_current_best)
    
    if (solution_data and 
        solution_data.get('opt_solver_status') in ['optimal', 'optimal_inaccurate']):
        
        latency_current_best = solution_data['opt_total_latency']
        print(f"基准 H_CONFIG: {h_current_best}")
        print(f"基准延迟: {latency_current_best:.4f}\n")
    else:
        print(f"!!! 错误: 基准 H_CONFIG {h_current_best} 求解失败。")
        print("请检查模型或资源约束。")
        return None
        
    best_solution_data = solution_data
    
    visited_set = set()
    visited_set.add(tuple(h_current_best))

    iteration = 0
    
    while True: 
        iteration += 1
        print(f"\n--- [ Iteration {iteration} ] ---")
        
        improved_in_this_iteration = False 
        
        iteration_best_h = list(h_current_best) 
        iteration_best_latency = latency_current_best
        iteration_best_data = best_solution_data

        for i in range(LAYER_NUM):
            for h_test_option in h_options_per_layer[i]:
                
                h_test_config = list(h_current_best) 
                h_test_config[i] = h_test_option
                
                if tuple(h_test_config) in visited_set:
                    continue 
                    
                print(f"Solving for H_CONFIG: {h_test_config}")
                
                if not check_resource_pruning(base_data, limit, h_test_config):
                    print(f"--- skip - resource pruning\n")
                    visited_set.add(tuple(h_test_config)) 
                    continue 
                    
                L_lower_bound = check_latency_pruning(base_data, h_test_config)
                if L_lower_bound > (latency_current_best + 1e-5): 
                    print(f"--- skip - latency pruning\n")
                    visited_set.add(tuple(h_test_config)) 
                    continue 
                
                print(f"--- start solving ---")
                data = json_parser(json_path) 
                solution_data = solve(copy.deepcopy(data), limit, h_test_config)
                visited_set.add(tuple(h_test_config)) 
                
                if (solution_data and 
                    solution_data.get('opt_solver_status') in ['optimal', 'optimal_inaccurate']):
                    
                    current_latency = solution_data['opt_total_latency']
                    
                    if current_latency <= (iteration_best_latency + 1e-5):
                        
                        iteration_best_latency = current_latency
                        iteration_best_h = h_test_config
                        iteration_best_data = solution_data 
                        
                        if current_latency < (latency_current_best - 1e-5):
                            improved_in_this_iteration = True 
                else:
                    status = "failed (invalid)"
                    if solution_data:
                         status = solution_data.get('opt_solver_status', 'failed')
                    print(f"H_CONFIG {h_test_config} solver failed: {status}\n")

        if h_current_best == iteration_best_h:
            break 
        else:
            h_current_best = iteration_best_h
            latency_current_best = iteration_best_latency
            best_solution_data = iteration_best_data
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\ntotal run time: {elapsed_time:.2f}s")


    
    if best_solution_data:
        print(f"==========================================")
        print(f"The best H Config: {h_current_best}\n")
        
        for i, layer in enumerate(best_solution_data['layers']):
            lut_info = f"LUT={layer.get('opt_lut_usage', 0)}"
            if layer.get('opt_input_lut_usage', 0) > 0:
                lut_info += f"+input_lut={layer['opt_input_lut_usage']}"
            if layer.get('opt_output_lut_usage', 0) > 0:
                lut_info += f"+output_lut={layer['opt_output_lut_usage']}"
            
            print(f"layer {i}: type={layer['type']}, w={layer['opt_w']}, h={layer['opt_h']}, "
                  f"S={layer['opt_start_time']}, C={layer['opt_computation_time']:.3f}, "
                  f"W={layer['opt_wait_time']:.3f}, T_comp={layer['opt_chunk_compute_time']:.3f}, {lut_info}")

        print(f"LUT Usage: {best_solution_data['opt_total_lut_used']}/{limit['LUT']} ({best_solution_data['opt_lut_utilization']:.2%})")
        print(f"FF Usage: {best_solution_data['opt_total_reg_used']}/{limit['FF']} ({best_solution_data['opt_reg_utilization']:.2%})")
        print(f"Total Latency: {best_solution_data['opt_total_latency']:.8f}")

        output_path = "model_execution_info_optimized.json"
        with open(output_path, "w") as f:
            json.dump(best_solution_data, f, indent=2)
        print(f"\nOptimized data saved to {output_path}")
        return best_solution_data

    else:
        print("\n\n==========================================")
        print(f"=== Failed ===")
        print("Not solution found.")
        print("==========================================")
        return None

if __name__ == "__main__":
    json_path = "model_execution_info_6g.json"
    
    limit = {
        "LUT": 2000000,
        "FF":  4000000
    }

    heuristic_h_search_best(json_path, limit)
