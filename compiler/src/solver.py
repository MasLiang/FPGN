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
                else: 
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
                
            col_i = layer['col']
            next_col_i = next_layer['col']
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            next_stride_i = next_layer.get('stride', 1)
            row_term = reg_base_const//reg_per_row
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
            
            
            h_ip1 = h_config[i+1]
            if h_ip1 != layer['row']//layer['stride']: 
                next_layer = data['layers'][i+1]
                kernel_size_next = next_layer.get('kernel_size', 3)
                stride_next = next_layer.get('stride', 1)
                padding_next = next_layer.get('padding', [1,1])
                
                mux_row = get_true_mux_row_output(next_layer['row'], row_term, h_ip1, stride_next, kernel_size_next, sum(padding_next))
                mux_sel = math.ceil(min(mux_row, math.ceil(math.log2(next_layer['row']+1))))
                total_mux_input = mux_row + (mux_sel if mux_row>1 else 0)
                if mux_row <= 1:
                    output_lut_per_bit = 0
                else:
                    output_lut_per_bit = math.ceil((total_mux_input-1)/5)
                
                out_channel = layer.get('out_channel', 1)
                if layer['type'] == 'lut_quant':
                    quant_channels = layer.get('quant_channels', 1)
                    output_total_bits = ((cp.sum(cp.multiply(w_vals_per_w[i+1], y[i+1]))-1)*stride_next+kernel_size_next) * ((h_next-1)*stride_next+kernel_size_next) * out_channel * quant_channels
                else:
                    output_total_bits = ((cp.sum(cp.multiply(w_vals_per_w[i+1], y[i+1]))-1)*stride_next+kernel_size_next) * ((h_next-1)*stride_next+kernel_size_next) * out_channel
                output_lut_count = output_lut_per_bit * output_total_bits
                lut_total_expr += output_lut_count
            
            pct_reg_term = layer.get('pct_reg_num', 0) * cp.sum(cp.multiply(t_vals_per_w[i], y[i]))
            
            if required_rows <= row_count:
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
                bottom_padding = padding[1] if len(padding) > 1 else 0  
                bottom_padding_required = (layer['row']+top_padding+bottom_padding-kernel_size)//stride*stride+kernel_size-layer['row'] - top_padding
                h_i = h_config[i]
                bottom_padding_blocks = bottom_padding_required / float(h_i*layer['stride']) if h_i > 0 else 0
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
        data['opt_solver_status'] = f'stage1_failed_{prob_stage1.status}'
        return data 
    
    optimal_latency = prob_stage1.value
    
    latency_tolerance = 1e-5 
    latency_constraint = total_latency <= optimal_latency + latency_tolerance
    stage2_constraints = constraints + [latency_constraint]
    
    normalized_lut = lut_total_expr / limit['LUT']
    normalized_reg = reg_total_expr / limit['FF']
    resource_objective = normalized_lut + normalized_reg
    
    prob = cp.Problem(cp.Minimize(resource_objective), stage2_constraints)
    prob.solve(solver=cp.GUROBI, verbose=False, NumericFocus=3, reoptimize=True) 

    if prob.status not in ['optimal', 'optimal_inaccurate']:
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
            idx = 0 
            w_choice = 1
            if w_vals_per_w[i].size > 0:
                 w_choice = w_vals_per_w[i][0]
        else:
            idx = int(np.argmax(ys))
            if idx >= len(w_vals_per_w[i]): 
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
                else: 
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
                
            col_i = layer['col']
            next_col_i = next_layer['col']
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            next_stride_i = next_layer.get('stride', 1)
            row_term = base_reg // reg_per_row
            mux_w_row = get_true_mux_row_input(next_layer['row'], row_term, h_choice)
            mux_r_sel = min(math.ceil(next_layer['row']/h_config[i+1]/next_stride_i*1.5), math.ceil(math.log2(next_layer['row']+1)))
            mux_r_col = 0 if (w_choices[i+1] >= (next_col_i//next_stride_i)) else 2
            total_mux_inputs = mux_w_row + mux_r_col + (0 if mux_r_col==0 else mux_r_sel)
            if total_mux_inputs <= 1:
                lut_per_reg = 0
            elif total_mux_inputs <= 6:
                lut_per_reg = 1
            else:
                lut_per_reg = math.ceil((total_mux_inputs-1)/5)
            input_lut = lut_per_reg * base_reg
            base_reg += padding_term
            
            layer['opt_input_lut_usage'] = int(input_lut)
            total_lut_used += int(input_lut)
            
            if next_h != layer['row']//layer['stride']: 
                next_layer = data['layers'][i+1]
                kernel_size_next = next_layer.get('kernel_size', 3)
                stride_next = next_layer.get('stride', 1)
                padding_next = next_layer.get('padding', [1,1])
                w_next = w_choices[i+1]
                
                mux_row = get_true_mux_row_output(next_layer['row'], row_term, next_h, stride_next, kernel_size_next, sum(padding_next))
                mux_sel = math.ceil(min(mux_row, math.ceil(math.log2(next_layer['row']+1))))
                total_mux_input = mux_row + (mux_sel if mux_row>1 else 0)
                if total_mux_input <= 1:
                    output_lut_per_bit = 0
                else:
                    output_lut_per_bit = math.ceil((total_mux_input-1)/5)
                
                out_channel = layer.get('out_channel', 1)
                if layer['type'] == 'lut_quant':
                    quant_channels = layer.get('quant_channels', 1)
                    output_total_bits = ((w_next-1)*stride_next+kernel_size_next) * ((next_h-1)*stride_next+kernel_size_next) * out_channel * quant_channels
                else:
                    output_total_bits = ((w_next-1)*stride_next+kernel_size_next) * ((next_h-1)*stride_next+kernel_size_next) * out_channel
                output_lut = output_lut_per_bit * output_total_bits
                layer['opt_output_lut_usage'] = int(output_lut)
                total_lut_used += int(output_lut)
            else:
                layer['opt_output_lut_usage'] = 0
            
            pct_reg = layer.get('pct_reg_num', 0) * t_vals_per_w[i][idx]
            
            if required_rows <= row_count:
                log_reg = 0
                log_reg = math.ceil(math.log2(next_layer['col']+1)) * 2 + math.ceil(math.ceil(math.log2(next_layer['row']+1)) * 2.5) 
                base_reg = base_reg + log_reg
            
            base_reg = base_reg + pct_reg 

        else: 
            layer['opt_input_lut_usage'] = 0
            layer['opt_output_lut_usage'] = 0
            
            base_reg = layer.get('reg_num', 0)
            pct_reg = layer.get('pct_reg_num', 0) * t_vals_per_w[i][idx]
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
                if layer['type'] == 'lut_quant':
                    reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                    expr2 = h_choice * reg_per_row
                else: 
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
                
            col_i = layer['col']
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            row_term = base_reg // reg_per_row
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
            
            h_next = h_config[i+1]
            if h_next != layer['row']//layer['stride']: 
                next_layer = base_data['layers'][i+1]
                kernel_size_next = next_layer.get('kernel_size', 3)
                stride_next = next_layer.get('stride', 1)
                padding_next = next_layer.get('padding', [1,1])
                w_next = 1  
                
                mux_row = get_true_mux_row_output(next_layer['row'], row_term, next_h, stride_next, kernel_size_next, sum(padding_next))
                mux_sel = math.ceil(min(mux_row, math.ceil(math.log2(next_layer['row']+1))))
                total_mux_input = mux_row + (mux_sel if mux_row>1 else 0)
                if total_mux_input <= 1:
                    output_lut_per_bit = 0
                else:
                    output_lut_per_bit = math.ceil((total_mux_input-1)/5)
                
                out_channel = layer.get('out_channel', 1)
                if layer['type'] == 'lut_quant':
                    quant_channels = layer.get('quant_channels', 1)
                    output_total_bits = ((w_next-1)*stride_next+kernel_size_next) * ((next_h-1)*stride_next+kernel_size_next) * out_channel * quant_channels
                else:
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
                bottom_padding = padding[1] if len(padding) > 1 else 0  
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
                h_divs = [1]
            else:
                h_divs = divisors(row_stride)
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
        exit()
        
    best_solution_data = solution_data

    iteration = 0
    improved_last_iteration = True 
    
    while improved_last_iteration:
        iteration += 1
        
        improved_in_this_iteration = False 
        
        iteration_best_h = list(h_current_best) 
        iteration_best_latency = latency_current_best
        iteration_best_data = best_solution_data

        for i in range(LAYER_NUM):
            for h_test_option in h_options_per_layer[i]:
                
                if h_test_option == h_current_best[i]:
                    continue
                    
                h_test_config = list(h_current_best) 
                h_test_config[i] = h_test_option
                
                
                if not check_resource_pruning(base_data, limit, h_test_config):
                    continue 
                    
                L_lower_bound = check_latency_pruning(base_data, h_test_config)
                if L_lower_bound > (latency_current_best - 1e-5):
                    continue 
                
                data = json_parser(json_path) 
                solution_data = solve(copy.deepcopy(data), limit, h_test_config)
                
                if (solution_data and 
                    solution_data.get('opt_solver_status') in ['optimal', 'optimal_inaccurate']):
                    current_latency = solution_data['opt_total_latency']
                    if current_latency < (iteration_best_latency - 1e-5):
                        iteration_best_latency = current_latency
                        iteration_best_h = h_test_config
                        iteration_best_data = solution_data 
                        improved_in_this_iteration = True 
                else:
                    status = "failed (invalid)"
                    if solution_data:
                         status = solution_data.get('opt_solver_status', 'failed')

        
        if improved_in_this_iteration:
             h_current_best = iteration_best_h
             latency_current_best = iteration_best_latency
             best_solution_data = iteration_best_data
        else:
            improved_last_iteration = False 
    end_time = time.time()
    elapsed_time = end_time - start_time


    
    if best_solution_data:
        for i, layer in enumerate(best_solution_data['layers']):
            lut_info = f"LUT={layer.get('opt_lut_usage', 0)}"
            if layer.get('opt_input_lut_usage', 0) > 0:
                lut_info += f"+input_lut={layer['opt_input_lut_usage']}"
            if layer.get('opt_output_lut_usage', 0) > 0:
                lut_info += f"+output_lut={layer['opt_output_lut_usage']}"
            
        output_path = "model_execution_info_optimized.json"
        with open(output_path, "w") as f:
            json.dump(best_solution_data, f, indent=2)
        return best_solution_data
    else:
        return None

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
                h_divs = [1]
            else:
                h_divs = divisors(row_stride)
            h_options_per_layer.append([h for h in h_divs if h > 0])
        else:
            h_options_per_layer.append([1]) 

    h_current_best = [options[0] for options in h_options_per_layer] 
    
    data = json_parser(json_path) 
    solution_data = solve(copy.deepcopy(data), limit, h_current_best)
    
    if (solution_data and 
        solution_data.get('opt_solver_status') in ['optimal', 'optimal_inaccurate']):
        
        latency_current_best = solution_data['opt_total_latency']
    else:
        return None
        
    best_solution_data = solution_data
    
    visited_set = set()
    visited_set.add(tuple(h_current_best))

    iteration = 0
    
    while True: 
        iteration += 1
        
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
                    
                
                if not check_resource_pruning(base_data, limit, h_test_config):
                    visited_set.add(tuple(h_test_config)) 
                    continue 
                    
                L_lower_bound = check_latency_pruning(base_data, h_test_config)
                if L_lower_bound > (latency_current_best + 1e-5): 
                    visited_set.add(tuple(h_test_config)) 
                    continue 
                
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

        if h_current_best == iteration_best_h:
            break 
        else:
            h_current_best = iteration_best_h
            latency_current_best = iteration_best_latency
            best_solution_data = iteration_best_data
        
    end_time = time.time()
    elapsed_time = end_time - start_time


    
    if best_solution_data:
        for i, layer in enumerate(best_solution_data['layers']):
            lut_info = f"LUT={layer.get('opt_lut_usage', 0)}"
            if layer.get('opt_input_lut_usage', 0) > 0:
                lut_info += f"+input_lut={layer['opt_input_lut_usage']}"
            if layer.get('opt_output_lut_usage', 0) > 0:
                lut_info += f"+output_lut={layer['opt_output_lut_usage']}"
            

        output_path = "model_execution_info_optimized.json"
        with open(output_path, "w") as f:
            json.dump(best_solution_data, f, indent=2)
        return best_solution_data

    else:
        return None

if __name__ == "__main__":
    json_path = "model_execution_info_6g.json"
    
    limit = {
        "LUT": 2000000,
        "FF":  4000000
    }

    heuristic_h_search_best(json_path, limit)
