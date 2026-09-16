# todo
# 1. add residual connection LUT
import json
import math
import itertools
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

def to_scalar(value, default=0.0):
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])

def solve(data, limit, h_config, fixed_w_config=None, verbose=True):
    LAYER_NUM = len(data['layers'])

    if fixed_w_config is not None and len(fixed_w_config) != LAYER_NUM:
        data['opt_solver_status'] = 'invalid_fixed_w_config_length'
        return data
    
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
            if fixed_w_config is not None:
                try:
                    fixed_w = int(fixed_w_config[i])
                except (TypeError, ValueError):
                    data['opt_solver_status'] = f'invalid_fixed_w_layer_{i}'
                    return data
                if fixed_w not in w_divs:
                    data['opt_solver_status'] = f'invalid_fixed_w_layer_{i}'
                    return data
                w_candidates = [fixed_w]
            else:
                w_candidates = w_divs

            for w_ in w_candidates:
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

            if fixed_w_config is not None:
                try:
                    fixed_w = int(fixed_w_config[i])
                except (TypeError, ValueError):
                    data['opt_solver_status'] = f'invalid_fixed_w_layer_{i}'
                    return data
                if fixed_w != 1:
                    data['opt_solver_status'] = f'invalid_fixed_w_layer_{i}'
                    return data

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
            next_layer = data['layers'][i+1]
            assert i < LAYER_NUM - 1, "last layer cannot have variable reg"
            h_next = h_config[i+1] 
            
            kernel_size_next = next_layer.get('kernel_size', 3)
            stride_next = next_layer.get('stride', 1)
            required_rows = max(2 * h_i, (2 * h_next - 1) * stride_next + kernel_size_next)
            res_required_rows = 0
            if layer['type'] == 'lut_res':
                res_required_rows = max(h_i, h_next * stride_next) * 2
            row_count = layer['row'] // layer['stride']  
            
            if layer['type'] == 'lut_quant':
                reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                res_reg_per_row = 0
            else: # lut_conv / lut_res
                reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel']
                res_reg_per_row = reg_per_row * layer['res_bit_width'] if layer['type'] == 'lut_res' else 0
            if required_rows <= row_count:
                reg_base = required_rows * reg_per_row
                padding_term = sum(next_layer.get('padding',[0,0])) * required_rows * next_layer.get("in_channel", 1)
            else:
                reg_base = row_count * reg_per_row
                padding_term = sum(next_layer.get('padding',[0,0])) * row_count * next_layer.get("in_channel", 1)
            if res_required_rows <= row_count:
                reg_base_res = res_required_rows * res_reg_per_row
            else:
                reg_base_res = row_count * res_reg_per_row
             
            # [!!!] 需求 1: 输入 LUT [!!!]
            col_i = layer['col']
            next_col_i = next_layer['col']
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            next_stride_i = next_layer.get('stride', 1)
            row_term = reg_base//reg_per_row
            mux_w_row = get_true_mux_row_input(next_layer['row'], row_term, h_i)
            if layer['type'] == 'lut_res':
                row_term_res = reg_base_res//res_reg_per_row
                mux_w_row_res = get_true_mux_row_input(next_layer['row'], row_term_res, h_i)
            else:
                row_term_res = 0
                mux_w_row_res = 0
            input_lut_list = []
            for w_val in w_vals_per_w[i+1]:
                # mux_w_col = col_i//w_val
                mux_w_col = 1
                total_mux_inputs = mux_w_row  + mux_w_col + 1 # the 1 is vertical shift
                if total_mux_inputs <= 1:
                    lut_per_reg = 0
                else:
                    lut_per_reg = math.ceil((total_mux_inputs-1)/5)

                if layer['type'] == 'lut_res':
                    total_mux_inputs_res = mux_w_row_res + mux_w_col + 1
                    if total_mux_inputs_res <= 1:
                        lut_per_reg_res = 0
                    else:
                        lut_per_reg_res = math.ceil((total_mux_inputs_res-1)/5)
                else:
                    lut_per_reg_res = 0

                input_lut_list.append(reg_base * lut_per_reg + reg_base_res * lut_per_reg_res)

            input_lut_count = cp.sum(cp.multiply(np.array(input_lut_list), y[i+1]))
            lut_total_expr += input_lut_count

            reg_base += reg_base_res
            reg_base += padding_term
            pct_reg_term = layer.get('pct_reg_num', 0) * cp.sum(cp.multiply(t_vals_per_w[i], y[i]))
            
            if required_rows <= row_count:
                # 情况1：行数足够，计算 log
                log_term = math.ceil(math.log2(layer['col']//layer['stride']+1)) * 2 + math.ceil(math.ceil(math.log2(layer['row']//layer['stride']+1)) * 2.5)
                reg_total_expr += reg_base + pct_reg_term + log_term
            else:
                reg_total_expr += reg_base + pct_reg_term

        else: 
            base_reg = layer.get('reg_num', 0)
            pct_reg_term = layer.get('pct_reg_num', 0) * cp.sum(cp.multiply(t_vals_per_w[i], y[i]))
            
            if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
                log_term = 0
                if layer['col'] > 0 and layer['row'] > 0:
                    log_term = math.ceil(math.log2(layer['col']//layer['stride']+1)) * 2 + math.ceil(math.ceil(math.log2(layer['row']//layer['stride']+1)) * 2.5)
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
        if verbose:
            print(f"Stage 1 failed with status: {prob_stage1.status}")
        data['opt_solver_status'] = f'stage1_failed_{prob_stage1.status}'
        return data 
    
    optimal_latency = prob_stage1.value
    if verbose:
        print(f"Optimal latency found: {optimal_latency:.8f}")
    
    if verbose:
        print("Stage 2: Minimizing resources with fixed latency...")
    latency_tolerance = 1e-5 
    latency_constraint = total_latency <= optimal_latency + latency_tolerance
    stage2_constraints = constraints + [latency_constraint]
    
    normalized_lut = lut_total_expr / limit['LUT']
    normalized_reg = reg_total_expr / limit['FF']
    resource_objective = normalized_lut + normalized_reg
    
    prob = cp.Problem(cp.Minimize(resource_objective), stage2_constraints)
    prob.solve(solver=cp.GUROBI, verbose=False, NumericFocus=3, reoptimize=True) 

    if verbose:
        print("status:", prob.status)
        if prob.value is not None:
            print(f"obj: {prob.value:.5f}")
        else:
            print("obj: (not found)")

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        if verbose:
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
            if verbose:
                print(f"Warning: No solution found for layer {i}")
            idx = 0 
            w_choice = 1
            if w_vals_per_w[i].size > 0:
                 w_choice = w_vals_per_w[i][0]
        else:
            idx = int(np.argmax(ys))
            if idx >= len(w_vals_per_w[i]): 
                if verbose:
                    print(f"Warning: Index {idx} out of bounds for w_vals layer {i}")
                idx = 0
            w_choice = int(w_vals_per_w[i][idx])
            
        w_choices.append(w_choice)

        layer['opt_w'] = w_choice
        layer['opt_h'] = h_choice
        
        if L_vars[i].value is not None:
            layer['opt_total_latency'] = to_scalar(L_vars[i].value)
        if C_vars[i].value is not None:
            layer['opt_computation_time'] = to_scalar(C_vars[i].value)
        if S_vars[i].value is not None:
            layer['opt_start_time'] = int(round(to_scalar(S_vars[i].value)))
        if W_vars[i].value is not None:
            layer['opt_wait_time'] = to_scalar(W_vars[i].value)
        if T_comp_scalar_vars[i].value is not None:
            layer['opt_chunk_compute_time'] = to_scalar(T_comp_scalar_vars[i].value)

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
            next_layer = data['layers'][i+1]
            next_h = h_config[i+1]
            
            kernel_size_next = next_layer.get('kernel_size', 3)
            stride_next = next_layer.get('stride', 1)
            required_rows = max(2 * h_choice, (2 * next_h - 1) * stride_next + kernel_size_next)
            res_required_rows = 0
            if layer['type'] == 'lut_res':
                res_required_rows = max(2 * h_choice, 2 * next_h * stride_next)
            row_count = layer['row']// layer['stride']

            if layer['type'] == 'lut_quant':
                reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                res_reg_per_row = 0
            else: # lut_conv / lut_res
                reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel']
                res_reg_per_row = reg_per_row * layer['res_bit_width'] if layer['type'] == 'lut_res' else 0
            if required_rows <= row_count:
                base_reg = required_rows * reg_per_row
                padding_term = sum(next_layer.get('padding',[0,0])) * required_rows * next_layer['in_channel']
            else:
                base_reg = row_count * reg_per_row
                padding_term = sum(next_layer.get('padding',[0,0])) * row_count * next_layer['in_channel']
            if res_required_rows <= row_count:
                base_reg_res = res_required_rows * res_reg_per_row
            else:
                base_reg_res = row_count * res_reg_per_row
                
            # 计算 input_lut 系数
            col_i = layer['col']
            next_col_i = next_layer['col']
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            next_stride_i = next_layer.get('stride', 1)
            row_term = base_reg // reg_per_row
            mux_w_row = get_true_mux_row_input(next_layer['row'], row_term, h_choice)
            # mux_w_col = col_i//w_choice
            mux_w_col = 1
            total_mux_inputs = mux_w_row + mux_w_col + 1 # the 1 is vertical shift
            if total_mux_inputs <= 1:
                lut_per_reg = 0
            else:
                lut_per_reg = math.ceil((total_mux_inputs-1)/5)
            input_lut = lut_per_reg * base_reg

            if layer['type'] == 'lut_res':
                row_term_res = base_reg_res // res_reg_per_row
                mux_w_row_res = get_true_mux_row_input(next_layer['row'], row_term_res, h_choice)
                total_mux_inputs_res = mux_w_row_res + mux_w_col + 1 # the 1 is vertical shift
                if total_mux_inputs_res <= 1:
                    lut_per_reg_res = 0
                else:
                    lut_per_reg_res = math.ceil((total_mux_inputs_res-1)/5)
                input_lut_res = lut_per_reg_res * base_reg_res
            else:
                row_term_res = 0
                mux_w_row_res = 0
                input_lut_res = 0

            base_reg += padding_term
            
            layer['opt_input_lut_usage'] = int(input_lut)
            layer['opt_input_lut_usage_res'] = int(input_lut_res)
            total_lut_used += int(input_lut+input_lut_res)
            
            pct_reg = layer.get('pct_reg_num', 0) * t_vals_per_w[i][idx]
            
            if required_rows <= row_count:
                log_reg = 0
                log_reg = math.ceil(math.log2(next_layer['col']+1)) * 2 + math.ceil(math.ceil(math.log2(next_layer['row']+1)) * 2.5) # addrs. 2 is col, 2.5 is row
                # print("base_reg: "+str(base_reg)+" pct_reg: "+str(pct_reg)+" log_reg: "+str(log_reg)+" col: "+str(next_layer['col'])+" row: "+str(next_layer['row']))
                base_reg = base_reg + log_reg
            
            base_reg = base_reg + pct_reg + base_reg_res

        else: 
            layer['opt_input_lut_usage'] = 0
            base_reg = layer.get('reg_num', 0)
            pct_reg = layer.get('pct_reg_num', 0) * t_vals_per_w[i][idx]
            # print("base_reg: "+str(base_reg)+" pct_reg: "+str(pct_reg)+" log_reg: "+str(0))
            base_reg = base_reg + pct_reg
            
            if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
                if layer['col'] > 0 and layer['row'] > 0:
                    log_term = math.ceil(math.log2(layer['col']//layer['stride']+1)) * 2 + math.ceil(math.ceil(math.log2(layer['row']//layer['stride']+1)) * 2.5)
                    base_reg += log_term

        if layer['type'] == 'lut_fc' and i == LAYER_NUM - 1:
            grp_sum = math.ceil(math.log2(layer['lut_num']//10))*(10+4+1)
            grp_sum += math.ceil(math.log2(10))
            base_reg += grp_sum

        layer['opt_reg_usage'] = int(base_reg)
        total_reg_used += layer['opt_reg_usage']

    data['opt_final_computation_time'] = to_scalar(C_vars[-1].value)
    data['opt_final_total_latency'] = to_scalar(L_vars[-1].value)
    data['opt_final_start_time'] = int(round(to_scalar(S_vars[-1].value)))
    data['opt_final_wait_time'] = to_scalar(W_vars[-1].value)
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
            next_layer = base_data['layers'][i+1]
            next_h = h_config[i+1]
            
            kernel_size_next = next_layer.get('kernel_size', 3)
            stride_next = next_layer.get('stride', 1)
            required_rows = max(2 * h_choice, (2 * next_h - 1) * stride_next + kernel_size_next)
            res_required_rows = 0
            if layer['type'] == 'lut_res':
                res_required_rows = max(2 * h_choice, 2 * next_h * stride_next)
            row_count = layer['row'] // layer['stride']
            
            if layer['type'] == 'lut_quant':
                reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
                res_reg_per_row = 0
            else: # lut_conv / lut_res
                reg_per_row = (layer['col'] // layer['stride']) * layer['out_channel']
                res_reg_per_row = reg_per_row * layer['res_bit_width'] if layer['type'] == 'lut_res' else 0
            if required_rows <= row_count:
                base_reg = required_rows * reg_per_row
                padding_term = sum(next_layer.get('padding',[0,0])) * required_rows * next_layer.get('in_channel', 1)
            else:
                base_reg = row_count * reg_per_row
                padding_term = sum(next_layer.get('padding',[0,0])) * row_count * next_layer.get('in_channel', 1)
            if res_required_rows <= row_count:
                base_reg_res = res_required_rows * res_reg_per_row
            else:
                base_reg_res = row_count * res_reg_per_row
                
            # 计算 input_lut 系数
            col_i = layer['col']
            kernel_size_i = layer.get('kernel_size', 3)
            stride_i = layer.get('stride', 1)
            row_term = base_reg // reg_per_row
            mux_w_row = get_true_mux_row_input(next_layer['row'], row_term, h_choice)
            # mux_w_col = col_i // w_choice
            mux_w_col = 1
            total_mux_inputs = mux_w_row + mux_w_col + 1 # the 1 is vertical shift
            if total_mux_inputs <= 1:
                lut_per_reg = 0
            else:
                lut_per_reg = math.ceil((total_mux_inputs-1)/5)
            input_lut = lut_per_reg * base_reg
            if layer['type'] == 'lut_res':
                row_term_res = base_reg_res // res_reg_per_row
                mux_w_row_res = get_true_mux_row_input(next_layer['row'], row_term_res, h_choice)
                total_mux_inputs_res = mux_w_row_res + mux_w_col + 1 # the 1 is vertical shift
                if total_mux_inputs_res <= 1:
                    lut_per_reg_res = 0
                else:
                    lut_per_reg_res = math.ceil((total_mux_inputs_res-1)/5)
                input_lut_res = lut_per_reg_res * base_reg_res
            else:
                row_term_res = 0
                mux_w_row_res = 0
                input_lut_res = 0

            base_reg = base_reg + padding_term + base_reg_res
        
            total_lut_base += input_lut + input_lut_res
            
            pct_reg = layer.get('pct_reg_num', 0) * t_choice
            
            if required_rows <= row_count:
                log_reg = 0
                log_reg = math.ceil(math.log2(layer['col']//layer['stride']+1)) * 2 + math.ceil(math.ceil(math.log2(layer['row']//layer['stride']+1)) * 2.5)
                base_reg = base_reg + log_reg
            
            base_reg = base_reg + pct_reg

        else: 
            base_reg = layer.get('reg_num', 0)
            pct_reg = layer.get('pct_reg_num', 0) * t_choice
            base_reg = base_reg + pct_reg
            if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
                if layer['col'] > 0 and layer['row'] > 0:
                    log_term = math.ceil(math.log2(layer['col']//layer['stride']+1)) * 2 + math.ceil(math.ceil(math.log2(layer['row']//layer['stride']+1)) * 2.5)
                    base_reg += log_term
            
        if layer['type'] == 'lut_fc' and i == LAYER_NUM - 1:
            grp_sum = math.ceil(math.log2(layer['lut_num']//10))*(10+4+1)
            grp_sum += math.ceil(math.log2(10))
            base_reg += grp_sum

        total_reg_base += int(base_reg)

    return total_lut_base <= limit['LUT'] and total_reg_base <= limit['FF']

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

def build_h_options_per_layer(base_data, verbose=True):
    h_options_per_layer = []
    for i, layer in enumerate(base_data['layers']):
        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            row_stride = layer['row'] // layer['stride']
            if row_stride <= 0:
                if verbose:
                    print(f"Warning: layer {i} row/stride <= 0. Falling back to [1].")
                h_divs = [1]
            else:
                h_divs = divisors(row_stride)
            h_options_per_layer.append([h for h in h_divs if h > 0])
        else:
            h_options_per_layer.append([1])
    return h_options_per_layer

def build_w_options_per_layer(base_data, verbose=True):
    w_options_per_layer = []
    for i, layer in enumerate(base_data['layers']):
        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            col_stride = layer['col'] // layer['stride']
            if col_stride <= 0:
                if verbose:
                    print(f"Warning: layer {i} col/stride <= 0. Falling back to [1].")
                w_divs = [1]
            else:
                w_divs = divisors(col_stride)
            w_options_per_layer.append([w for w in w_divs if w > 0])
        else:
            w_options_per_layer.append([1])
    return w_options_per_layer

def extract_pareto_points(points):
    """
    2D Pareto for minimization on (latency, lut).
    Returns non-dominated points sorted by latency.
    """
    if not points:
        return []

    points_sorted = sorted(points, key=lambda p: (p['latency'], p['lut']))
    pareto = []
    best_lut = float('inf')

    for p in points_sorted:
        if p['lut'] < best_lut:
            pareto.append(p)
            best_lut = p['lut']

    return pareto

def exhaustive_pareto_plot(json_path, limit, force_no_packing=False, max_configs=None,
                          output_png='pareto_curve.png', show_plot=True, enumerate_w=True):
    """
    全量遍历配置并绘制 Pareto 曲线。

    默认遍历所有 (h, w) 组合；当 enumerate_w=False 时，仅遍历 h 并由 solve 自动选择 w。

    Args:
        json_path: JSON 配置文件路径
        limit: 资源限制字典，如 {"LUT": ..., "FF": ...}
        force_no_packing: 传递给 json_parser 的开关
        max_configs: 最多评估多少个配置（None 表示全部）
        output_png: 图像输出路径
        show_plot: 是否调用 plt.show()
        enumerate_w: 是否同时遍历 w（默认 True）

    Plot:
        主图会带一个 inset，自动放大左下角最密集区域。
    """
    base_data = json_parser(json_path, force_no_packing=force_no_packing)
    h_options_per_layer = build_h_options_per_layer(base_data, verbose=False)
    w_options_per_layer = build_w_options_per_layer(base_data, verbose=False)

    if enumerate_w:
        option_pairs_per_layer = []
        for h_opts, w_opts in zip(h_options_per_layer, w_options_per_layer):
            option_pairs_per_layer.append(list(itertools.product(h_opts, w_opts)))
        total_configs = 1
        for options in option_pairs_per_layer:
            total_configs *= len(options)
    else:
        option_pairs_per_layer = [list((h, 1) for h in h_opts) for h_opts in h_options_per_layer]
        total_configs = 1
        for options in h_options_per_layer:
            total_configs *= len(options)

    print("=== Exhaustive Pareto Search ===")
    if enumerate_w:
        print(f"Total h*w-config combinations: {total_configs}")
    else:
        print(f"Total h-config combinations: {total_configs}")
    if max_configs is not None:
        print(f"Evaluation cap enabled: max_configs={max_configs}")

    template_data = json_parser(json_path, force_no_packing=force_no_packing)
    all_points = []
    feasible_count = 0
    failed_count = 0

    start_time = time.time()
    progress_interval = max(1, total_configs // 50)
    for idx, hw_tuple in enumerate(itertools.product(*option_pairs_per_layer), start=1):
        if max_configs is not None and idx > max_configs:
            break

        if idx == 1 or idx % progress_interval == 0:
            if enumerate_w:
                h_progress = [p[0] for p in hw_tuple]
                w_progress = [p[1] for p in hw_tuple]
                print(f"[{idx}] evaluating h={h_progress}, w={w_progress}")
            else:
                h_progress = [p[0] for p in hw_tuple]
                print(f"[{idx}] evaluating h_config={h_progress}")

        h_config = [p[0] for p in hw_tuple]
        w_config = [p[1] for p in hw_tuple]
        fixed_w = w_config if enumerate_w else None

        solution_data = solve(
            copy.deepcopy(template_data),
            limit,
            h_config,
            fixed_w_config=fixed_w,
            verbose=False
        )

        if (solution_data and
            solution_data.get('opt_solver_status') in ['optimal', 'optimal_inaccurate']):
            if enumerate_w:
                selected_w = w_config
            else:
                selected_w = [layer.get('opt_w', 1) for layer in solution_data.get('layers', [])]
            all_points.append({
                'h_config': h_config,
                'w_config': selected_w,
                'latency': float(solution_data.get('opt_total_latency', float('inf'))),
                'lut': int(solution_data.get('opt_total_lut_used', 0)),
                'ff': int(solution_data.get('opt_total_reg_used', 0))
            })
            feasible_count += 1
        else:
            failed_count += 1

    elapsed = time.time() - start_time
    print(f"Search finished in {elapsed:.2f}s")
    print(f"Feasible points: {feasible_count}, failed/infeasible: {failed_count}")

    if not all_points:
        print("No feasible points found. Skip plotting.")
        return {
            'all_points': [],
            'pareto_points': []
        }

    pareto_points = extract_pareto_points(all_points)
    print(f"Pareto front size: {len(pareto_points)}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install it with: pip install matplotlib")
        return {
            'all_points': all_points,
            'pareto_points': pareto_points
        }

    pareto_points_for_plot = sorted(pareto_points, key=lambda p: (p['lut'], p['latency']))

    x_all = [p['lut'] for p in all_points]
    y_all = [p['latency'] for p in all_points]
    x_p = [p['lut'] for p in pareto_points_for_plot]
    y_p = [p['latency'] for p in pareto_points_for_plot]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_all, y_all, s=22, alpha=0.45, color='tab:blue', label='Feasible points')
    ax.plot(x_p, y_p, color='tab:red', marker='o', markersize=4, linewidth=2, label='Pareto front')
    ax.set_xlabel('LUT usage')
    ax.set_ylabel('Estimated latency')
    ax.set_title('LUT vs Latency Pareto Curve')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend()

    # Build an inset for the densest zone in the lower-left corner.
    if len(x_all) >= 20:
        x_arr = np.asarray(x_all, dtype=float)
        y_arr = np.asarray(y_all, dtype=float)

        x_ll_max = float(np.quantile(x_arr, 0.45))
        y_ll_max = float(np.quantile(y_arr, 0.45))
        ll_mask = (x_arr <= x_ll_max) & (y_arr <= y_ll_max)

        if int(np.count_nonzero(ll_mask)) >= 8:
            x_focus = x_arr[ll_mask]
            y_focus = y_arr[ll_mask]
        else:
            x_focus = x_arr
            y_focus = y_arr

        bins = max(8, min(40, int(math.sqrt(len(x_focus)))))
        hist, x_edges, y_edges = np.histogram2d(x_focus, y_focus, bins=bins)
        max_bin = np.unravel_index(np.argmax(hist), hist.shape)
        i_bin, j_bin = int(max_bin[0]), int(max_bin[1])

        x_left_idx = max(i_bin - 1, 0)
        x_right_idx = min(i_bin + 2, len(x_edges) - 1)
        y_bottom_idx = max(j_bin - 1, 0)
        y_top_idx = min(j_bin + 2, len(y_edges) - 1)

        x_min_zoom = float(x_edges[x_left_idx])
        x_max_zoom = float(x_edges[x_right_idx])
        y_min_zoom = float(y_edges[y_bottom_idx])
        y_max_zoom = float(y_edges[y_top_idx])

        if x_max_zoom <= x_min_zoom:
            x_span = max(1.0, float(np.ptp(x_arr)) * 0.05)
            x_center = float(np.median(x_focus))
            x_min_zoom = x_center - x_span
            x_max_zoom = x_center + x_span
        if y_max_zoom <= y_min_zoom:
            y_span = max(1.0, float(np.ptp(y_arr)) * 0.05)
            y_center = float(np.median(y_focus))
            y_min_zoom = y_center - y_span
            y_max_zoom = y_center + y_span

        axins = ax.inset_axes([0.56, 0.08, 0.4, 0.4])
        axins.scatter(x_all, y_all, s=14, alpha=0.5, color='tab:blue')
        axins.plot(x_p, y_p, color='tab:red', marker='o', markersize=3, linewidth=1.4)
        axins.set_xlim(x_min_zoom, x_max_zoom)
        axins.set_ylim(y_min_zoom, y_max_zoom)
        axins.set_title('Zoom: dense lower-left', fontsize=9)
        axins.grid(True, linestyle=':', alpha=0.25)

        try:
            ax.indicate_inset_zoom(axins, edgecolor='black', alpha=0.6)
        except Exception:
            pass

    fig.tight_layout()

    fig.savefig(output_png, dpi=200)
    print(f"Pareto figure saved to: {output_png}")

    if show_plot:
        try:
            plt.show()
        except Exception as err:
            print(f"plt.show() failed: {err}")

    plt.close(fig)

    return {
        'all_points': all_points,
        'pareto_points': pareto_points
    }

def heuristic_h_search_best(json_path, limit, force_no_packing=False, mode='steepest', random_seed=None):
    """
    启发式搜索 - 带横向移动的爬山算法，使用随机起始点
    
    Args:
        json_path: JSON配置文件路径
        limit: 资源限制字典
        force_no_packing: 是否强制不使用 packing（默认 False）
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
    #h_current_best = [options[0] for options in h_options_per_layer] 
    h_current_best = [1, 2, 1, 1, 1, 1, 1, 1, 1]
    
    print(f"=== 启发式搜索启动 (模式: {mode}, 随机起始点) ===")
    print(f"随机选择的起始 H_CONFIG: {h_current_best}")
    data = json_parser(json_path, force_no_packing=force_no_packing) 
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
                data = json_parser(json_path, force_no_packing=force_no_packing) 
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
    json_path = "model_execution_info.json"
    
    limit = {
        "LUT": 2000000,
        "FF":  4000000
    }

    run_mode = 'pareto'  # 'heuristic' or 'pareto'

    if run_mode == 'pareto':
        exhaustive_pareto_plot(
            json_path,
            limit,
            force_no_packing=False,
            max_configs=None,
            output_png='pareto_curve.png',
            show_plot=True,
            enumerate_w=True
        )
    else:
        heuristic_h_search_best(json_path, limit)
