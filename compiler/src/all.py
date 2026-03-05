import json
import math
import copy
from json_parser import json_parser
import time

def divisors(n):
    # 确保 n > 0
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

def evaluate_solution(data, w_config, h_config):
    """
    根据我们确定的物理模型，计算一个给定 (w, h) 完整配置的
    确切延迟和资源消耗。
    
    Args:
        data: 原始模型 JSON
        w_config (list): 每层的 w (int)
        h_config (list): 每层的 h (int)
        
    Returns:
        dict: 包含详细结果的字典, 或 None (如果配置无效)
    """
    
    print(f"--- 正在评估 H_CONFIG: {h_config}")
    print(f"--- 配合 W_CONFIG: {w_config}")
    
    LAYER_NUM = len(data['layers'])
    if len(w_config) != LAYER_NUM or len(h_config) != LAYER_NUM:
        print("!!! 错误: w_config 或 h_config 长度与层数不匹配。")
        return None

    # 创建一个副本以存储每层的详细结果
    eval_data = copy.deepcopy(data)
    
    # 存储计算出的下限
    T_comp_list = [0.0] * LAYER_NUM
    N_const_list = [0.0] * LAYER_NUM
    S_const_list = [0.0] * LAYER_NUM
    K_const_list = [0.0] * LAYER_NUM
    Ratio_const_list = [0.0] * LAYER_NUM
    D_list = [0.0] * LAYER_NUM
    
    W_list = [0.0] * LAYER_NUM
    C_list = [0.0] * LAYER_NUM
    S_list = [0.0] * LAYER_NUM
    
    # 1. 预计算所有常数 (基于固定的 w 和 h)
    for i, layer in enumerate(eval_data['layers']):
        h_i = h_config[i]
        w_i = w_config[i]
        D_list[i] = layer.get('additional_latency', 0)
        
        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            # N_i
            N_const_list[i] = (layer['row'] // layer['stride']) / float(h_i) if h_i > 0 else 1.0
            
            # T_comp_i (w 和 h 都是固定的)
            T_calc = (layer['col'] // layer['stride']) / float(w_i)
            T_comp = T_calc
            if i == 0: # lut_quant (第0层) 受 BW 限制
                data_per_chunk = (layer['col'] // layer['stride']) * h_i * layer.get('in_channel', 1)
                T_input_0 = data_per_chunk / float(eval_data.get('BW', 1))
                T_comp = max(T_calc, T_input_0)
            T_comp_list[i] = T_comp
            
            # S_0
            if i == 0:
                kernel_size_0 = layer.get('kernel_size', 3)
                stride_0 = layer.get('stride', 1)
                padding_0 = layer.get('padding', [0, 0])[0]
                required_rows_0 = kernel_size_0 + (h_i - 1) * stride_0 - padding_0
                required_rows_0 = max(1.0, required_rows_0)
                initial_data_0 = (layer['col'] // stride_0) * required_rows_0 * layer.get('in_channel', 1)
                S_const_list[i] = math.ceil(initial_data_0 / float(eval_data.get('BW', 1)))
            
            # K_i 和 Ratio_i
            if i > 0:
                h_prev = h_config[i-1]
                kernel_size_i = layer.get('kernel_size', 3)
                stride_i = layer.get('stride', 1)
                padding_i = layer.get('padding', [0, 0])[0]
                required_rows_i = kernel_size_i + (h_i - 1) * stride_i - padding_i
                required_rows_i = max(1.0, required_rows_i)
                
                K_const_list[i] = max(1.0, math.ceil(required_rows_i / h_prev)) if h_prev > 0 else 1.0
                Ratio_const_list[i] = (h_i * stride_i) / h_prev if h_prev > 0 else 1.0
            else:
                K_const_list[i] = 1.0
                Ratio_const_list[i] = 1.0
        
        # (FC 层所有值保持为 0 或 1)

    # 2. 迭代计算延迟 (确定性计算)
    prev_layer_type = 'init'
    for i, layer in enumerate(eval_data['layers']):
        D_i = D_list[i]
        D_prev = D_list[i-1] if i > 0 else 0

        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            # W_i
            W_i = T_comp_list[i] # W_i >= T_comp_i
            if i > 0:
                if prev_layer_type in ['lut_conv', 'lut_res', 'lut_quant']:
                    W_i = max(W_i, W_list[i-1] * Ratio_const_list[i])
            W_list[i] = W_i
            
            # C_i
            N_minus_1 = N_const_list[i] - 1
            C_i = N_minus_1 * W_i + T_comp_list[i]
            C_list[i] = C_i
            
            # S_i
            if i == 0:
                S_i = S_const_list[i]
            elif prev_layer_type in ['lut_fc']:
                S_i = S_list[i-1] + C_list[i-1] + D_prev
            else:
                K_minus_1 = K_const_list[i] - 1
                DataReadyTime = K_minus_1 * W_list[i-1] + T_comp_list[i-1]
                S_i = S_list[i-1] + DataReadyTime + D_prev
            S_list[i] = S_i
                
        elif layer['type'] in ['lut_fc']:
            # W_i, T_comp_i 保持为 0
            C_i = 4 if i == LAYER_NUM - 1 else 0
            C_list[i] = C_i
            
            S_i = 0
            if i > 0:
                S_i = S_list[i-1] + C_list[i-1] + D_prev
            S_list[i] = S_i
        
        prev_layer_type = layer['type']

    # 3. 计算资源
    total_lut = 0
    total_reg = 0
    
    for i, layer in enumerate(eval_data['layers']):
        w_choice = w_config[i]
        h_choice = h_config[i]
        t_choice = w_choice * h_choice

        # LUT
        if layer['type'] in ['lut_conv', 'lut_res', 'lut_quant']:
            layer_lut = layer['lut_num'] * t_choice
        else:
            layer_lut = layer.get('lut_num', 0)
        total_lut += layer_lut
        layer['opt_lut_usage'] = int(layer_lut)

        # REG (与 solve() 中的恢复逻辑完全相同)
        if layer.get('has_variable_reg', False) and i < LAYER_NUM - 1:
            next_h = h_config[i+1]
            expr1 = layer['reg_num'] + layer['reg_bias'] * next_h
            
            expr2 = 0
            if layer['type'] == 'lut_quant':
                expr2 = h_choice * (layer['col'] // layer['stride']) * layer['out_channel'] * layer['quant_channels']
            else: # lut_conv / lut_res
                expr2 = h_choice * (layer['col'] // layer['stride']) * layer['out_channel']
                
            base_reg = max(expr1, expr2) + expr2
            pct_reg = layer.get('pct_reg_num', 0) * t_choice
            
            if layer['type'] in ['lut_conv', 'lut_res']:
                 padding_reg = sum(layer['padding']) * (layer['row'] // layer['stride'])
                 log_reg = 0
                 if layer['col'] > 0 and layer['row'] > 0:
                     log_reg = math.log2(layer['col']) * 2 + math.ceil(math.log2(layer['row']) * 2.5)
                 base_reg = base_reg + padding_reg + log_reg
            
            base_reg = base_reg + pct_reg 

        else: # not has_variable_reg
            base_reg = layer.get('reg_num', 0)
            pct_reg = layer.get('pct_reg_num', 0) * t_choice
            base_reg = base_reg + pct_reg
            
            if layer['type'] in ['lut_conv', 'lut_res']:
                padding_reg = sum(layer['padding']) * (layer['row'] // layer['stride'])
                log_reg = 0
                if layer['col'] > 0 and layer['row'] > 0:
                    log_reg = math.log2(layer['col']) * 2 + math.ceil(math.log2(layer['row']) * 2.5)
                base_reg = base_reg + padding_reg + log_reg
        
        if layer['type'] == 'lut_fc' and i == LAYER_NUM - 1:
            grp_sum = math.ceil(math.log2(layer['lut_num']//10))*(10+4+1)
            grp_sum += math.ceil(math.log2(10))
            base_reg += grp_sum

        layer['opt_reg_usage'] = int(base_reg)
        total_reg += int(base_reg)
        
        # 存储时序结果
        layer['opt_w'] = w_choice
        layer['opt_h'] = h_choice
        layer['opt_start_time'] = int(round(S_list[i]))
        layer['opt_computation_time'] = C_list[i]
        layer['opt_wait_time'] = W_list[i]
        layer['opt_chunk_compute_time'] = T_comp_list[i]
        layer['opt_total_latency'] = C_list[i] + D_list[i]

    # 4. 存储全局结果
    L_last = C_list[-1] + D_list[-1]
    total_latency = S_list[-1] + L_last
    
    eval_data['opt_total_latency'] = total_latency
    eval_data['opt_total_lut_used'] = total_lut
    eval_data['opt_total_reg_used'] = total_reg
    
    return eval_data

# --- [!!!] 示例用法 [!!!] ---
# (你需要你的 json_parser 和 divisors 函数)
if __name__ == '__main__':
    json_path = "model_execution_info_6g.json"
    data = json_parser(json_path)

    start_time = time.time()
    for h_1 in range(5):
        for h_2 in range(4):
            for h_3 in range(4):
                for h_4 in range(3):
                    for h_5 in range(3):
                        for h_6 in range(2):
                            for h_7 in range(2):
                                for w_1 in range(5):
                                    for w_2 in range(4):
                                        for w_3 in range(4):
                                            for w_4 in range(3):
                                                for w_5 in range(3):
                                                    for w_6 in range(2):
                                                        for w_7 in range(2):
                                                            h = [h_1+1, h_2+1, h_3+1, h_4+1, h_5+1, h_6+1, h_7+1, 0, 0] 
                                                            w = [w_1+1, w_2+1, w_3+1, w_4+1, w_5+1, w_6+1, h_7+1, 0, 0]

                                                            eval_results = evaluate_solution(data, w, h)
    # h = [1,1,1,1,1,1,1,0,0]
    # w = [1,1,1,1,1,1,1,0,0]
    eval_results = evaluate_solution(data, w, h)
    end_time = time.time()
    print(f"\n评估完成，耗时 {end_time - start_time:.2f} 秒。")

    if eval_results:
        print("\n--- 评估结果 ---")
        for i, layer in enumerate(eval_results['layers']):
            print(f"layer {i}: type={layer['type']}, w={layer['opt_w']}, h={layer['opt_h']}, "
                  f"S={layer['opt_start_time']}, C={layer['opt_computation_time']:.3f}, "
                  f"W={layer['opt_wait_time']:.3f}, T_comp={layer['opt_chunk_compute_time']:.3f}")

        print(f"\n总延迟: {eval_results['opt_total_latency']:.4f}")
        print(f"总 LUT: {eval_results['opt_total_lut_used']}")
        print(f"总 FF: {eval_results['opt_total_reg_used']}")
        
        # 结果应该与你日志中的 L=20.0000 完全匹配