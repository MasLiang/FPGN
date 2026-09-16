import json
import math

def adder_tree(in_num, freq=200):
    period = 1000/freq 
    period_safe = period/2
    period_lut6 = 0.053     

    def adder_latency(bit_width):
        period_in = 0.269       
        period_lut6 = 0.053     
        period_s0co3 = 0.313    
        period_cico = [0.179, 0.132, 0.094, 0.058]
        period_cio = [0.139, 0.213, 0.136, 0.179] 

        if bit_width==3 or bit_width==4:
            return period_in + period_lut6
        elif bit_width==5:
            return period_in + period_lut6*2
        else:
            carry_chain_num = math.floor((bit_width-4)/4)
            out_idx = (bit_width-4)%4
            c_delay = period_in + period_lut6 + period_s0co3 + carry_chain_num*period_cico[3] + period_cico[out_idx]
            s_delay = period_in + period_lut6 + period_s0co3 + carry_chain_num*period_cico[3] + period_cio[out_idx]
            return max(c_delay, s_delay)

    data_num = in_num
    bit_width = 3
    period_left = period_safe 
    reg_num = 0
    lut_num = 0
    reg_idx = []
    add_idx = 0
    while data_num>1:
        adder_num = math.floor(data_num/2)
        lut_num_per_bit = 0
        #if bit_width==3:
        #    lut_num_per_bit = 1
        #elif bit_width==4:
        #    lut_num_per_bit = 3
        #else:
        lut_num_per_bit = bit_width
        lut_num += adder_num*lut_num_per_bit
        data_num = adder_num + data_num%2
        bit_width += 1
        add_idx += 1
    return lut_num, reg_num, reg_idx, bit_width

def orignal_json_parser(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    data['lut_size'] = 6
    data['bit_width'] = 8
    data['col'] = 32
    data['row'] = 32
    data['channel_in'] = 3
    data['out_class'] = 10
    conv_num = 0
    fc_num = 0

    for layer in data['layers']:
        if layer['type'] == 'BatchNorm2d' or layer['type'] == 'SyncBatchNorm':
            last_idx = data['layers'].index(layer) - 1
            last_layer = data['layers'][last_idx]
            if last_layer['type'] in ['lut_conv', 'lut_res']:
                last_layer['threshold'] = layer['threshold']
                data['layers'].remove(layer)
            elif last_layer['type'] == 'lut_quant':
                data['layers'].remove(layer)
        if layer['type'] == 'lut_quant':
            quant_channels = 1
            layer_idx = data['layers'].index(layer)
            layer['lut_weights'] = [layer['lut_weights']]
            next_layer = data['layers'][layer_idx + 1]
            while next_layer['type'] == 'lut_quant':
                quant_channels += 1
                layer['lut_weights'].append(next_layer['lut_weights'])
                data['layers'].remove(next_layer)
                next_layer = data['layers'][layer_idx + 1]
            layer['quant_channels'] = quant_channels

    in_channel = 0
    fc_idx = 0
    for layer in data['layers']:
        layer_idx = data['layers'].index(layer)
        if layer_idx==len(data['layers'])-1:
            next_layer = {'type': 'end'}
        else:
            next_layer = data['layers'][layer_idx + 1]
        if layer['type'] == 'lut_quant':
            layer['kernel_size'] = 1
            layer['stride'] = 1
            layer['padding'] = [0,0]
            layer['row'] = data['row']
            layer['col'] = data['col']
            res_bit_width = 0
            lut_num_tmp = data['bit_width']
            lut_num = 0
            while (lut_num_tmp>1):
                lut_num_tmp = math.ceil(lut_num_tmp/data['lut_size'])
                lut_num += lut_num_tmp
            lut_num = lut_num*layer['quant_channels']
            layer['lut_num'] = lut_num*layer['out_channel']
            if next_layer['type'] in ['lut_conv', 'lut_res']:
                mem_row = next_layer['kernel_size']  - next_layer['stride']
                reg_num = mem_row*next_layer['col']*layer['out_channel']*layer['quant_channels'] 
                layer['has_variable_reg'] = True
                layer['reg_bias'] = next_layer['stride'] *next_layer['col']*layer['out_channel']*layer['quant_channels']
            elif next_layer['type'] in ['lut_fc']:
                reg_num = layer['row']*layer['col']*layer['out_channel']*layer['quant_channels'] 
                layer['has_variable_reg'] = False
            layer['reg_num'] = reg_num
            in_channel = layer['out_channel']*layer['quant_channels']
        elif layer['type'] in ['lut_conv', 'lut_res']:
            layer['in_channel'] = in_channel
            lut_num = math.ceil(layer['kernel_size']*layer['kernel_size']*in_channel/data['lut_size'])
            pct_lut_nums = 0
            pct_reg_nums = 0
            pct_reg_idxs = []
            pct_in_num = 0
            previous_res_bit_width = res_bit_width
            if lut_num%6>1:
                pct_in_num = lut_num//6*3+2
                pct_reg_nums += (lut_num//6*3+2) * layer['out_channel']
                pct_lut_nums += (lut_num//6*3+2) * layer['out_channel']
                pct_in_num = lut_num//6+1
            elif lut_num%6==1:
                pct_reg_nums += (lut_num//6*3+1) * layer['out_channel']
                pct_lut_nums += 0
                pct_in_num = lut_num//6+1
            else:
                pct_reg_nums += (lut_num//6*3) * layer['out_channel']
                pct_lut_nums += (lut_num//6*3) * layer['out_channel']
                pct_in_num = lut_num//6
            for out_channel_idx in range(layer['out_channel']): 
                pct_lut_num, pct_reg_num, pct_reg_idx, res_bit_width = adder_tree(pct_in_num, freq=data['freq'])
                pct_lut_num += math.ceil(math.log2(layer['threshold'][out_channel_idx])) 
                pct_lut_nums += pct_lut_num
                pct_reg_nums += pct_reg_num
                pct_reg_idxs.append([0] + pct_reg_idx)
            
            layer['lut_num'] = math.ceil(lut_num*layer['out_channel']) + pct_lut_nums
            if layer['type'] == 'lut_res':
                layer['lut_num'] += max(previous_res_bit_width, res_bit_width) * layer['out_channel'] 
                res_bit_width = max(previous_res_bit_width, res_bit_width) + 1
                layer['res_bit_width'] = res_bit_width
            else:
                layer['res_bit_width'] = 0
            in_channel = layer['out_channel']
            if next_layer['type'] in ['lut_conv', 'lut_res']:
                mem_row = next_layer['kernel_size'] - next_layer['stride']
                reg_num = mem_row*next_layer['col']*layer['out_channel'] 
                layer['has_variable_reg'] = True
                layer['reg_bias'] = next_layer['stride'] *next_layer['col']*layer['out_channel']
            elif next_layer['type'] in ['lut_fc']:
                reg_num = (layer['col']//layer['stride'])*(layer['row']//layer['stride'])*layer['out_channel']
                layer['has_variable_reg'] = False
            layer['reg_num'] = reg_num
            layer['pct_reg_num'] = pct_reg_nums
            layer['pct_reg_idxs'] = pct_reg_idxs
            layer['additional_latency'] = max([len(_) for _ in pct_reg_idxs]) if len(pct_reg_idxs)>0 else 0

        elif layer['type'] == 'lut_fc':
            if fc_idx%4==0 and fc_idx>0:
                layer['reg_num'] = layer['lut_num']
            else:
                layer['reg_num'] = 0
            if next_layer['type'] != 'lut_fc':
                layer['reg_num'] = layer['lut_num']
                bit_pct = math.ceil(layer['lut_num']//10)

                pct_lut_nums = 0
                if bit_pct%6>1:
                    pct_lut_nums += (bit_pct//6*3+2)
                    pct_in_num = bit_pct//6+1
                elif bit_pct%6==1:
                    pct_lut_nums += (bit_pct//6*3+1)
                    pct_in_num = bit_pct//6+1
                else:
                    pct_lut_nums += (bit_pct//6*3)
                    pct_in_num = bit_pct//6
                pct_lut_num, _, _, _ = adder_tree(pct_in_num, freq=data['freq'])
                pct_lut_nums += pct_lut_num
                grp_sum = pct_lut_nums*10
                grp_sum += math.ceil(math.log2(10))
                layer['grp_sum_lut_num'] = grp_sum

    for layer in data['layers']:
        if layer['type'] in ['lut_conv', 'lut_res']:
            conv_num += 1
        elif layer['type'] == 'lut_fc':
            fc_num += 1
    data['conv_num'] = conv_num
    data['fc_num'] = fc_num

    return data

def _compress_6lut_to_5lut(lut_hex, active_inputs):
    # Keep entries with removed input fixed to 1, e.g., remove 5 -> [32:63], remove 0 -> odd indices.
    active_set = set(active_inputs)
    missing_inputs = [idx for idx in range(6) if idx not in active_set]
    if len(missing_inputs) == 0:
        value = int(lut_hex, 16) & 0xFFFFFFFF
        return f"{value:08x}"

    removed_input = missing_inputs[0]
    value = int(lut_hex, 16)
    compressed = 0
    out_idx = 0
    for src_idx in range(64):
        if ((src_idx >> removed_input) & 1) == 1:
            bit = (value >> src_idx) & 1
            compressed |= (bit << out_idx)
            out_idx += 1
    return f"{compressed:08x}"


def _build_channel_offsets(active_input_indices, lut_weights_len):
    if not active_input_indices:
        return []

    lengths = [len(indices_per_channel) for indices_per_channel in active_input_indices]
    sum_lengths = sum(lengths)
    if sum_lengths == lut_weights_len:
        offsets = [0]
        for ln in lengths[:-1]:
            offsets.append(offsets[-1] + ln)
        return offsets

    # Fallback: assume LUT tables are uniformly split by channel.
    channel_num = len(active_input_indices)
    per_channel = lut_weights_len // channel_num if channel_num > 0 else 0
    return [ch * per_channel for ch in range(channel_num)]


def _get_channel_lut_entry(channel, lut_local_idx, lut_weights, active_input_indices, channel_offsets):
    global_idx = channel_offsets[channel] + lut_local_idx
    lut_hex = lut_weights[global_idx]
    active_inputs = active_input_indices[channel][lut_local_idx]
    return lut_hex, active_inputs


def _normalize_packing_plan(packing_plan):
    if not packing_plan:
        return []
    if 'types' in packing_plan[0]:
        return packing_plan

    normalized = []
    current = None
    current_key = None
    for item in packing_plan:
        key = (item['u'], item['v'])
        if key != current_key:
            current = {'u': item['u'], 'v': item['v'], 'types': []}
            normalized.append(current)
            current_key = key

        l_idx = item.get('l', len(current['types']))
        while len(current['types']) <= l_idx:
            current['types'].append('DEAD')
        current['types'][l_idx] = item['type']

    return normalized


def pack_parser(packing_plan, lut_weights=None, active_input_indices=None):
    packing_plan = _normalize_packing_plan(packing_plan)

    # Compatibility mode: only count LUT usage.
    lut_num = 0
    for pair in packing_plan:
        pair_types = pair.get('types', [])
        for type_name in pair_types:
            if type_name == 'DEAD':
                continue
            if type_name == 'PACKED' or type_name.startswith('SINGLE_'):
                lut_num += 1
            else:
                lut_num += 2

    channel_offsets = _build_channel_offsets(active_input_indices, len(lut_weights))
    packed_lut_tables = []

    for pair in packing_plan:
        u = pair['u']
        v = pair['v']
        pair_types = pair.get('types', [])

        for lut_local_idx, type_name in enumerate(pair_types):
            if type_name == 'DEAD':
                packed_lut_tables.append('0000000000000000')
            elif type_name == 'PACKED':
                lut_u_hex, lut_u_active = _get_channel_lut_entry(
                    u, lut_local_idx, lut_weights, active_input_indices, channel_offsets
                )
                lut_v_hex, lut_v_active = _get_channel_lut_entry(
                    v, lut_local_idx, lut_weights, active_input_indices, channel_offsets
                )
                lut_u_32 = _compress_6lut_to_5lut(lut_u_hex, lut_u_active)
                lut_v_32 = _compress_6lut_to_5lut(lut_v_hex, lut_v_active)
                # Concatenate as u(upper 32b) + v(lower 32b).
                packed_lut_tables.append((lut_u_32 + lut_v_32).lower())
            elif type_name == 'SINGLE_U':
                lut_u_hex, _ = _get_channel_lut_entry(
                    u, lut_local_idx, lut_weights, active_input_indices, channel_offsets
                )
                packed_lut_tables.append(lut_u_hex.lower())
            elif type_name == 'SINGLE_V':
                lut_v_hex, _ = _get_channel_lut_entry(
                    v, lut_local_idx, lut_weights, active_input_indices, channel_offsets
                )
                packed_lut_tables.append(lut_v_hex.lower())
            else:
                lut_u_hex, _ = _get_channel_lut_entry(
                    u, lut_local_idx, lut_weights, active_input_indices, channel_offsets
                )
                lut_v_hex, _ = _get_channel_lut_entry(
                    v, lut_local_idx, lut_weights, active_input_indices, channel_offsets
                )
                packed_lut_tables.append([lut_u_hex.lower(), lut_v_hex.lower()])

    return lut_num, packed_lut_tables

def pruned_json_parser(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    data['lut_size'] = 6
    data['bit_width'] = 8
    data['col'] = 32
    data['row'] = 32
    data['channel_in'] = 3
    data['out_class'] = 10
    conv_num = 0
    fc_num = 0

    for layer in data['layers']:
        if layer['type'] == 'BatchNorm2d' or layer['type'] == 'SyncBatchNorm':
            last_idx = data['layers'].index(layer) - 1
            last_layer = data['layers'][last_idx]
            if last_layer['type'] in ['lut_conv', 'lut_res']:
                last_layer['threshold'] = layer['threshold']
                data['layers'].remove(layer)
            elif last_layer['type'] == 'lut_quant':
                data['layers'].remove(layer)
        if layer['type'] == 'lut_quant':
            quant_channels = 1
            layer_idx = data['layers'].index(layer)
            layer['lut_weights'] = [layer['lut_weights']]
            next_layer = data['layers'][layer_idx + 1]
            while next_layer['type'] == 'lut_quant':
                quant_channels += 1
                layer['lut_weights'].append(next_layer['lut_weights'])
                data['layers'].remove(next_layer)
                next_layer = data['layers'][layer_idx + 1]
            layer['quant_channels'] = quant_channels

    in_channel = 0
    fc_idx = 0
    for layer in data['layers']:
        layer_idx = data['layers'].index(layer)
        if layer_idx==len(data['layers'])-1:
            next_layer = {'type': 'end'}
        else:
            next_layer = data['layers'][layer_idx + 1]
        if layer['type'] == 'lut_quant':
            layer['kernel_size'] = 1
            layer['stride'] = 1
            layer['padding'] = [0,0]
            layer['row'] = data['row']
            layer['col'] = data['col']
            lut_num_tmp = data['bit_width']
            lut_num = 0
            while (lut_num_tmp>1):
                lut_num_tmp = math.ceil(lut_num_tmp/data['lut_size'])
                lut_num += lut_num_tmp
            lut_num = lut_num*layer['quant_channels']
            layer['lut_num'] = lut_num*layer['out_channel']
            if next_layer['type'] in ['lut_conv', 'lut_res']:
                mem_row = next_layer['kernel_size']  - next_layer['stride']
                reg_num = mem_row*next_layer['col']*layer['out_channel']*layer['quant_channels'] 
                layer['has_variable_reg'] = True
                layer['reg_bias'] = next_layer['stride'] *next_layer['col']*layer['out_channel']*layer['quant_channels']
            elif next_layer['type'] in ['lut_fc']:
                reg_num = layer['row']*layer['col']*layer['out_channel']*layer['quant_channels'] 
                layer['has_variable_reg'] = False
            layer['reg_num'] = reg_num
            in_channel = layer['out_channel']*layer['quant_channels']
            res_bit_width = 0
        if layer['type'] == 'BatchNorm2d' or layer['type'] == 'SyncBatchNorm':
            layer['has_variable_reg'] = False
            layer['reg_num'] = reg_num
            in_channel = layer['out_channel']*layer['quant_channels']
        elif layer['type'] in ['lut_conv','lut_res']:
            layer['in_channel'] = in_channel
            lut_num = math.ceil(layer['kernel_size']*layer['kernel_size']*in_channel/data['lut_size'])
            pct_lut_nums = 0
            pct_reg_nums = 0
            pct_reg_idxs = []
            pct_in_num = 0
            previous_res_bit_width = res_bit_width
            if lut_num%6>1:
                pct_in_num = lut_num//6*3+2
                pct_reg_nums += (lut_num//6*3+2) * layer['out_channel']
                pct_lut_nums += (lut_num//6*3+2) * layer['out_channel']
                pct_in_num = lut_num//6+1
            elif lut_num%6==1:
                pct_reg_nums += (lut_num//6*3+1) * layer['out_channel']
                pct_lut_nums += 0
                pct_in_num = lut_num//6+1
            else:
                pct_reg_nums += (lut_num//6*3) * layer['out_channel']
                pct_lut_nums += (lut_num//6*3) * layer['out_channel']
                pct_in_num = lut_num//6
            for out_channel_idx in range(layer['out_channel']): 
                pct_lut_num, pct_reg_num, pct_reg_idx, res_bit_width = adder_tree(pct_in_num, freq=data['freq'])
                pct_lut_num += math.ceil(math.log2(layer['threshold'][out_channel_idx])) 
                pct_lut_nums += pct_lut_num
                pct_reg_nums += pct_reg_num
                pct_reg_idxs.append([0] + pct_reg_idx)
            
            layer['lut_num'] = math.ceil(lut_num*layer['out_channel']) + pct_lut_nums
            if layer['type'] == 'lut_res':
                layer['lut_num'] += max(previous_res_bit_width, res_bit_width) * layer['out_channel'] 
                res_bit_width = max(previous_res_bit_width, res_bit_width) + 1
                layer['res_bit_width'] = res_bit_width
            else:
                layer['res_bit_width'] = 0
            in_channel = layer['out_channel']
            if next_layer['type'] in ['lut_conv', 'lut_res']:
                mem_row = next_layer['kernel_size'] - next_layer['stride']
                reg_num = mem_row*next_layer['col']*layer['out_channel'] 
                layer['has_variable_reg'] = True
                layer['reg_bias'] = next_layer['stride'] *next_layer['col']*layer['out_channel']
            elif next_layer['type'] in ['lut_fc']:
                reg_num = (layer['col']//layer['stride'])*(layer['row']//layer['stride'])*layer['out_channel']
                layer['has_variable_reg'] = False
            layer['reg_num'] = reg_num
            layer['pct_reg_num'] = pct_reg_nums
            layer['pct_reg_idxs'] = pct_reg_idxs
            layer['additional_latency'] = max([len(_) for _ in pct_reg_idxs]) if len(pct_reg_idxs)>0 else 0

            if 'packing_plan' in layer and 'active_input_indices' in layer and 'lut_weights' in layer:
                packed_lut_num, packed_lut_tables = pack_parser(
                    layer['packing_plan'],
                    layer['lut_weights'],
                    layer['active_input_indices']
                )
                raw_lut_vector_num = math.ceil(lut_num * layer['out_channel'])
                layer['lut_num'] = layer['lut_num'] - raw_lut_vector_num + packed_lut_num
                layer['lut_weights'] = packed_lut_tables
        elif layer['type'] == 'lut_fc':
            if fc_idx%4==0 and fc_idx>0:
                layer['reg_num'] = layer['lut_num']
            else:
                layer['reg_num'] = 0
            if next_layer['type'] != 'lut_fc':
                layer['reg_num'] = layer['lut_num']
                bit_pct = math.ceil(layer['lut_num']//10)

                pct_lut_nums = 0
                if bit_pct%6>1:
                    pct_lut_nums += (bit_pct//6*3+2)
                    pct_in_num = bit_pct//6+1
                elif bit_pct%6==1:
                    pct_lut_nums += (bit_pct//6*3+1)
                    pct_in_num = bit_pct//6+1
                else:
                    pct_lut_nums += (bit_pct//6*3)
                    pct_in_num = bit_pct//6
                pct_lut_num, _, _, _ = adder_tree(pct_in_num, freq=data['freq'])
                pct_lut_nums += pct_lut_num
                grp_sum = pct_lut_nums*10
                grp_sum += math.ceil(math.log2(10))
                layer['grp_sum_lut_num'] = grp_sum

    for layer in data['layers']:
        if layer['type'] in ['lut_conv', 'lut_res']:
            conv_num += 1
        elif layer['type'] == 'lut_fc':
            fc_num += 1
    data['conv_num'] = conv_num
    data['fc_num'] = fc_num

    return data


def json_parser(json_path, force_no_packing=False):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    layers = raw_data.get('layers', [])
    is_packing = any(
        isinstance(layer, dict) and isinstance(layer.get('packing_plan'), list) and len(layer.get('packing_plan')) > 0
        for layer in layers
    )

    if is_packing and not force_no_packing:
        data = pruned_json_parser(json_path)
    else:
        data = orignal_json_parser(json_path)

    data['packing'] = is_packing and not force_no_packing
    return data


if __name__ == '__main__':
    json_path = "Net1x6l_pruned_fpga_maping.json"
    data = json_parser(json_path)
    with open("Net1x6l_pruned_fpga_mapping_parsed.json", "w") as f:
        json.dump(data, f, indent=4)
