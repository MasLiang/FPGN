import json
import math


def adder_tree(in_num, freq=200):
    period = 1000 / freq
    period_safe = period / 2
    period_lut6 = 0.053

    def adder_latency(bit_width):
        period_in = 0.269
        period_lut6 = 0.053
        period_s0co3 = 0.313
        period_cico = [0.179, 0.132, 0.094, 0.058]
        period_cio = [0.139, 0.213, 0.136, 0.179]

        if bit_width == 3 or bit_width == 4:
            return period_in + period_lut6
        elif bit_width == 5:
            return period_in + period_lut6 * 2
        else:
            carry_chain_num = math.floor((bit_width - 4) / 4)
            out_idx = (bit_width - 4) % 4
            c_delay = period_in + period_lut6 + period_s0co3 + carry_chain_num * period_cico[3] + period_cico[out_idx]
            s_delay = period_in + period_lut6 + period_s0co3 + carry_chain_num * period_cico[3] + period_cio[out_idx]
            return max(c_delay, s_delay)

    data_num = in_num
    bit_width = 3
    period_left = period_safe
    reg_num = 0
    lut_num = 0
    reg_idx = []
    add_idx = 0
    while data_num > 1:
        adder_num = math.floor(data_num / 2)
        lut_num_per_bit = bit_width
        lut_num += adder_num * lut_num_per_bit
        data_num = adder_num + data_num % 2
        bit_width += 1
        add_idx += 1
    return lut_num, reg_num, reg_idx, bit_width


def orignal_json_parser(json_path):
    """Parse a standard model description JSON file."""
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
    res_bit_width = 0
    for layer in data['layers']:
        layer_idx = data['layers'].index(layer)
        if layer_idx == len(data['layers']) - 1:
            next_layer = {'type': 'end'}
        else:
            next_layer = data['layers'][layer_idx + 1]

        if layer['type'] == 'lut_quant':
            layer['kernel_size'] = 1
            layer['stride'] = 1
            layer['padding'] = [0, 0]
            layer['row'] = data['row']
            layer['col'] = data['col']
            res_bit_width = 0
            lut_num_tmp = data['bit_width']
            lut_num = 0
            while lut_num_tmp > 1:
                lut_num_tmp = math.ceil(lut_num_tmp / data['lut_size'])
                lut_num += lut_num_tmp
            lut_num = lut_num * layer['quant_channels']
            layer['lut_num'] = lut_num * layer['out_channel']
            if next_layer['type'] in ['lut_conv', 'lut_res']:
                mem_row = next_layer['kernel_size'] - next_layer['stride']
                reg_num = mem_row * next_layer['col'] * layer['out_channel'] * layer['quant_channels']
                layer['has_variable_reg'] = True
                layer['reg_bias'] = next_layer['stride'] * next_layer['col'] * layer['out_channel'] * layer['quant_channels']
            elif next_layer['type'] == 'lut_fc':
                reg_num = layer['row'] * layer['col'] * layer['out_channel'] * layer['quant_channels']
                layer['has_variable_reg'] = False
            layer['reg_num'] = reg_num
            in_channel = layer['out_channel'] * layer['quant_channels']

        elif layer['type'] in ['lut_conv', 'lut_res']:
            layer['in_channel'] = in_channel
            lut_num = math.ceil(layer['kernel_size'] * layer['kernel_size'] * in_channel / data['lut_size'])
            pct_lut_nums = 0
            pct_reg_nums = 0
            pct_reg_idxs = []
            previous_res_bit_width = res_bit_width

            if lut_num % 6 > 1:
                pct_in_num = lut_num // 6 + 1
                pct_reg_nums += (lut_num // 6 * 3 + 2) * layer['out_channel']
                pct_lut_nums += (lut_num // 6 * 3 + 2) * layer['out_channel']
            elif lut_num % 6 == 1:
                pct_in_num = lut_num // 6 + 1
                pct_reg_nums += (lut_num // 6 * 3 + 1) * layer['out_channel']
                pct_lut_nums += 0
            else:
                pct_in_num = lut_num // 6
                pct_reg_nums += (lut_num // 6 * 3) * layer['out_channel']
                pct_lut_nums += (lut_num // 6 * 3) * layer['out_channel']

            for out_channel_idx in range(layer['out_channel']):
                pct_lut_num, pct_reg_num, pct_reg_idx, res_bit_width = adder_tree(pct_in_num, freq=data['freq'])
                pct_lut_num += math.ceil(math.log2(layer['threshold'][out_channel_idx]))
                pct_lut_nums += pct_lut_num
                pct_reg_nums += pct_reg_num
                pct_reg_idxs.append([0] + pct_reg_idx)

            layer['lut_num'] = math.ceil(lut_num * layer['out_channel']) + pct_lut_nums
            if layer['type'] == 'lut_res':
                layer['lut_num'] += max(previous_res_bit_width, res_bit_width) * layer['out_channel']
                res_bit_width = max(previous_res_bit_width, res_bit_width) + 1
                layer['res_bit_width'] = res_bit_width
            else:
                layer['res_bit_width'] = 0
            in_channel = layer['out_channel']

            if next_layer['type'] in ['lut_conv', 'lut_res']:
                mem_row = next_layer['kernel_size'] - next_layer['stride']
                reg_num = mem_row * next_layer['col'] * layer['out_channel']
                layer['has_variable_reg'] = True
                layer['reg_bias'] = next_layer['stride'] * next_layer['col'] * layer['out_channel']
            elif next_layer['type'] == 'lut_fc':
                reg_num = (layer['col'] // layer['stride']) * (layer['row'] // layer['stride']) * layer['out_channel']
                layer['has_variable_reg'] = False
            layer['reg_num'] = reg_num
            layer['pct_reg_num'] = pct_reg_nums
            layer['pct_reg_idxs'] = pct_reg_idxs
            layer['additional_latency'] = max([len(_) for _ in pct_reg_idxs]) if pct_reg_idxs else 0

        elif layer['type'] == 'lut_fc':
            if fc_idx % 4 == 0 and fc_idx > 0:
                layer['reg_num'] = layer['lut_num']
            else:
                layer['reg_num'] = 0
            if next_layer['type'] != 'lut_fc':
                layer['reg_num'] = layer['lut_num']
                bit_pct = math.ceil(layer['lut_num'] // 10)
                if bit_pct % 6 > 1:
                    pct_lut_nums = bit_pct // 6 * 3 + 2
                    pct_in_num = bit_pct // 6 + 1
                elif bit_pct % 6 == 1:
                    pct_lut_nums = bit_pct // 6 * 3 + 1
                    pct_in_num = bit_pct // 6 + 1
                else:
                    pct_lut_nums = bit_pct // 6 * 3
                    pct_in_num = bit_pct // 6
                pct_lut_num, _, _, _ = adder_tree(pct_in_num, freq=data['freq'])
                pct_lut_nums += pct_lut_num
                layer['grp_sum_lut_num'] = pct_lut_nums * 10 + math.ceil(math.log2(10))
    for layer in data['layers']:
        if layer['type'] in ['lut_conv', 'lut_res']:
            conv_num += 1
        elif layer['type'] == 'lut_fc':
            fc_num += 1
    data['conv_num'] = conv_num
    data['fc_num'] = fc_num
    return data


def json_parser(json_path, force_no_packing=False):
    """Parse a model JSON without packing transformations.

    ``force_no_packing`` is retained for compatibility with existing solver
    callers. The returned configuration always reports ``packing=False``.
    """
    data = orignal_json_parser(json_path)
    data['packing'] = False
    return data


if __name__ == '__main__':
    json_path = "model_execution_info.json"
    data = json_parser(json_path)
    with open("model_execution_info_parsed.json", "w") as f:
        json.dump(data, f, indent=4)
