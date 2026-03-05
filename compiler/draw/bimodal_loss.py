import re
import matplotlib.pyplot as plt
import numpy as np

# 平滑函数：使用指数移动平均 (Exponential Moving Average)
# weight 越大，曲线越平滑 (取值范围 0 到 1)
def smooth_ema(scalars, weight=0.85):
    if len(scalars) == 0:
        return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# 假设这是你的日志文件名
log_filename = 'log_bimodal.txt'

def parse_and_plot(content=None, filename=None):
    # 1. 获取日志内容
    if filename:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"错误：找不到文件 {filename}，请检查路径。")
            return
    else:
        text = content

    # 2. 正则提取 Average loss
    pattern = re.compile(r"Average loss:\s*([\d\.]+)")
    matches = pattern.findall(text)
    
    losses = [float(x) for x in matches]
    total_count = len(losses)
    
    print(f"总共提取到 {total_count} 个数据点。")
    if total_count == 0:
        print("未找到任何符合格式的数据。")
        return

    # 3. 数据分组逻辑 (5大组)
    group_size = total_count // 5
    if total_count % 5 != 0:
        print(f"警告：数据总量 {total_count} 不能被5整除，末尾 {total_count % 5} 个数据将被忽略。")
        losses = losses[:group_size * 5]
        
    groups = np.array_split(losses, 5)
    
    # 4. 准备绘图 (使用面向对象的方式 fig, ax)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 颜色定义
    colors = ['#1f78b4', '#33a02c', '#a6cee3', '#b2df8a', '#c5b0d5','#e31a1c']
    
    # 定义阶段的分界线
    phase1_end = 40
    phase2_end = 140
    zoom_start_step = 140
    
    # 创建局部放大图 (画中画)
    # [x0, y0, width, height] 是相对于主图 ax 的归一化坐标 (0 到 1)
    axins = ax.inset_axes([0.35, 0.5, 0.45, 0.4])
    
    for i, group_data in enumerate(groups):
        even_lines_data = group_data[1::2] # 偶数行数据
        x_vals = np.arange(len(even_lines_data))
        
        # --- 主图：绘制原始数据 (线条加粗到 3.0) ---
        ax.plot(x_vals, even_lines_data, 
                 linestyle='-', 
                 color=colors[i], 
                 label=f'$\mu={list([0.5, 1, 2, 3, 4])[i]}$',
                 linewidth=3.0) 
        
        # --- 放大图：绘制平滑后的数据 (线条加粗到 4.0) ---
        smoothed_data = smooth_ema(even_lines_data, weight=0.9)
        axins.plot(x_vals, smoothed_data, 
                   linestyle='-', 
                   color=colors[i], 
                   linewidth=3.0)

    # ================= 新增：绘制阶段分割线和文字 =================
    # 添加竖直虚线
    ax.axvline(x=phase1_end, color='gray', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(x=phase2_end, color='gray', linestyle='--', linewidth=2.5, alpha=0.8)
    
    # 添加阶段文字标签 (使用 transform=ax.get_xaxis_transform() 使 y 坐标为 0-1 的相对比例，保证文字始终在顶部)
    text_kwargs = dict(transform=ax.get_xaxis_transform(), fontsize=16, fontweight='bold', 
                       ha='center', va='top', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.text(phase1_end / 3, 0.05, 'FP Training', **text_kwargs)
    ax.text((phase1_end + phase2_end) / 2, 0.05, 'Gradually Quant', **text_kwargs)
    
    # 假设最大 step 在 200 左右，将 Phase 3 的文字放在 140 到末尾的中间
    max_step_for_text = len(groups[0][1::2])
    ax.text((phase2_end + max_step_for_text) / 2, 0.05, 'Binary Training', **text_kwargs)
    # ==============================================================

    # 5. 设置主图属性 (文字放大)
    ax.set_title('Average Loss Extraction', fontsize=20, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Loss', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18) # 放大主图刻度数字
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=14) # 放大图例文字
    
    # 6. 设置放大图 (Inset) 属性 (文字放大)
    max_step = len(groups[0][1::2])
    axins.set_xlim(zoom_start_step, max_step)
    axins.set_ylim(0.17, 0.21) 
    
    axins.set_title('Exponential Moving Average', fontsize=14, fontweight='bold')
    axins.grid(True, linestyle=':', alpha=0.6)
    axins.tick_params(axis='both', which='major', labelsize=12) # 放大画中画刻度数字
    
    # 绘制连接主图和放大图的框线 (稍微加粗一点框线)
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8, linewidth=2.0)
    
    # 7. 保存并显示
    save_path = 'bimodal_loss.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存为: {save_path}")
    plt.show()

# 运行函数
parse_and_plot(filename=log_filename)