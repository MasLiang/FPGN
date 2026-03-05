import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ==========================================
# 1. 数据准备 (Data Preparation)
# ==========================================

# --- A. LUT 数据 (你提供的原始数据) ---
lut_compiler = np.array([
    22408, 22435, 22489, 23110, 23218, 24460, 24676, 26468, 28952, 29384,
    32968, 37936, 44848, 49968, 65009, 76785, 116044, 198163, 372138
])

lut_synth = np.array([
    23286, 23216, 23273, 23718, 23800, 24873, 25067, 26871, 28669, 29104, 
    32336, 39066, 45845, 50224, 66814, 79067, 110432, 192209, 357744
])

lut_impl = np.array([
    20723, 20663, 20690, 21190, 21209, 21991, 22227, 23791, 25409, 25708,
    28627, 33336, 39599, 43539, 58950, 69879, 84592, 147632, 291957
])

# --- B. Register (FF) 数据 (!!! 请在此处填入你的真实数据 !!!) ---
# 这里我暂时用模拟数据代替，为了让代码能跑通
# 假设 Register 数量大约是 LUT 的 1.5 倍左右 (仅做演示)
reg_compiler = np.array([ 
    8938, 8938, 8938, 9066, 9066, 9322, 9322, 9706, 10218, 10218, 10986, 
    12316, 13852, 15388, 18948, 22020, 33108, 61086, 102046])

reg_synth    = np.array([ 
    8940, 8940, 8940, 9068, 9068, 9324, 9324, 9708, 10220, 10220, 10988,
    12319, 13855, 15391, 18953, 22025, 33116, 61162, 102122])

reg_impl     = np.array([ 
    8217, 8217, 8217, 8217, 8300, 8461, 8457, 8844, 9358, 9365, 10143,
    13018, 14563, 16095, 18162, 21220, 32277, 60383, 104029])


# ==========================================
# 2. 绘图函数封装 (Plotting Function)
# ==========================================
def plot_metric(ax, x_data, y_synth, y_impl, resource_name):
    """
    在指定的 ax 上绘制 Compiler vs Ground Truth 的对比图
    """
    # 2.1 计算相关系数
    r_synth, _ = stats.pearsonr(x_data, y_synth)
    r_impl, _ = stats.pearsonr(x_data, y_impl)

    # 2.2 计算拟合线 (针对 Impl 数据)
    slope, intercept, _, _, _ = stats.linregress(x_data, y_impl)
    line_fit = slope * x_data + intercept

    # 2.3 绘制散点图
    # Post-Synthesis (绿色三角形)
    ax.scatter(x_data, y_synth, 
               color='#2ca02c', marker='^', alpha=0.7, s=50, 
               label='vs. Post-Synthesis')
    
    # Post-Implementation (红色圆形)
    ax.scatter(x_data, y_impl, 
               color='#d62728', marker='o', alpha=0.7, s=50, 
               label='vs. Post-Implementation')

    # 2.4 辅助线
    # y=x 理想线 (灰色虚线)
    max_val = max(x_data.max(), y_synth.max(), y_impl.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle=':', alpha=0.6, label='Ideal (y=x)')
    
    # 拟合线 (红色虚线)
    ax.plot(x_data, line_fit, color='#d62728', linestyle='--', alpha=0.5)

    # 2.5 装饰与标签
    ax.set_xlabel(f'Compiler Estimated {resource_name}', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'Vivado Reported {resource_name}', fontsize=16, fontweight='bold')
    ax.set_title(f'{resource_name} Estimation Accuracy', fontsize=18, fontweight='bold')
    
    # 2.6 统计数据文本框
    textstr = '\n'.join((
        r'$\bf{Validation\ Metrics:}$',
        r'vs. Synth: $r = %.4f$' % (r_synth, ),
        r'vs. Impl:  $r = %.4f$' % (r_impl, )
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # 设置图例
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=13)
    
    # 如果数据跨度非常大，建议开启对数坐标 (可选)
    # ax.set_xscale('log')
    # ax.set_yscale('log')


# ==========================================
# 3. 主程序 (Main Execution)
# ==========================================

# 设置风格
try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.style.use('ggplot')

# 创建画布：1行2列
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# 绘制左图 (LUT)
plot_metric(axes[0], lut_compiler, lut_synth, lut_impl, "LUTs")

# 绘制右图 (Registers)
plot_metric(axes[1], reg_compiler, reg_synth, reg_impl, "Registers")

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('resource_estimation_comparison.png', bbox_inches='tight')
plt.show()