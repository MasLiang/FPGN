import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Patch

networks_group1 = ['3-S', '3-M', '3-L', '3-G']
LUTs_group1 = [15, 44, 160, 623]
accuracy_group1 = [57.89, 64.35, 69.92, 73.75]

accuracy_group2 = [58.49, 64.19, 69.78, 73.15]

networks_group3 = ['6-S', '6-M', '6-L', '6-G']
LUTs_group3 = [32, 112, 434, 1717]
accuracy_group3 = [58.60, 67.32, 75.17, 77.87]

accuracy_group4 = [59.37, 67.80, 74.72, 78.31]

accuracy_group5 = [47.73, 57.35, 68.14, 73.08]

color_group1_bar = '#a6cee3'
color_group3_bar = '#b2df8a'
color_group1_line = '#1f78b4'
color_group3_line = '#33a02c'
color_group1_word= '#1f78b4' 
color_group3_word= '#33a02c'

color_group2_line = '#e31a1c'  # red
color_group4_line = '#6a3d9a'  # purple
color_group2_word = '#e31a1c'
color_group4_word = '#6a3d9a'

color_group5_line = '#ff7f00'  # orange
color_group5_word = '#ff7f00'

networks_for_bars = []
LUTs_for_bars = []
colors_for_bars = []
for i in range(len(networks_group1)):
    networks_for_bars.append(networks_group1[i])
    LUTs_for_bars.append(LUTs_group1[i])
    colors_for_bars.append(color_group1_bar)
    
    networks_for_bars.append(networks_group3[i])
    LUTs_for_bars.append(LUTs_group3[i])
    colors_for_bars.append(color_group3_bar)

x = np.arange(len(networks_for_bars))

fig, ax1 = plt.subplots(figsize=(9, 5))

bars = ax1.bar(x, LUTs_for_bars, color=colors_for_bars, width=0.4)

ax1.set_ylabel('LUTs (K)', fontsize=20)
ax1.set_xlabel('Neural Networks', fontsize=20)
ax1.set_yscale('log')
ax1.set_ylim(top=max(LUTs_for_bars) * 100)
ax1.tick_params(axis='y', labelsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(networks_for_bars, ha='center', fontsize=16)

for i, bar in enumerate(bars):
    yval = bar.get_height()
    color = color_group1_word if i % 2 == 0 else color_group3_word
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval}', va='bottom', ha='center', fontsize=14, color=color)

ax2 = ax1.twinx()

x_group1_lines = x[0::2] 
x_group3_lines = x[1::2] 

line1, = ax2.plot(x_group1_lines, accuracy_group1, label='FPGA-3', marker='o', color=color_group1_line, linestyle='--')
line2, = ax2.plot(x_group1_lines, accuracy_group2, label='FPGA-3 pruning', marker='x', color=color_group2_line, linestyle='--')
line3, = ax2.plot(x_group3_lines, accuracy_group3, label='FPGA-6', marker='s', color=color_group3_line, linestyle=':')
line4, = ax2.plot(x_group3_lines, accuracy_group4, label='FPGA-6 reposition', marker='^', color=color_group4_line, linestyle=':')
line5, = ax2.plot(x_group3_lines, accuracy_group5, label='FPGA-6 pruning', marker='p', color=color_group5_line, linestyle=':')

ax2.set_ylabel('Accuracy (%)', fontsize=20)
ax2.set_ylim(bottom=40, top=85)
ax2.grid(False)
ax2.tick_params(axis='y', labelsize=16)

offsets_g1 = [[-0.7,1,0.7,0.5],[0.7,-1.5,-1.5,-1.0]]
offsets_g3 = [[-1.0,-1.5,0.5,-1.5],[0.2,0.3,-1.5,0.5],[0,0,0,-1]]

groups_on_x1 = [accuracy_group1, accuracy_group2]
colors_on_x1 = [color_group1_word, color_group2_word]
for i, group in enumerate(groups_on_x1):
    for j, acc in enumerate(group):
        ax2.text(x_group1_lines[j], acc + offsets_g1[i][j], f'{acc:.2f}', va='center', ha='center', fontsize=12, color=colors_on_x1[i])

groups_on_x3 = [accuracy_group3, accuracy_group4, accuracy_group5]
colors_on_x3 = [color_group3_word, color_group4_word, color_group5_word]
for i, group in enumerate(groups_on_x3):
    for j, acc in enumerate(group):
        ax2.text(x_group3_lines[j], acc + offsets_g3[i][j], f'{acc:.2f}', va='center', ha='center', fontsize=12, color=colors_on_x3[i])

legend_elements = [line1, line2, line3, line5, line4,
                   Patch(facecolor=color_group1_bar, edgecolor='grey', label='FPGN-3 LUTs'),
                   Patch(facecolor=color_group3_bar, edgecolor='grey', label='FPGN-6 LUTs'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=12, ncol=2)

plt.tight_layout(rect=[0, 0.03, 1, 1.0])
plt.savefig('compare_networks.png', dpi=300, bbox_inches='tight')
plt.show()


