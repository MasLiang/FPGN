import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'sans-serif']

blue_shade = '#1f78b4' 
green_shade = '#33a02c'
light_blue = '#a6cee3' 
light_green = '#b2df8a'

triangle_color = blue_shade  
series_color = green_shade   

colors = [blue_shade, green_shade, light_blue, light_green]

group1_data = [540.64, 399.84, 416.58, 416.56, 431.97, 418.76]  
group2_data = [549.45, 373.13, 378.50, 416.67, 387.60, 387.15]  
group3_data = [526.32, 298.51, 304.41, 285.71, 294.12, 299.58]    
group4_data = [510.20, 217.49, 238.10, 228.41, 224.72, 208.33]  

all_data = [group1_data, group2_data, group3_data, group4_data]
group_names = ['500', '1000', '2000', '3000']

fig, ax = plt.subplots(figsize=(5, 4))

x_positions = np.arange(len(group_names))

x_bias = [[0.15,0,0,-0.15],[0.4,0,0,-0.4]]
y_bias = [[-55,-55,-55,-55],[45,45,40,50]]
for i, (data, name) in enumerate(zip(all_data, group_names)):
    first_value = data[0]
    ax.scatter(i, first_value, marker='^', s=100, color=triangle_color, 
              label=f'{name} - Single' if i == 0 else "", edgecolors='black', linewidth=1)
    
    series_data = data[1:]
    mean_value = np.mean(series_data)
    std_value = np.std(series_data)
    
    ax.scatter(i, mean_value, marker='o', s=20, color=series_color, 
              label=f'{name} - Series Mean' if i == 0 else "", alpha=0.7)
    
    ax.errorbar(i, mean_value, yerr=std_value, color=series_color, 
               capsize=5, capthick=2, linewidth=2, alpha=0.7)
    
    ax.text(i+x_bias[0][i], first_value+y_bias[0][i], f'{first_value:.1f}', 
           ha='center', va='bottom', fontsize=14, color=triangle_color, fontweight='bold')
    ax.text(i+x_bias[1][i], mean_value+y_bias[1][i], f'{mean_value:.1f}±{std_value:.1f}', 
           ha='center', va='top', fontsize=14, color=series_color, fontweight='bold')

ax.set_xlabel('LUT Numbers', fontsize=16)
ax.set_ylabel('Fmax(MHz)', fontsize=16)

ax.set_xticks(x_positions)
ax.set_xticklabels(group_names, fontsize=14)

ax.tick_params(axis='y', labelsize=14)
ax.grid(True, alpha=0.3, linestyle='--')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='^', color=triangle_color, linestyle='None', 
           markersize=10, label='in-order', markeredgecolor='black'),
    Line2D([0], [0], marker='o', color=series_color, linestyle='None', 
           markersize=6, label='out-of-order', alpha=0.7)
]

ax.legend(handles=legend_elements, loc='lower right', fontsize=16, 
         bbox_to_anchor=(0.6, 0.0))

plt.tight_layout()

plt.savefig('random_routing.png', 
           dpi=300, bbox_inches='tight')
plt.show()
