import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

sns.set(style="whitegrid")
title_size = 22
other_size = 20
legend_size = 16
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': other_size,
    'axes.titlesize': title_size,
    'axes.labelsize': other_size,
    'xtick.labelsize': other_size,
    'ytick.labelsize': other_size
})

def sigmoid(x, tau):
    return 1 / (1 + np.exp(-x / tau))

def sign_func(x):
    return np.where(x >= 0, 1, 0)

def sigmoid_deriv(x, tau):
    s = sigmoid(x, tau)
    return (1/tau) * s * (1 - s)

x = np.linspace(-10, 10, 1000)
taus = [1, 0.8, 0.5, 0.2, 0.1]
colors = sns.color_palette("viridis", len(taus))

plt.figure(figsize=(10, 7), dpi=300)

main_ax = plt.gca()

for i, (tau, color) in enumerate(zip(taus, colors)):
    y = sigmoid(x, tau)
    y_deriv = sigmoid_deriv(x, tau)
    main_ax.plot(x, y, label=f'Sigmoid τ={tau}', color=color, linewidth=2.5)
    main_ax.plot(x, y_deriv, label=f'Deriv τ={tau}', color=color, linestyle=':', linewidth=2)
    main_ax.scatter([0], [0.5], color=color, s=80, zorder=5, edgecolor='white')

main_ax.plot(x, sign_func(x), label='sign(x)', color='black', linestyle='--', linewidth=2)

rect_x1 = 1.7
rect_x2 = 2.3
rect_y1 = -0.02
rect_y2 = 0.12
rect = Rectangle((rect_x1, rect_y1), rect_x2-rect_x1, rect_y2-rect_y1,
                linewidth=1, edgecolor='gray', facecolor='none', alpha=0.5)
main_ax.add_patch(rect)

main_ax.grid(True, linestyle='--', alpha=0.7)
main_ax.minorticks_on()
main_ax.grid(which='minor', linestyle=':', alpha=0.4)
main_ax.set_xlim(-6, 6)
main_ax.set_ylim(-0.05, 1.05)

axins = inset_axes(main_ax, width="30%", height="50%", 
                   bbox_to_anchor=(0.3, 0, 1, 1),  
                   bbox_transform=main_ax.transAxes,  
                   loc='center',  
                   borderpad=0)

axins.set_title('Zoomed area', fontsize=14, pad=5)

mask = (x >= rect_x1) & (x <= rect_x2)
for i, (tau, color) in enumerate(zip(taus, colors)):
    y = sigmoid(x, tau)
    y_deriv = sigmoid_deriv(x, tau)
    axins.plot(x[mask], y[mask], color=color, linewidth=2.5)
    axins.plot(x[mask], y_deriv[mask], color=color, linestyle=':', linewidth=2)

axins.plot(x[mask], sign_func(x[mask]), color='black', linestyle='--', linewidth=2)

axins.set_xlim(rect_x1, rect_x2)
axins.set_ylim(rect_y1, rect_y2)
axins.grid(True, linestyle='--', alpha=0.7)

axins.tick_params(axis='both', which='major', labelsize=12)  
axins.xaxis.set_major_locator(plt.MaxNLocator(3))  
axins.yaxis.set_major_locator(plt.MaxNLocator(3))

con1 = ConnectionPatch(xyA=(rect_x1, rect_y2), coordsA=main_ax.transData,
                      xyB=(rect_x1, rect_y1), coordsB=axins.transData,
                      color="gray", alpha=0.5, linewidth=1)
con2 = ConnectionPatch(xyA=(rect_x2, rect_y2), coordsA=main_ax.transData,
                      xyB=(rect_x2, rect_y1), coordsB=axins.transData,
                      color="gray", alpha=0.5, linewidth=1)
main_ax.add_artist(con1)
main_ax.add_artist(con2)

main_ax.set_title(r'Sigmoid & Derivative: $y = \frac{1}{1 + e^{-x/\tau}}$', fontweight='bold', pad=20)
main_ax.set_xlabel('Input (x)', labelpad=10, fontsize=title_size)
main_ax.set_ylabel('Value', labelpad=10, fontsize=title_size)
main_ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True, fontsize=legend_size)

plt.tight_layout()
plt.savefig("sigmoid_and_deriv.png", dpi=300, bbox_inches='tight')
plt.savefig("sigmoid_and_deriv.svg", format='svg', bbox_inches='tight')
plt.show()
