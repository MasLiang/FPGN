import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset

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
    'ytick.labelsize': other_size,
    'legend.fontsize': legend_size,
})

def bimodal_pdf(x, mu, sigma):
    return 0.5 * norm.pdf(x, loc=-mu, scale=sigma) + 0.5 * norm.pdf(x, loc=mu, scale=sigma)

def sigmoid(x, tau=1):
    return 1 / (1 + np.exp(-x / tau))

def sigmoid_derivative(x, tau=1):
    s = sigmoid(x, tau)
    return s * (1 - s) / tau

def sign_func(x):
    return np.where(x >= 0, 1, 0)

fig = plt.figure(figsize=(10, 10), dpi=300)  

x = np.linspace(-6, 6, 1000)

ax1 = fig.add_subplot(2, 1, 1)  

taus = [1, 0.8, 0.5, 0.2, 0.1]
colors_tau = sns.color_palette("viridis", len(taus))

sigmoid_lines = []  
for i, (tau, color) in enumerate(zip(taus, colors_tau)):
    y = sigmoid(x, tau)
    y_deriv = sigmoid_derivative(x, tau)
    line1, = ax1.plot(x, y, color=color, linewidth=2.5)
    line2, = ax1.plot(x, y_deriv, color=color, linestyle=':', linewidth=2)
    ax1.scatter([0], [0.5], color=color, s=80, zorder=5, edgecolor='white')
    sigmoid_lines.append((line1, line2))

sign_line, = ax1.plot(x, sign_func(x), color='black', linestyle='--', linewidth=2)

rect_x1 = 1.7
rect_x2 = 2.3
rect_y1 = -0.02
rect_y2 = 0.12
rect = Rectangle((rect_x1, rect_y1), rect_x2-rect_x1, rect_y2-rect_y1,
                linewidth=1, edgecolor='gray', facecolor='none', alpha=0.5)
ax1.add_patch(rect)

ax1.grid(True, linestyle='--', alpha=0.7)
ax1.minorticks_on()
ax1.grid(which='minor', linestyle=':', alpha=0.4)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-0.05, 1.05)
ax1.tick_params(axis='y', which='both', length=0)

axins = inset_axes(ax1, width="30%", height="50%", 
                  bbox_to_anchor=(0.3, 0, 1, 1),
                  bbox_transform=ax1.transAxes,
                  loc='center',
                  borderpad=0)

axins.set_title('Zoomed area', fontsize=14, pad=5)

mask = (x >= rect_x1) & (x <= rect_x2)
for i, (tau, color) in enumerate(zip(taus, colors_tau)):
    y = sigmoid(x, tau)
    y_deriv = sigmoid_derivative(x, tau)
    axins.plot(x[mask], y[mask], color=color, linewidth=2.5)
    axins.plot(x[mask], y_deriv[mask], color=color, linestyle=':', linewidth=2)

axins.plot(x[mask], sign_func(x[mask]), color='black', linestyle='--', linewidth=2)

axins.set_xlim(rect_x1, rect_x2)
axins.set_ylim(rect_y1, rect_y2)
axins.grid(True, linestyle='--', alpha=0.7)

axins.tick_params(axis='both', which='major', labelsize=12)
axins.tick_params(axis='y', which='both', length=0)
axins.xaxis.set_major_locator(plt.MaxNLocator(3))
axins.yaxis.set_major_locator(plt.MaxNLocator(3))

con1 = ConnectionPatch(xyA=(rect_x1, rect_y2), coordsA=ax1.transData,
                      xyB=(rect_x1, rect_y1), coordsB=axins.transData,
                      color="gray", alpha=0.5, linewidth=1)
con2 = ConnectionPatch(xyA=(rect_x2, rect_y2), coordsA=ax1.transData,
                      xyB=(rect_x2, rect_y1), coordsB=axins.transData,
                      color="gray", alpha=0.5, linewidth=1)
ax1.add_artist(con1)
ax1.add_artist(con2)

ax1.set_title(r'Sigmoid & Derivative: $\sigma(x) = \frac{1}{1 + e^{-x/\tau}}$', fontweight='bold', pad=10)
ax1.set_xlabel('Input (x)', labelpad=10, fontsize=title_size)
ax1.set_ylabel('Value', labelpad=10, fontsize=title_size)

legend_labels = [r'$\sigma(x)\ &\ d\sigma(x)\ \tau={0}$'.format(tau) for tau in taus] + ['sign(x)']
legend_handles = sigmoid_lines + [(sign_line,)]
ax1.legend(legend_handles, legend_labels, handler_map={tuple: plt.matplotlib.legend_handler.HandlerTuple(None)},
          loc='upper left', frameon=True, framealpha=0.95, edgecolor='gray', 
          fancybox=True, shadow=True, fontsize=legend_size)

ax2 = fig.add_subplot(2, 1, 2)  

means = [4, 2, 1, 0.5]
variance = 0.1
sigma = np.sqrt(variance)
colors_bimodal = sns.color_palette("viridis", len(means))

ax2.grid(True, which='major', linestyle='--', alpha=0.7)
ax2.minorticks_on()
ax2.set_yticks(np.arange(0, 1.1, 0.04), minor=True)
ax2.set_xticks(np.arange(-6, 6.1, 0.4), minor=True)
ax2.grid(which='minor', linestyle=':', alpha=0.4)

ax2_2 = ax2.twinx()

lines_main = []
for i, mu in enumerate(means):
    y_bimodal = bimodal_pdf(x, mu, sigma)
    line, = ax2.plot(x, y_bimodal, color=colors_bimodal[i], linewidth=2.5, 
                   label=f'μ=±{mu}')
    lines_main.append(line)

y_sigmoid = sigmoid(x)
y_sigmoid_deriv = sigmoid_derivative(x)

line_sig, = ax2_2.plot(x, y_sigmoid, color='red', linestyle='--', linewidth=2.5, 
                     label='Sigmoid')
line_deriv, = ax2_2.plot(x, y_sigmoid_deriv, color='red', linestyle=':', linewidth=2.5, 
                       label='Sigmoid Derivative')

ax2.set_xlabel('Value (x)', labelpad=15, fontsize=title_size)
ax2.set_ylabel('Probability Density', labelpad=15, fontsize=title_size)
ax2.set_xlim(-6, 6)
ax2.set_ylim(0, 1.05)
ax2.tick_params(axis='y', which='both', length=0)

ax2_2.set_ylabel('Sigmoid Value', color='red', labelpad=15, fontsize=title_size)
ax2_2.tick_params(axis='y', labelcolor='red', length=0)
ax2_2.set_ylim(0, 1.05)

first_legend_lines = lines_main + [line_sig, line_deriv]
first_legend_labels = [l.get_label() for l in lines_main] + [r'$\sigma(x)$', r'$d\sigma(x)$']

leg = ax2_2.legend(first_legend_lines, first_legend_labels, 
          frameon=True, framealpha=1.0, edgecolor='black', 
          fancybox=True, shadow=True, ncol=2,  
          bbox_to_anchor=(0.01, 0.99), loc='upper left')
leg.set_zorder(9999)

ax2.set_title('Bimodal Distributions', 
            fontsize=title_size, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  
plt.savefig("sigmoid_bimodal.png", dpi=300, bbox_inches='tight')
print("Combined plot saved as sigmoid_bimodal.png")

plt.show()
