import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gaussian_kde

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
    """
    Calculates the probability density function of a bimodal distribution
    which is a mixture of two normal distributions with means -mu and +mu.
    """
    return 0.5 * norm.pdf(x, loc=-mu, scale=sigma) + 0.5 * norm.pdf(x, loc=mu, scale=sigma)

def sigmoid(x, tau=1):
    """Calculates the sigmoid function."""
    return 1 / (1 + np.exp(-x / tau))

def sigmoid_derivative(x, tau=1):
    """Calculates the derivative of the sigmoid function."""
    s = sigmoid(x, tau)
    return s * (1 - s) / tau

def main():
    """
    Generates and plots bimodal distributions and the distribution of sigmoid(x)
    where x follows each bimodal distribution as two separate figures.
    """
    means = [4, 2, 1, 0.5]
    
    variance = 0.1
    sigma = np.sqrt(variance)
    
    x = np.linspace(-6, 6, 1000)
    
    colors = sns.color_palette("viridis", len(means))
    
    fig1 = plt.figure(figsize=(10, 7), dpi=300)
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    
    ax1.grid(True, which='major', linestyle='--', alpha=0.7)
    ax1.minorticks_on()
    
    ax1.set_yticks(np.arange(0, 1.1, 0.04), minor=True)
    ax1.set_xticks(np.arange(-6, 6.1, 0.4), minor=True)
    ax1.grid(which='minor', linestyle=':', alpha=0.4)
    
    ax1_2 = ax1.twinx()
    
    lines_main = []
    for i, mu in enumerate(means):
        y_bimodal = bimodal_pdf(x, mu, sigma)
        line, = ax1.plot(x, y_bimodal, color=colors[i], linewidth=2.5, 
                       label=f'Bimodal (μ=±{mu}, σ²={variance})')
        lines_main.append(line)
    
    y_sigmoid = sigmoid(x)
    y_sigmoid_deriv = sigmoid_derivative(x)
    
    line_sig, = ax1_2.plot(x, y_sigmoid, color='red', linestyle='--', linewidth=2.5, 
                         label='Sigmoid')
    line_deriv, = ax1_2.plot(x, y_sigmoid_deriv, color='red', linestyle=':', linewidth=2.5, 
                           label='Sigmoid Derivative')
    
    ax1.set_xlabel('Value (x)', labelpad=15, fontsize=title_size)
    ax1.set_ylabel('Probability Density', labelpad=15, fontsize=title_size)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(0, 1.05)
    
    ax1.tick_params(axis='y', which='both', length=0)
    
    ax1_2.set_ylabel('Sigmoid Value', color='red', labelpad=15, fontsize=title_size)
    ax1_2.tick_params(axis='y', labelcolor='red', length=0)
    ax1_2.set_ylim(0, 1.05)
    
    fig1.canvas.draw()
    
    first_legend_lines = lines_main + [line_sig, line_deriv]
    first_legend_labels = [l.get_label() for l in lines_main] + ['Sigmoid', 'Sigmoid Derivative']
    
    leg = ax1_2.legend(first_legend_lines, first_legend_labels, 
              frameon=True, framealpha=1.0, edgecolor='black', 
              fancybox=True, shadow=True, 
              bbox_to_anchor=(0.01, 0.99), loc='upper left')
    
    leg.set_zorder(9999)
    
    fig1.suptitle('Original Bimodal Distributions', 
                fontsize=title_size, fontweight='bold', y=0.97)
    
    fig1.savefig("bimodal_distributions.png", dpi=300, bbox_inches='tight')
    fig1.savefig("bimodal_distributions.svg", format='svg', bbox_inches='tight')
    print("First plot saved as bimodal_distributions.png/svg")
    
    fig2 = plt.figure(figsize=(10, 7), dpi=300)
    ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
    
    lines_sec = []
    
    x_sigmoid = np.linspace(0, 1, 1000)
    
    for i, mu in enumerate(means):
        np.random.seed(42+i)  
        num_samples = 1000000
        samples = []
        for _ in range(num_samples):
            if np.random.random() < 0.5:
                samples.append(np.random.normal(-mu, sigma))
            else:
                samples.append(np.random.normal(mu, sigma))
        
        sigmoid_samples = sigmoid(np.array(samples))
        
        hist, bin_edges = np.histogram(sigmoid_samples, bins=100, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        line, = ax2.plot(bin_centers, hist/1000000, color=colors[i], linestyle='-', linewidth=2.5,
                       label=f'Sigmoid(x) for x ~ Bimodal(±{mu})')
        lines_sec.append(line)
    
    ax2.set_xlabel('Sigmoid(x) Value [0-1 range]', labelpad=15, fontsize=title_size)
    ax2.set_ylabel('Probability Density', labelpad=15, fontsize=title_size)
    
    ax2.grid(True, which='major', linestyle='--', alpha=0.7)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    ax2.set_xlim(0, 1)  
    ax2.set_ylim(bottom=0)  
    
    y_max = ax2.get_ylim()[1]
    rect_width = 0.08  
    rect_center = 0.8  
    
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    
    axins = zoomed_inset_axes(ax2, zoom=4, 
                              bbox_to_anchor=(0.82, 0.70),
                              bbox_transform=ax2.transAxes,
                              loc='upper right',
                              borderpad=1)
    
    for i, mu in enumerate(means):
        np.random.seed(42+i)  
        num_samples = 1000000
        samples = []
        for _ in range(num_samples):
            if np.random.random() < 0.5:
                samples.append(np.random.normal(-mu, sigma))
            else:
                samples.append(np.random.normal(mu, sigma))
        
        sigmoid_samples = sigmoid(np.array(samples))
        
        hist, bin_edges = np.histogram(sigmoid_samples, bins=100, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        axins.plot(bin_centers, hist/1000000, color=colors[i], linestyle='-', linewidth=2.5)
    
    x1, x2 = rect_center - rect_width/2, rect_center + rect_width/2
    y1, y2 = 0, y_max*0.1
    
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    
    axins.tick_params(labelleft=False, labelbottom=False)
    
    axins.grid(True, which='major', linestyle='--', alpha=0.7)
    
    mark_inset(ax2, axins, loc1=3, loc2=4, fc="none", ec="0.5")

    fig2.canvas.draw()
    
    leg2 = ax2.legend(frameon=True, framealpha=1.0, edgecolor='black', 
                      fancybox=True, shadow=True, loc='upper center')
    leg2.set_zorder(9999)
    
    fig2.suptitle('Sigmoid(x) Distributions', 
                fontsize=title_size+4, fontweight='bold', y=1.02)
    
    fig2.savefig("sigmoid_distributions.png", dpi=300, bbox_inches='tight')
    fig2.savefig("sigmoid_distributions.svg", format='svg', bbox_inches='tight')
    print("Second plot saved as sigmoid_distributions.png/svg")
    
    plt.show()

if __name__ == '__main__':
    main()
