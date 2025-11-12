import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

original_data = np.random.normal(loc=0, scale=1, size=10000)

binarized_data = np.where(original_data >= 0, 1, 0)

random_shift = np.random.uniform(low=-2.0, high=2.0)
shifted_data = original_data + random_shift
shifted_binarized_data = np.where(shifted_data >= 0, 1, 0)

sns.set_style("whitegrid")
plt.figure(figsize=(14, 7))

blue_shade = '#1f78b4' 
green_shade = '#33a02c'
light_blue = '#a6cee3' 
light_green = '#b2df8a'

hist_original, bins_original, _ = plt.hist(original_data, bins=50, density=True, alpha=0.6, color=blue_shade, label='Original Distribution')

hist_shifted, bins_shifted, _ = plt.hist(shifted_data, bins=50, density=True, alpha=0.6, color=green_shade, label=f'Add Bias (Bias: {random_shift:.2f})')

binarized_counts = np.bincount(binarized_data, minlength=2)
binarized_freq = binarized_counts / len(binarized_data)
shifted_binarized_counts = np.bincount(shifted_binarized_data, minlength=2)
shifted_binarized_freq = shifted_binarized_counts / len(shifted_binarized_data)

plt.bar([0, 1], binarized_freq, width=0.2, color=light_blue, alpha=0.9, label='Binarized Data')

plt.bar([0.2, 1.2], shifted_binarized_freq, width=0.2, color=light_green, alpha=0.9, label='Shifted Binarized Data')

original_peak_x = bins_original[np.argmax(hist_original)]
original_peak_y = np.max(hist_original)

binarized_peak_x = 1
binarized_peak_y = np.max(binarized_freq)

plt.annotate('', 
             xy=(binarized_peak_x, binarized_peak_y), 
             xytext=(original_peak_x, original_peak_y),
             arrowprops=dict(facecolor='black', shrink=0.05, width=4, headwidth=8))

shifted_peak_x = bins_shifted[np.argmax(hist_shifted)]
shifted_peak_y = np.max(hist_shifted)

shifted_binarized_peak_x = 1.2
shifted_binarized_peak_y = np.min(shifted_binarized_freq)

plt.annotate('', 
             xy=(shifted_binarized_peak_x, shifted_binarized_peak_y), 
             xytext=(shifted_peak_x, shifted_peak_y),
             arrowprops=dict(facecolor='black', shrink=0.05, width=4, headwidth=8))

diff_value = binarized_freq[1] - shifted_binarized_freq[1]  
mid_x = (1 + 1.2) / 2  
max_y = max(binarized_freq[1], shifted_binarized_freq[1])
min_y = min(binarized_freq[1], shifted_binarized_freq[1])


plt.plot([mid_x, mid_x], [min_y, max_y], color='blue', linewidth=3, alpha=0.8)

plt.plot([mid_x-0.05, mid_x+0.05], [max_y, max_y], color='blue', linewidth=3, alpha=0.8)
plt.plot([mid_x-0.05, mid_x+0.05], [min_y, min_y], color='blue', linewidth=3, alpha=0.8)


plt.text(mid_x + 0.1, (max_y + min_y) / 2, f'Δ={diff_value:.3f}', 
         fontsize=24, color='blue', fontweight='bold', 
         verticalalignment='center', horizontalalignment='left')

plt.xlim(-4, 3.5)
plt.ylim(0.0, 1.0)
plt.title('Comparison of Distributions', fontsize=34, fontweight='bold')
plt.xlabel('Value', fontsize=30)
plt.ylabel('Density / Frequency', fontsize=30)
plt.legend(fontsize=30, loc='upper left')

plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

plt.tight_layout()
plt.savefig('bn_bias.png', dpi=300)
plt.close()
