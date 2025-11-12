import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 18})

def plot_grouped_barchart(ax, data, yname):
    n_groups, n_bars_per_group = data.shape
    index = np.arange(n_groups)
    bar_width = 0.4 / n_bars_per_group
    opacity = 0.8

    colors = plt.cm.viridis(np.linspace(0.8, 0.4, n_bars_per_group))

    for i in range(n_bars_per_group):
        bar_position = index - 0.4 + i * bar_width + bar_width / 2
        ax.bar(bar_position, data[:, i], bar_width,
               alpha=opacity,
               color=colors[i],
               label=f'LUT{6-2*i}')

        for j, value in enumerate(data[:, i]):
            ax.text(bar_position[j], value + 1, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel(yname)
    ax.set_xticks(index)
    ax.set_xticklabels(['FPGN-3-S', 'FPGN-3-M', 'FPGN-3-L', 'FPGN-3-G'])

    ax.tick_params(axis='x', labelrotation=20)


def main():
    data1 = np.array([[57.89, 56.27, 54.00],
                      [64.35, 62.03, 60.00],
                      [69.92, 67.52, 60.00],
                      [73.75, 72.65, 70.00]])
    data2 = np.array([[15, 17, 20],
                      [44, 52, 60],
                      [160, 192, 200],
                      [623, 747, 800]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    plot_grouped_barchart(ax1, data1, "Accuracy (%)")

    plot_grouped_barchart(ax2, data2, "LUTs (K)")

    ax1_title_y = -0.5
    ax2_title_y = -0.5

    fig.set_size_inches(12, 3)

    handles, labels = ax1.get_legend_handles_labels()
    unique_handles_labels = list(dict(zip(labels, handles)).items())[:3]
    unique_labels, unique_handles = zip(*unique_handles_labels)

    fig.legend(unique_handles, unique_labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))

    plt.subplots_adjust(top=0.85, bottom=0.25, wspace=0.3) 

    plt.show()

    plt.savefig("diff_lut_comp.png")
    print("Chart saved to barchart_example.png")


if __name__ == '__main__':
    main()
