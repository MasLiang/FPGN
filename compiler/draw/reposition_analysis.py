import matplotlib.pyplot as plt
import numpy as np

def plot_merged_bar_chart(data_group1, data_group2, group_names=None, subgroup_names=None, 
                         title="Merged Bar Chart", save_path=None):
    """
    Plot merged bar chart with all data in one chart using dual y-axes
    
    Parameters:
    data_group1: list of tuples, data for first main group, each tuple contains (A_count, B_count)
    data_group2: list of tuples, data for second main group, each tuple contains (A_count, B_count)
    group_names: list, names for the two main groups, default ["Group1", "Group2"]
    subgroup_names: list, names for the six subgroups, default ["Sub1", "Sub2", ...]
    title: str, chart title
    save_path: str, save path, if None then don't save
    """
    
    if group_names is None:
        group_names = ["Baseline", "Reposition"]
    if subgroup_names is None:
        subgroup_names = [f"Sub{i+1}" for i in range(6)]
    
    group1_A = [item[0] for item in data_group1]
    group1_B = [item[1] for item in data_group1]
    group2_A = [item[0] for item in data_group2]
    group2_B = [item[1] for item in data_group2]
    
    x = np.arange(len(subgroup_names))
    width = 0.2  
    
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    
    bars1_A = ax1.bar(x - 1.5*width, group1_A, width, label=f'{group_names[0]} - DSP', 
                      alpha=0.8, color='#b2df8a')
    bars2_A = ax1.bar(x - 0.5*width, group2_A, width, label=f'{group_names[1]} - DSP', 
                      alpha=0.8, color='#33a02c')
    
    ax1.set_xlabel('Layer Index', fontsize=22)
    ax1.set_ylabel('DSP', fontsize=22, color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subgroup_names, rotation=0, ha='center', fontsize=20)
    ax1.tick_params(axis='y', labelsize=20, labelcolor='black')
    ax1.grid(True, alpha=0.3, axis='y')
    
    max_A = max(max(group1_A), max(group2_A))
    if max_A > 0:
        ax1.set_ylim(0, max_A * 1.2)
    
    ax2 = ax1.twinx()
    
    bars1_B = ax2.bar(x + 0.5*width, group1_B, width, label=f'{group_names[0]} - LUTs', 
                      alpha=0.8, color='#1f78b4')
    bars2_B = ax2.bar(x + 1.5*width, group2_B, width, label=f'{group_names[1]} - LUTs', 
                      alpha=0.8, color='#a6cee3')
    
    ax2.set_ylabel('LUTs(k)', fontsize=22, color='black')
    ax2.tick_params(axis='y', labelsize=20, labelcolor='black')
    
    max_B = max(max(group1_B), max(group2_B))
    ax2.set_ylim(0, max_B * 1.2)
    
    def add_value_labels_single_axis(ax, bars, offset=3):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=16)
    
    add_value_labels_single_axis(ax1, bars1_A, 3)
    add_value_labels_single_axis(ax1, bars2_A, 8)
    add_value_labels_single_axis(ax2, bars1_B, 3)
    add_value_labels_single_axis(ax2, bars2_B, 18)  
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=18)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()

def create_sample_data():
    """
    Create sample data - replace with your actual data
    """
    data_group1 = [
        (146,   560),  
        ( 48,   960),  
        (290,  1920),  
        ( 96,  3840),  
        (578,  7680),  
        (192, 12288)   
    ]
    
    data_group2 = [
        (0,   624),  
        (0,  1040),  
        (0,  2080),  
        (0,  4032),  
        (0,  8064),  
        (0, 12736)   
    ]
    
    return data_group1, data_group2

def main():
    print("Merged Bar Chart Generator")
    print("=" * 50)
    print("Features:")
    print("- Single chart with dual y-axes")
    print("- DSP data on left y-axis (blue tones)")
    print("- LUTs data on right y-axis (red tones)")
    print("- All data visible in one view")
    print("- Enlarged fonts for better readability")
    print("=" * 50)
    
    data_group1, data_group2 = create_sample_data()
    
    group_names = ["Baseline", "Reposition"]
    subgroup_names = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5", "Conv6"]
    
    plot_merged_bar_chart(
        data_group1, 
        data_group2, 
        group_names=group_names,
        subgroup_names=subgroup_names,
        title="Merged Data Comparison",
        save_path="reposition_analysis.png"
    )
    
    print("\nData Summary:")
    print(f"{group_names[0]} Group:")
    for i, (a, b) in enumerate(data_group1):
        print(f"  {subgroup_names[i]}: DSP={a}, LUTs={b}")
    
    print(f"\n{group_names[1]} Group:")
    for i, (a, b) in enumerate(data_group2):
        print(f"  {subgroup_names[i]}: DSP={a}, LUTs={b}")

if __name__ == "__main__":
    main()
