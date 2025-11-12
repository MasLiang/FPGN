import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_multi_seed_file(filename):
    """
    解析包含多个seed结果的文件，返回每个seed的test loss和accuracy数据
    假设文件中按顺序包含5个seed的完整训练过程
    """
    all_test_losses = []
    all_accuracies = []
    current_test_losses = []
    current_accuracies = []
    
    if not os.path.exists(filename):
        print(f"警告: 文件 {filename} 不存在")
        return [], []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        for line in lines:
            line = line.strip()
            
            test_match = re.search(r'####\s*test\s+loss\s*:\s*([\d.-]+)\s*####', line, re.IGNORECASE)
            if test_match:
                current_test_losses.append(float(test_match.group(1)))
            
            if 'dwn' in filename.lower():
                acc_match = re.search(r'Test\s+Accuracy:\s*([\d.]+)', line, re.IGNORECASE)
                if acc_match:
                    accuracy = float(acc_match.group(1)) * 100  
                    current_accuracies.append(accuracy)
            else:
                acc_match = re.search(r'Accuracy:\s*(\d+)/(\d+)', line, re.IGNORECASE)
                if acc_match:
                    correct = float(acc_match.group(1))
                    total = float(acc_match.group(2))
                    accuracy = (correct / total) * 100
                    current_accuracies.append(accuracy)
        
        if current_test_losses and current_accuracies:
            total_epochs = len(current_test_losses)
            epochs_per_seed = total_epochs // 5  
            
            print(f"文件 {filename}: 总计 {total_epochs} 个epoch, 每个seed {epochs_per_seed} 个epoch")
            
            for i in range(5):
                start_idx = i * epochs_per_seed
                end_idx = start_idx + epochs_per_seed
                
                seed_test_losses = current_test_losses[start_idx:end_idx]
                seed_accuracies = current_accuracies[start_idx:end_idx]
                
                if seed_test_losses:
                    all_test_losses.append(seed_test_losses)
                if seed_accuracies:
                    all_accuracies.append(seed_accuracies)
                    
    return all_test_losses, all_accuracies

def calculate_statistics(seeds_data):
    if not seeds_data:
        return [], [], []
    
    min_length = min(len(seed_data) for seed_data in seeds_data)
    
    aligned_data = [seed_data[:min_length] for seed_data in seeds_data]
    
    data_array = np.array(aligned_data)
    
    mean_values = np.mean(data_array, axis=0)
    std_values = np.std(data_array, axis=0)
    
    upper_bound = mean_values + std_values
    lower_bound = mean_values - std_values
    
    return mean_values, upper_bound, lower_bound

def plot_loss_curves():
    dwn_color = '#33a02c'   
    fpgn_color = '#1f78b4'  

    fpgn_seeds_test, fpgn_seeds_acc = parse_multi_seed_file('fpgn.txt')
    dwn_seeds_test, dwn_seeds_acc = parse_multi_seed_file('dwn.txt')
    
    print("多Seed数据摘要:")
    print(f"FPGN - Seeds数量: {len(fpgn_seeds_test)} (test loss), {len(fpgn_seeds_acc)} (accuracy)")
    print(f"DWN - Seeds数量: {len(dwn_seeds_test)} (test loss), {len(dwn_seeds_acc)} (accuracy)")
    
    if fpgn_seeds_test:
        lengths = [len(seed_data) for seed_data in fpgn_seeds_test]
        print(f"FPGN - 每个seed的epoch数: {lengths}")
    if dwn_seeds_test:
        lengths = [len(seed_data) for seed_data in dwn_seeds_test]
        print(f"DWN - 每个seed的epoch数: {lengths}")
    
    fpgn_test_mean, fpgn_test_upper, fpgn_test_lower = calculate_statistics(fpgn_seeds_test)
    fpgn_acc_mean, fpgn_acc_upper, fpgn_acc_lower = calculate_statistics(fpgn_seeds_acc)
    dwn_test_mean, dwn_test_upper, dwn_test_lower = calculate_statistics(dwn_seeds_test)
    dwn_acc_mean, dwn_acc_upper, dwn_acc_lower = calculate_statistics(dwn_seeds_acc)
    
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Loss', fontsize=14, fontweight='bold', color='black')
    
    if len(fpgn_test_mean) > 0:
        epochs = range(1, len(fpgn_test_mean) + 1)
        ax1.plot(epochs, fpgn_test_mean, linewidth=2, 
                color=fpgn_color, label='FPGN Test Loss', zorder=3)
        ax1.fill_between(epochs, fpgn_test_lower, fpgn_test_upper, 
                        alpha=0.25, color=fpgn_color, zorder=1)
    
    if len(dwn_test_mean) > 0:
        epochs = range(1, len(dwn_test_mean) + 1)
        ax1.plot(epochs, dwn_test_mean, linewidth=2, 
                color=dwn_color, label='DWN Test Loss', zorder=3)
        ax1.fill_between(epochs, dwn_test_lower, dwn_test_upper, 
                        alpha=0.25, color=dwn_color, zorder=1)
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.tick_params(axis='both', labelsize=12)  
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='black')
    
    if len(fpgn_acc_mean) > 0:
        epochs = range(1, len(fpgn_acc_mean) + 1)
        ax2.plot(epochs, fpgn_acc_mean, linewidth=2, 
                color=fpgn_color, label='FPGN Accuracy', linestyle='--', zorder=3)
        ax2.fill_between(epochs, fpgn_acc_lower, fpgn_acc_upper, 
                        alpha=0.2, color=fpgn_color, zorder=1)
    
    if len(dwn_acc_mean) > 0:
        epochs = range(1, len(dwn_acc_mean) + 1)
        ax2.plot(epochs, dwn_acc_mean, linewidth=2, 
                color=dwn_color, label='DWN Accuracy', linestyle='--', zorder=3)
        ax2.fill_between(epochs, dwn_acc_lower, dwn_acc_upper, 
                        alpha=0.2, color=dwn_color, zorder=1)
    
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelsize=12)  
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=12, ncol=2, 
               bbox_to_anchor=(1.03, 0.55), columnspacing=0.5, frameon=False)
    legend.set_zorder(0)  
    
    plt.tight_layout()
    
    plt.savefig('loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print_statistical_report(fpgn_test_mean, fpgn_acc_mean, dwn_test_mean, dwn_acc_mean,
                           fpgn_seeds_acc, dwn_seeds_acc)

def parse_runtime_data(filename):
    runtime_values = []
    
    if not os.path.exists(filename):
        return []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        for line in lines:
            line = line.strip()
            
            runtime_patterns = [
                r'runtime[:：]\s*([\d.]+)',
                r'run\s+time[:：]\s*([\d.]+)',
                r'total\s+time[:：]\s*([\d.]+)',
                r'total_time[:：]\s*([\d.]+)',
                r'time[:：]\s*([\d.]+)\s*s',
                r'elapsed[:：]\s*([\d.]+)',
                r'duration[:：]\s*([\d.]+)'
            ]
            
            for pattern in runtime_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    runtime_values.append(float(match.group(1)))
                    break
                    
    return runtime_values

def plot_runtime_comparison():
    fpgn_runtimes = parse_runtime_data('fpgn.txt')
    dwn_runtimes = parse_runtime_data('dwn.txt')
    
    fpgn_avg_runtime = np.mean(fpgn_runtimes) if fpgn_runtimes else 0
    dwn_avg_runtime = np.mean(dwn_runtimes) if dwn_runtimes else 1  
    
    if dwn_avg_runtime == 0:
        dwn_avg_runtime = 1
    
    fpgn_ratio = fpgn_avg_runtime / dwn_avg_runtime if dwn_avg_runtime > 0 else 0
    dwn_ratio = 1.0  
    
    categories = ['Run Time', 'Memory']
    dwn_values = [dwn_ratio, 1.0]  
    fpgn_values = [fpgn_ratio, 2783/11503]  
    
    x = np.arange(len(categories))  
    width = 0.35  
    
    dwn_color = '#b2df8a'   
    fpgn_color = '#a6cee3'  
    
    fig, ax = plt.subplots(figsize=(3, 3.5))
    
    bars1 = ax.bar(x - width/2, dwn_values, width, label='DWN', color=dwn_color, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, fpgn_values, width, label='FPGN', color=fpgn_color, alpha=0.8, edgecolor='black', linewidth=1)
    
    for i in range(len(categories)):
        dwn_height = dwn_values[i]
        fpgn_height = fpgn_values[i]
        
        change_percent = ((fpgn_height - dwn_height) / dwn_height) * 100
        
        if change_percent > 0:
            change_text = f'+{change_percent:.0f}%'
            text_color = 'red'
        else:
            change_text = f'{change_percent:.0f}%'
            text_color = 'green'
        
        higher_height = max(dwn_height, fpgn_height)
        lower_height = min(dwn_height, fpgn_height)
        
        x_pos = x[i]
        
        if i == 0:  
            line_color = 'green' if dwn_height > fpgn_height else 'red'
        else:  
            line_color = 'red' if dwn_height < fpgn_height else 'green'
        
        line_extend = 0.15  
        ax.plot([x_pos - line_extend, x_pos + line_extend], 
                [higher_height, higher_height], 
                color=line_color, linewidth=1.5)
        
        ax.plot([x_pos - line_extend, x_pos + line_extend], 
                [lower_height, lower_height], 
                color=line_color, linewidth=1.5)
        
        arrow_x = x_pos + 0.2  
        ax.annotate('', xy=(arrow_x, lower_height), xytext=(arrow_x, higher_height),
                   arrowprops=dict(arrowstyle='<->', color=text_color, lw=2))
        
        text_y = (higher_height + lower_height) / 2
        ax.text(arrow_x + 0.05, text_y, change_text, 
                ha='left', va='center', fontsize=11, 
                fontweight='bold', color=text_color,
                rotation=90)  
    
    ax.set_ylabel('Relative Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    
    ax.tick_params(axis='y', labelsize=12)
    
    max_value = max(max(dwn_values), max(fpgn_values))
    ax.set_ylim(0, max_value * 1.2)
    ax.set_xlim(-0.5, len(categories) - 0.2)  
    ax.grid(True, alpha=0.3, linestyle='-', axis='y')
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper right', fontsize=12, columnspacing=0.5, 
              bbox_to_anchor=(1.03, 1.0))
    
    plt.tight_layout()
    
    plt.savefig('runtime_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    plot_loss_curves()
    plot_runtime_comparison()
