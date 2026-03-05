import matplotlib.pyplot as plt
import numpy as np
import re
import os
from scipy import stats

def parse_loss_file(filename):
    """
    解析文件，提取test loss和train loss的数据
    """
    test_losses = []
    train_losses = []
    
    if not os.path.exists(filename):
        print(f"警告: 文件 {filename} 不存在")
        return test_losses, train_losses
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                
                # 查找test loss模式: #### test loss : xxxx ####
                test_match = re.search(r'####\s*test\s+loss\s*:\s*([\d.-]+)\s*####', line, re.IGNORECASE)
                if test_match:
                    test_losses.append(float(test_match.group(1)))
                
                # 查找train loss模式: #### train loss : xxxx ####
                train_match = re.search(r'####\s*train\s+loss\s*:\s*([\d.-]+)\s*####', line, re.IGNORECASE)
                if train_match:
                    train_losses.append(float(train_match.group(1)))
                    
    except Exception as e:
        print(f"读取文件 {filename} 时出错: {e}")
    
    return test_losses, train_losses

def analyze_convergence_pattern(losses, name):
    """
    分析收敛模式
    """
    if len(losses) == 0:
        return
    
    print(f"\n=== {name} 收敛分析 ===")
    
    # 基本统计
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)
    
    print(f"初始损失: {initial_loss:.4f}")
    print(f"最终损失: {final_loss:.4f}")
    print(f"最小损失: {min_loss:.4f}")
    print(f"最大损失: {max_loss:.4f}")
    print(f"总体下降: {initial_loss - final_loss:.4f} ({((initial_loss - final_loss)/initial_loss*100):.2f}%)")
    
    # 收敛速度分析
    # 计算每个阶段的下降速度
    if len(losses) > 10:
        early_stage = losses[:10]  # 前10个epoch
        middle_stage = losses[10:len(losses)//2] if len(losses) > 20 else []
        late_stage = losses[-10:]  # 后10个epoch
        
        early_drop = early_stage[0] - early_stage[-1] if len(early_stage) > 1 else 0
        late_drop = late_stage[0] - late_stage[-1] if len(late_stage) > 1 else 0
        
        print(f"早期下降速度 (前10轮): {early_drop:.4f}")
        print(f"后期下降速度 (后10轮): {late_drop:.4f}")
        
        # 稳定性分析
        late_std = np.std(late_stage)
        print(f"后期稳定性 (标准差): {late_std:.4f}")
        
        if late_std < 0.01:
            print("状态: 已收敛 (稳定)")
        elif late_std < 0.05:
            print("状态: 基本收敛")
        else:
            print("状态: 仍在变化")
    
    # 趋势分析
    x = np.arange(len(losses))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, losses)
    print(f"整体趋势斜率: {slope:.6f}")
    print(f"相关系数: {r_value:.4f}")
    
    return {
        'initial': initial_loss,
        'final': final_loss,
        'min': min_loss,
        'max': max_loss,
        'total_drop': initial_loss - final_loss,
        'drop_percentage': (initial_loss - final_loss)/initial_loss*100,
        'slope': slope,
        'correlation': r_value,
        'stability': np.std(losses[-10:]) if len(losses) >= 10 else np.std(losses)
    }

def compare_methods(fpgn_stats, dwn_stats):
    """
    比较两种方法
    """
    print("\n" + "="*50)
    print("方法对比分析")
    print("="*50)
    
    print(f"\n📊 收敛效果:")
    print(f"FPGN 总下降: {fpgn_stats['train']['total_drop']:.4f} ({fpgn_stats['train']['drop_percentage']:.2f}%)")
    print(f"DWN  总下降: {dwn_stats['train']['total_drop']:.4f} ({dwn_stats['train']['drop_percentage']:.2f}%)")
    
    print(f"\n🎯 最终性能:")
    print(f"FPGN 最终训练损失: {fpgn_stats['train']['final']:.4f}")
    print(f"DWN  最终训练损失: {dwn_stats['train']['final']:.4f}")
    print(f"FPGN 最终测试损失:  {fpgn_stats['test']['final']:.4f}")
    print(f"DWN  最终测试损失:  {dwn_stats['test']['final']:.4f}")
    
    print(f"\n📈 收敛趋势:")
    print(f"FPGN 训练趋势斜率: {fpgn_stats['train']['slope']:.6f}")
    print(f"DWN  训练趋势斜率: {dwn_stats['train']['slope']:.6f}")
    
    print(f"\n🔄 稳定性:")
    print(f"FPGN 训练稳定性: {fpgn_stats['train']['stability']:.4f}")
    print(f"DWN  训练稳定性: {dwn_stats['train']['stability']:.4f}")
    print(f"FPGN 测试稳定性:  {fpgn_stats['test']['stability']:.4f}")
    print(f"DWN  测试稳定性:  {dwn_stats['test']['stability']:.4f}")
    
    # 过拟合分析
    print(f"\n⚠️  过拟合分析:")
    fpgn_gap = fpgn_stats['train']['final'] - fpgn_stats['test']['final']
    dwn_gap = dwn_stats['train']['final'] - dwn_stats['test']['final']
    print(f"FPGN 训练-测试gap: {fpgn_gap:.4f}")
    print(f"DWN  训练-测试gap: {dwn_gap:.4f}")
    
    if abs(fpgn_gap) < abs(dwn_gap):
        print("FPGN 过拟合程度较低")
    else:
        print("DWN 过拟合程度较低")

def identify_patterns():
    """
    识别训练模式和规律
    """
    # 解析数据
    fpgn_test, fpgn_train = parse_loss_file('fpgn.txt')
    dwn_test, dwn_train = parse_loss_file('dwn.txt')
    
    print("🔍 训练损失数据规律分析")
    print("="*60)
    
    # 分析每个数据集
    fpgn_train_stats = analyze_convergence_pattern(fpgn_train, "FPGN 训练损失")
    fpgn_test_stats = analyze_convergence_pattern(fpgn_test, "FPGN 测试损失")
    dwn_train_stats = analyze_convergence_pattern(dwn_train, "DWN 训练损失")
    dwn_test_stats = analyze_convergence_pattern(dwn_test, "DWN 测试损失")
    
    # 整理统计结果
    fpgn_stats = {'train': fpgn_train_stats, 'test': fpgn_test_stats}
    dwn_stats = {'train': dwn_train_stats, 'test': dwn_test_stats}
    
    # 方法比较
    compare_methods(fpgn_stats, dwn_stats)
    
    # 总结规律
    print("\n" + "="*50)
    print("🎯 主要发现和规律")
    print("="*50)
    
    print("\n1️⃣ 收敛模式:")
    if dwn_train_stats and dwn_train_stats['total_drop'] > 1.0:
        print("   • DWN 表现出强烈的学习能力，损失大幅下降")
    if fpgn_train_stats and fpgn_train_stats['stability'] < 0.05:
        print("   • FPGN 训练过程相对稳定，变化较小")
    
    print("\n2️⃣ 性能特征:")
    if dwn_train_stats and dwn_train_stats['final'] < 0.3:
        print("   • DWN 能够达到很低的训练损失")
    if fpgn_test_stats and dwn_test_stats:
        if fpgn_test_stats['final'] < dwn_test_stats['final']:
            print("   • FPGN 在测试集上表现更好")
        else:
            print("   • DWN 在测试集上表现更好")
    
    print("\n3️⃣ 训练动态:")
    if dwn_train_stats and dwn_train_stats['slope'] < -0.01:
        print("   • DWN 显示持续的下降趋势")
    if fpgn_train_stats and abs(fpgn_train_stats['slope']) < 0.001:
        print("   • FPGN 训练损失基本平稳")
    
    print("\n4️⃣ 实际应用建议:")
    if dwn_train_stats and dwn_test_stats:
        gap = dwn_train_stats['final'] - dwn_test_stats['final']
        if gap > 0.5:
            print("   • DWN 可能存在过拟合，需要正则化")
        else:
            print("   • DWN 训练效果良好，泛化能力可接受")
    
    if fpgn_train_stats and fpgn_train_stats['stability'] < 0.02:
        print("   • FPGN 适合需要稳定性的应用场景")

if __name__ == "__main__":
    identify_patterns()