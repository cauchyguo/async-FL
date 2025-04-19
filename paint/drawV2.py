import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import re

# 自定义算法排序顺序
custom_order = ["FLEC", "HiFlash", "FedAT"]

# 创建排序键函数
def custom_sort(item):
    # 对于列表中的项目，返回其索引
    for i, prefix in enumerate(custom_order):
        if item.startswith(prefix):
            return i
    # 对于不在列表中的项目，返回一个较大的数，使它们排在最后
    return len(custom_order)

def plot_federated_learning_performance(
    main_dir, 
    max_epoch=500,  # 默认最大轮数
    max_time=50,    # 默认最大时间（分钟）
    save_dir=None,  # 保存图片的目录，默认为main_dir
    fig_size=(12, 8),
    dpi=300,
    dataset_name=None,  # 数据集名称，如果为None则从目录名提取
    dir_alpha=None      # Dir参数值，如果为None则从目录名提取
):
    """
    从各个联邦学习算法目录中读取性能数据并绘制对比图
    
    参数:
        main_dir (str): 主实验目录路径
        max_epoch (int): 要绘制的最大轮数
        max_time (int): 时间图的最大时间限制（分钟）
        save_dir (str): 保存图片的目录，默认为main_dir
        fig_size (tuple): 图表大小
        dpi (int): 图表DPI
        dataset_name (str): 数据集名称，如果为None则从目录名提取
        dir_alpha (str): Dir参数值，如果为None则从目录名提取
        
    返回:
        dict: 包含成功加载的算法数据和图表路径
    """
    # 如果未指定保存目录，使用主目录
    if save_dir is None:
        save_dir = main_dir
    
    # 如果未指定数据集名称和dir_alpha，从目录名提取
    if dataset_name is None or dir_alpha is None:
        dir_name = os.path.basename(main_dir)
        # 尝试从目录名解析数据集名称和dir值
        match = re.match(r".*-(\w+)-dir([\d\.]+)", dir_name)
        if match:
            dataset_name = match.group(1) if dataset_name is None else dataset_name
            dir_alpha = match.group(2) if dir_alpha is None else dir_alpha
        else:
            dataset_name = "Unknown" if dataset_name is None else dataset_name
            dir_alpha = "Unknown" if dir_alpha is None else dir_alpha
    
    # 设置字体
    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.grid'] = True
    
    # 用于存储各算法数据的字典
    algo_data = {}
    results = {
        "loaded_algorithms": [],
        "skipped_algorithms": [],
        "figures": {}
    }

    # 遍历主目录找出所有算法子目录
    algo_dirs = []
    for item in os.listdir(main_dir):
        full_path = os.path.join(main_dir, item)
        if os.path.isdir(full_path):
            algo_dirs.append(item)
    
    # 按照自定义顺序排序算法
    algo_dirs.sort(key=custom_sort)
    
    # 遍历所有算法目录
    for algo_dir in algo_dirs:
        full_path = os.path.join(main_dir, algo_dir)
        
        # 提取算法名称（去掉-dir部分）
        algo_name = re.sub(r'-dir[\d\.]+$', '', algo_dir)
        
        # 尝试读取三个文件
        try:
            accuracy_path = os.path.join(full_path, 'accuracy.txt')
            loss_path = os.path.join(full_path, 'loss.txt')
            update_time_path = os.path.join(full_path, 'update_time.txt')
            
            # 检查文件是否都存在
            if os.path.exists(accuracy_path) and os.path.exists(loss_path) and os.path.exists(update_time_path):
                # 读取文件内容
                with open(accuracy_path, 'r') as f:
                    accuracy_data = eval(f.read())  # 假设是Python列表格式
                
                with open(loss_path, 'r') as f:
                    loss_data = eval(f.read())
                
                with open(update_time_path, 'r') as f:
                    time_data = eval(f.read())
                
                # 存储数据
                algo_data[algo_name] = {
                    'accuracy': accuracy_data[:max_epoch],
                    'loss': loss_data[:max_epoch],
                    'time': time_data[:max_epoch]
                }
                
                results["loaded_algorithms"].append(algo_name)
                print(f"成功加载 {algo_name} 的数据")
            else:
                missing_files = []
                if not os.path.exists(accuracy_path): missing_files.append("accuracy.txt")
                if not os.path.exists(loss_path): missing_files.append("loss.txt")
                if not os.path.exists(update_time_path): missing_files.append("update_time.txt")
                
                results["skipped_algorithms"].append(f"{algo_name} (缺失文件: {', '.join(missing_files)})")
                print(f"跳过 {algo_name}: 缺少必要的文件 {', '.join(missing_files)}")
        except Exception as e:
            results["skipped_algorithms"].append(f"{algo_name} (读取错误: {str(e)})")
            print(f"读取 {algo_name} 数据时出错: {e}")
    
    # 检查是否成功加载了数据
    if not algo_data:
        print("未找到任何有效数据，请检查目录结构和文件")
        return results
    
    # 1. 绘制准确率随轮数变化图
    plt.figure(figsize=fig_size)
    
    for algo_name, data in algo_data.items():
        epochs = list(range(1, len(data['accuracy']) + 1))
        plt.plot(epochs, data['accuracy'], 
                linewidth=2, marker='o', markersize=2, label=algo_name)
    
    plt.xlabel('全局轮数', fontsize=20)
    plt.ylabel('准确率（%）', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=20)
    
    # 设置刻度字体为Times New Roman
    plt.xticks(fontproperties='Times New Roman', fontsize=18)
    plt.yticks(fontproperties='Times New Roman', fontsize=18)
    
    plt.tight_layout()
    
    # 保存图片
    file_name_head = os.path.basename(main_dir)
    accuracy_plot_path = os.path.join(save_dir, f'{file_name_head}_accuracy_vs_epoch_{max_epoch}.png')
    plt.savefig(accuracy_plot_path, dpi=dpi)
    results["figures"]["accuracy_plot"] = accuracy_plot_path
    print(f"保存图表: {accuracy_plot_path}")
    
    # 2. 绘制损失值随轮数变化图
    plt.figure(figsize=fig_size)
    
    for algo_name, data in algo_data.items():
        epochs = list(range(1, len(data['loss']) + 1))
        plt.plot(epochs, data['loss'], 
                linewidth=2, marker='o', markersize=2, label=algo_name)
    
    plt.xlabel('全局轮数', fontsize=20)
    plt.ylabel('损失值', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=20)
    
    # 设置刻度字体为Times New Roman
    plt.xticks(fontproperties='Times New Roman', fontsize=18)
    plt.yticks(fontproperties='Times New Roman', fontsize=18)
    
    plt.tight_layout()
    
    # 保存图片
    loss_plot_path = os.path.join(save_dir, f'{file_name_head}_loss_vs_epoch_{max_epoch}.png')
    plt.savefig(loss_plot_path, dpi=dpi)
    results["figures"]["loss_plot"] = loss_plot_path
    print(f"保存图表: {loss_plot_path}")
    
    # 3. 绘制准确率随更新时间变化图
    plt.figure(figsize=fig_size)
    
    for algo_name, data in algo_data.items():
        # 过滤掉超过max_time的数据点
        valid_indices = [i for i, t in enumerate(data['time']) if t <= max_time * 60]  # 转换为秒
        if valid_indices:
            filtered_time = [data['time'][i] / 60 for i in valid_indices]  # 转换为分钟
            filtered_accuracy = [data['accuracy'][i] for i in valid_indices]
            plt.plot(filtered_time, filtered_accuracy, 
                    linewidth=2, marker='o', markersize=2, label=algo_name)
    
    plt.xlabel('挂钟时间（分钟）', fontsize=20)
    plt.ylabel('准确率（%）', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=20)
    
    # 设置刻度字体为Times New Roman
    plt.xticks(fontproperties='Times New Roman', fontsize=18)
    plt.yticks(fontproperties='Times New Roman', fontsize=18)
    
    plt.tight_layout()
    
    # 保存图片
    time_plot_path = os.path.join(save_dir, f'{file_name_head}_accuracy_vs_time_{max_epoch}.png')
    plt.savefig(time_plot_path, dpi=dpi)
    results["figures"]["time_plot"] = time_plot_path
    print(f"保存图表: {time_plot_path}")
    
    print("绘图完成！")
    return results