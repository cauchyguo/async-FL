import re
import pandas as pd

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from pathlib import Path
# 定义自定义顺序
custom_order = ["FedAGSA", "PyramidFL","FedDocs","TiFL","FedAvg",]

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
    max_time=50,  # 默认50分钟 
    total_rounds=500,  # 默认500轮
    save_epoch_plot=True,
    save_time_plot=True,
    save_loss_plot=True,
    fig_size=(12, 8),
    
    dpi=300,
    dataset_name="Fashion-MNIST",
    dir_alpha="0.5"
):
    """
    从各个联邦学习算法目录中读取性能数据并绘制对比图
    
    参数:
        main_dir (str): 主实验目录路径
        max_time (int): 时间图的最大时间限制（秒）
        total_rounds (int): 轮数图的最大轮数限制
        save_epoch_plot (bool): 是否保存轮数图
        save_time_plot (bool): 是否保存时间图
        fig_size (tuple): 图表大小
        dpi (int): 图表DPI
        dataset_name (str): 数据集名称
        dir_alpha (str): Dir参数值
        
    返回:
        dict: 包含成功加载的算法数据和图表路径
    """
    # 设置字体
    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 用于存储各算法数据的字典
    accuracy_data = {}
    results = {
        "loaded_algorithms": [],
        "skipped_algorithms": [],
        "figures": {}
    }

    # 遍历所有算法目录
    for algo_dir in sorted(os.listdir(main_dir),key=custom_sort):
        full_path = os.path.join(main_dir, algo_dir)
        
        # 确保是目录
        if os.path.isdir(full_path):
            csv_path = os.path.join(full_path, 'epoch_results_with_cumtime.csv')
            
            # 检查CSV文件是否存在
            if os.path.exists(csv_path):
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 确保CSV包含所需列
                    if 'epoch' in df.columns and 'accuracy' in df.columns and 'epoch_cum_time' in df.columns:
                        # 存储数据 - 算法名作为键
                        accuracy_data[algo_dir] = {
                            'epoch': df['epoch'].iloc[:total_rounds].values,
                            'accuracy': df['accuracy'].iloc[:total_rounds].values,
                            'cum_time': df['epoch_cum_time'].iloc[:total_rounds].values / 60,
                            'loss': df['loss'].iloc[:total_rounds].values
                        }
                        results["loaded_algorithms"].append(algo_dir)
                        print(f"成功加载 {algo_dir} 的数据")
                    else:
                        results["skipped_algorithms"].append(f"{algo_dir} (缺少必要的列)")
                        print(f"警告: {csv_path} 缺少必要的列")
                except Exception as e:
                    results["skipped_algorithms"].append(f"{algo_dir} (读取错误: {str(e)})")
                    print(f"读取 {csv_path} 时出错: {e}")
            else:
                results["skipped_algorithms"].append(f"{algo_dir} (未找到CSV文件)")
                print(f"跳过 {algo_dir}: 未找到 epoch_results_with_cumtime.csv 文件")

    # 检查是否成功加载了数据
    if not accuracy_data:
        print("未找到任何有效数据，请检查目录结构和CSV文件")
        return results
    
    # 使用默认样式，启用网格以增强可读性
    plt.rcParams['axes.grid'] = True
    
    # 绘制第一张图: 准确率 vs 轮数
    if save_epoch_plot:
        plt.figure(figsize=fig_size)
        
        for algo_name, data in accuracy_data.items():
            algo_name = algo_name.split('-')[0]
            plt.plot(data['epoch'][:total_rounds], data['accuracy'][:total_rounds], 
                    linewidth=2, marker='o', markersize=2, label=algo_name)
        
        # plt.title(f'{dataset_name} (Dir={dir_alpha}): 准确率 vs 轮数', fontsize=14)
        plt.xlabel('全局轮数', fontsize=20)
        plt.ylabel('准确率（%）', fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=20)
        
        # 设置刻度字体为Times New Roman
        plt.xticks(fontproperties='Times New Roman',fontsize=18)
        plt.yticks(fontproperties='Times New Roman',fontsize=18)
        
        plt.tight_layout()
        file_name_head = main_dir.split("/")[-1]
        file_name_tail = str(total_rounds)
        epoch_plot_path = os.path.join(main_dir, f'{file_name_head}_accuracy_vs_epoch_{file_name_tail}.png')
        plt.savefig(epoch_plot_path, dpi=dpi)
        results["figures"]["epoch_plot"] = epoch_plot_path
        print(f"保存图表: {epoch_plot_path}")
    
    # 绘制第二张图: 准确率 vs 累计时间
    if save_time_plot:
        plt.figure(figsize=fig_size)
        
        for algo_name, data in accuracy_data.items():
            algo_name = algo_name.split('-')[0]
            # 使用NumPy布尔索引
            mask = data['cum_time'] <= max_time
            plt.plot(data['cum_time'][mask], data['accuracy'][mask], 
                    linewidth=2, marker='o', markersize=2, label=algo_name)
        
        # plt.title(f'{dataset_name} (Dir={dir_alpha}): 准确率 vs 累计时间', fontsize=14)
        plt.xlabel('挂钟时间（分钟）', fontsize=20)
        plt.ylabel('准确率（%）', fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=20)
        
        # 设置刻度字体为Times New Roman
        plt.xticks(fontproperties='Times New Roman',fontsize=18)
        plt.yticks(fontproperties='Times New Roman',fontsize=18)
        
        plt.tight_layout()

        file_name_head = main_dir.split("/")[-1]
        file_name_tail = str(total_rounds)
        time_plot_path = os.path.join(main_dir, f'{file_name_head}_accuracy_vs_time_{file_name_tail}.png')
        plt.savefig(time_plot_path, dpi=dpi)
        results["figures"]["time_plot"] = time_plot_path
        print(f"保存图表: {time_plot_path}")

    # 绘制第三张图: 损失 vs 轮数
    if save_loss_plot:
        plt.figure(figsize=fig_size)
        
        for algo_name, data in accuracy_data.items():
            algo_name = algo_name.split('-')[0]
            # 使用NumPy布尔索引
            # mask = data['cum_time'] <= max_time
            plt.plot(data['epoch'][:total_rounds], data['loss'][:total_rounds], 
                    linewidth=2, marker='o', markersize=2, label=algo_name)
        
        # plt.title(f'{dataset_name} (Dir={dir_alpha}): 准确率 vs 累计时间', fontsize=14)
        plt.xlabel('全局轮数', fontsize=20)
        plt.ylabel('损失值', fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=20)
        
        # 设置刻度字体为Times New Roman
        plt.xticks(fontproperties='Times New Roman',fontsize=18)
        plt.yticks(fontproperties='Times New Roman',fontsize=18)
        
        plt.tight_layout()

        file_name_head = main_dir.split("/")[-1]
        file_name_tail = str(total_rounds)
        loss_plot_path = os.path.join(main_dir, f'{file_name_head}_loss_vs_epoch_{file_name_tail}.png')
        plt.savefig(loss_plot_path, dpi=dpi)
        results["figures"]["loss_plot"] = loss_plot_path
        print(f"保存图表: {loss_plot_path}")
    
    print("绘图完成！")
    return results


def plot_federated_learning_performance_lambda(
    main_dir, 
    max_time=50,  # 默认50分钟 
    total_rounds=500,  # 默认500轮
    save_epoch_plot=True,
    save_time_plot=True,
    fig_size=(12, 8),
    
    dpi=300,
    dataset_name="Fashion-MNIST",
    dir_alpha="0.5"
):
    """
    从各个联邦学习算法目录中读取性能数据并绘制对比图
    
    参数:
        main_dir (str): 主实验目录路径
        max_time (int): 时间图的最大时间限制（秒）
        total_rounds (int): 轮数图的最大轮数限制
        save_epoch_plot (bool): 是否保存轮数图
        save_time_plot (bool): 是否保存时间图
        fig_size (tuple): 图表大小
        dpi (int): 图表DPI
        dataset_name (str): 数据集名称
        dir_alpha (str): Dir参数值
        
    返回:
        dict: 包含成功加载的算法数据和图表路径
    """
    # 设置字体
    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 用于存储各算法数据的字典
    accuracy_data = {}
    results = {
        "loaded_algorithms": [],
        "skipped_algorithms": [],
        "figures": {}
    }
    custom_order = ["lambda0", "lambda0.01","lambda0.02","lambda0.035"]

    # 创建排序键函数
    def custom_sort(item):
        # 对于列表中的项目，返回其索引
        for i, prefix in enumerate(custom_order):

            if item.split('-')[-2].endswith(prefix):
                return i
        # 对于不在列表中的项目，返回一个较大的数，使它们排在最后
        return len(custom_order)
    # 遍历所有算法目录
    for algo_dir in sorted(os.listdir(main_dir),key=custom_sort):
        full_path = os.path.join(main_dir, algo_dir)
        
        # 确保是目录
        if os.path.isdir(full_path):
            csv_path = os.path.join(full_path, 'epoch_results_with_cumtime.csv')
            
            # 检查CSV文件是否存在
            if os.path.exists(csv_path):
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 确保CSV包含所需列
                    if 'epoch' in df.columns and 'accuracy' in df.columns and 'epoch_cum_time' in df.columns:
                        # 存储数据 - 算法名作为键
                        accuracy_data[algo_dir] = {
                            'epoch': df['epoch'].iloc[:total_rounds].values,
                            'accuracy': df['accuracy'].iloc[:total_rounds].values,
                            'cum_time': df['epoch_cum_time'].iloc[:total_rounds].values / 60
                        }
                        results["loaded_algorithms"].append(algo_dir)
                        print(f"成功加载 {algo_dir} 的数据")
                    else:
                        results["skipped_algorithms"].append(f"{algo_dir} (缺少必要的列)")
                        print(f"警告: {csv_path} 缺少必要的列")
                except Exception as e:
                    results["skipped_algorithms"].append(f"{algo_dir} (读取错误: {str(e)})")
                    print(f"读取 {csv_path} 时出错: {e}")
            else:
                results["skipped_algorithms"].append(f"{algo_dir} (未找到CSV文件)")
                print(f"跳过 {algo_dir}: 未找到 epoch_results_with_cumtime.csv 文件")

    # 检查是否成功加载了数据
    if not accuracy_data:
        print("未找到任何有效数据，请检查目录结构和CSV文件")
        return results
    
    # 使用默认样式，启用网格以增强可读性
    plt.rcParams['axes.grid'] = True
    
    # 绘制第一张图: 准确率 vs 轮数
    if save_epoch_plot:
        plt.figure(figsize=fig_size)
        
        for algo_name, data in accuracy_data.items():
            param = algo_name.split('-')[-2]
            # param_name = param[:-1]
            param_name = r'$\lambda$'
            import re
            pattern = re.compile(r'lambda(\d+(?:\.\d+)?)')
            param_value = pattern.findall(param)[0]
            plt.plot(data['epoch'][:total_rounds], data['accuracy'][:total_rounds], 
                    linewidth=2, marker='o', markersize=2, label=f"{param_name}={param_value}")
        
        # plt.title(f'{dataset_name} (Dir={dir_alpha}): 准确率 vs 轮数', fontsize=14)
        plt.xlabel('全局轮数', fontsize=20)
        plt.ylabel('准确率（%）', fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=20)
        
        # 设置刻度字体为Times New Roman
        plt.xticks(fontproperties='Times New Roman',fontsize=18)
        plt.yticks(fontproperties='Times New Roman',fontsize=18)
        
        plt.tight_layout()
        file_name_head = main_dir.split("/")[-1]
        file_name_tail = str(total_rounds)
        epoch_plot_path = os.path.join(main_dir, f'{file_name_head}_accuracy_vs_epoch_{file_name_tail}.png')
        plt.savefig(epoch_plot_path, dpi=dpi)
        results["figures"]["epoch_plot"] = epoch_plot_path
        print(f"保存图表: {epoch_plot_path}")
    
    # # 绘制第二张图: 准确率 vs 累计时间
    # if save_time_plot:
    #     plt.figure(figsize=fig_size)
        
    #     for algo_name, data in accuracy_data.items():
    #         #algo_name = algo_name.split('-')[0]

    #         param = algo_name.split('-')[-1]
    #         param_name = param[:-1]
    #         # if param_name == 'epsilon':
    #         param_name = r'$\epsilon$'
    #         # elif param_name == 'lambda':
    #         #     param_name = r'$\lambda$'
    #         param_value = param[-1]
    #         # 使用NumPy布尔索引
    #         mask = data['cum_time'] <= max_time
    #         plt.plot(data['cum_time'][mask], data['accuracy'][mask], 
    #                 linewidth=2, marker='o', markersize=2, label=f"{param_name}={param_value}")
        
    #     # plt.title(f'{dataset_name} (Dir={dir_alpha}): 准确率 vs 累计时间', fontsize=14)
    #     plt.xlabel('挂钟时间（分钟）', fontsize=20)
    #     plt.ylabel('准确率（%）', fontsize=20)
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend(fontsize=20)
        
    #     # 设置刻度字体为Times New Roman
    #     plt.xticks(fontproperties='Times New Roman',fontsize=18)
    #     plt.yticks(fontproperties='Times New Roman',fontsize=18)
        
    #     plt.tight_layout()

    #     file_name_head = main_dir.split("/")[-1]
    #     file_name_tail = str(total_rounds)
    #     time_plot_path = os.path.join(main_dir, f'{file_name_head}_accuracy_vs_time_{file_name_tail}.png')
    #     plt.savefig(time_plot_path, dpi=dpi)
    #     results["figures"]["time_plot"] = time_plot_path
    #     print(f"保存图表: {time_plot_path}")
    
    # print("绘图完成！")
    # return results

def plot_federated_learning_performance_epsilon(
    main_dir, 
    max_time=50,  # 默认50分钟 
    total_rounds=500,  # 默认500轮
    save_epoch_plot=True,
    save_time_plot=True,
    fig_size=(12, 8),
    
    dpi=300,
    dataset_name="Fashion-MNIST",
    dir_alpha="0.5"
):
    """
    从各个联邦学习算法目录中读取性能数据并绘制对比图
    
    参数:
        main_dir (str): 主实验目录路径
        max_time (int): 时间图的最大时间限制（秒）
        total_rounds (int): 轮数图的最大轮数限制
        save_epoch_plot (bool): 是否保存轮数图
        save_time_plot (bool): 是否保存时间图
        fig_size (tuple): 图表大小
        dpi (int): 图表DPI
        dataset_name (str): 数据集名称
        dir_alpha (str): Dir参数值
        
    返回:
        dict: 包含成功加载的算法数据和图表路径
    """
    # 设置字体
    config = {
        "font.family": 'serif',
        "font.size": 20,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 用于存储各算法数据的字典
    accuracy_data = {}
    results = {
        "loaded_algorithms": [],
        "skipped_algorithms": [],
        "figures": {}
    }
    custom_order = ["epsilon0", "epsilon1","epsilon2"]

    # 创建排序键函数
    def custom_sort(item):
        # 对于列表中的项目，返回其索引
        for i, prefix in enumerate(custom_order):
            if item.endswith(prefix):
                return i
        # 对于不在列表中的项目，返回一个较大的数，使它们排在最后
        return len(custom_order)
    # 遍历所有算法目录
    for algo_dir in sorted(os.listdir(main_dir),key=custom_sort):
        full_path = os.path.join(main_dir, algo_dir)
        
        # 确保是目录
        if os.path.isdir(full_path):
            csv_path = os.path.join(full_path, 'epoch_results_with_cumtime.csv')
            
            # 检查CSV文件是否存在
            if os.path.exists(csv_path):
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    
                    # 确保CSV包含所需列
                    if 'epoch' in df.columns and 'accuracy' in df.columns and 'epoch_cum_time' in df.columns:
                        # 存储数据 - 算法名作为键
                        accuracy_data[algo_dir] = {
                            'epoch': df['epoch'].iloc[:total_rounds].values,
                            'accuracy': df['accuracy'].iloc[:total_rounds].values,
                            'cum_time': df['epoch_cum_time'].iloc[:total_rounds].values / 60
                        }
                        results["loaded_algorithms"].append(algo_dir)
                        print(f"成功加载 {algo_dir} 的数据")
                    else:
                        results["skipped_algorithms"].append(f"{algo_dir} (缺少必要的列)")
                        print(f"警告: {csv_path} 缺少必要的列")
                except Exception as e:
                    results["skipped_algorithms"].append(f"{algo_dir} (读取错误: {str(e)})")
                    print(f"读取 {csv_path} 时出错: {e}")
            else:
                results["skipped_algorithms"].append(f"{algo_dir} (未找到CSV文件)")
                print(f"跳过 {algo_dir}: 未找到 epoch_results_with_cumtime.csv 文件")

    # 检查是否成功加载了数据
    if not accuracy_data:
        print("未找到任何有效数据，请检查目录结构和CSV文件")
        return results
    
    # 使用默认样式，启用网格以增强可读性
    plt.rcParams['axes.grid'] = True
    
    # # 绘制第一张图: 准确率 vs 轮数
    # if save_epoch_plot:
    #     plt.figure(figsize=fig_size)
        
    #     for algo_name, data in accuracy_data.items():
    #         param = algo_name.split('-')[-1]
    #         param_name = param[:-1]
    #         # if param_name == 'epsilon':
    #         param_name = r'$\epsilon$'
    #         # elif param_name == 'lambda':
    #         #     param_name = r'$\lambda$'
    #         param_value = param[-1]
    #         plt.plot(data['epoch'][:total_rounds], data['accuracy'][:total_rounds], 
    #                 linewidth=2, marker='o', markersize=2, label=f"{param_name}={param_value}")
        
    #     # plt.title(f'{dataset_name} (Dir={dir_alpha}): 准确率 vs 轮数', fontsize=14)
    #     plt.xlabel('全局轮数', fontsize=20)
    #     plt.ylabel('准确率（%）', fontsize=20)
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend(fontsize=20)
        
    #     # 设置刻度字体为Times New Roman
    #     plt.xticks(fontproperties='Times New Roman',fontsize=18)
    #     plt.yticks(fontproperties='Times New Roman',fontsize=18)
        
    #     plt.tight_layout()
    #     file_name_head = main_dir.split("/")[-1]
    #     file_name_tail = str(total_rounds)
    #     epoch_plot_path = os.path.join(main_dir, f'{file_name_head}_accuracy_vs_epoch_{file_name_tail}.png')
    #     plt.savefig(epoch_plot_path, dpi=dpi)
    #     results["figures"]["epoch_plot"] = epoch_plot_path
    #     print(f"保存图表: {epoch_plot_path}")
    
    # 绘制第二张图: 准确率 vs 累计时间
    if save_time_plot:
        plt.figure(figsize=fig_size)
        
        for algo_name, data in accuracy_data.items():
            #algo_name = algo_name.split('-')[0]

            param = algo_name.split('-')[-1]
            param_name = param[:-1]
            # if param_name == 'epsilon':
            param_name = r'$\epsilon$'
            # elif param_name == 'lambda':
            #     param_name = r'$\lambda$'
            param_value = param[-1]
            # 使用NumPy布尔索引
            mask = data['cum_time'] <= max_time
            plt.plot(data['cum_time'][mask], data['accuracy'][mask], 
                    linewidth=2, marker='o', markersize=2, label=f"{param_name}={param_value}")
        
        # plt.title(f'{dataset_name} (Dir={dir_alpha}): 准确率 vs 累计时间', fontsize=14)
        plt.xlabel('挂钟时间（分钟）', fontsize=20)
        plt.ylabel('准确率（%）', fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=20)
        
        # 设置刻度字体为Times New Roman
        plt.xticks(fontproperties='Times New Roman',fontsize=18)
        plt.yticks(fontproperties='Times New Roman',fontsize=18)
        
        plt.tight_layout()

        file_name_head = main_dir.split("/")[-1]
        file_name_tail = str(total_rounds)
        time_plot_path = os.path.join(main_dir, f'{file_name_head}_accuracy_vs_time_{file_name_tail}.png')
        plt.savefig(time_plot_path, dpi=dpi)
        results["figures"]["time_plot"] = time_plot_path
        print(f"保存图表: {time_plot_path}")
    
    print("绘图完成！")
    return results


# def plot_federated_learning_performance(
#     main_dir, 
#     max_time=50*60,  # 默认50分钟 
#     total_rounds=500,  # 默认500轮
#     save_epoch_plot=True,
#     save_time_plot=True,
#     fig_size=(12, 8),
#     dpi=300,
#     dataset_name="Fashion-MNIST",
#     dir_alpha="0.5"
# ):
#     """
#     从各个联邦学习算法目录中读取性能数据并绘制对比图
    
#     参数:
#         main_dir (str): 主实验目录路径
#         max_time (int): 时间图的最大时间限制（秒）
#         total_rounds (int): 轮数图的最大轮数限制
#         save_epoch_plot (bool): 是否保存轮数图
#         save_time_plot (bool): 是否保存时间图
#         fig_size (tuple): 图表大小
#         dpi (int): 图表DPI
#         dataset_name (str): 数据集名称
#         dir_alpha (str): Dir参数值
        
#     返回:
#         dict: 包含成功加载的算法数据和图表路径
#     """
#     # 设置字体
#     config = {
#         "font.family": 'serif',
#         "font.size": 12,
#         "mathtext.fontset": 'stix',
#         "font.serif": ['SimSun'],
#     }
#     rcParams.update(config)
#     plt.rcParams['axes.unicode_minus'] = False
    
#     # 用于存储各算法数据的字典
#     accuracy_data = {}
#     results = {
#         "loaded_algorithms": [],
#         "skipped_algorithms": [],
#         "figures": {}
#     }

#     # 遍历所有算法目录
#     for algo_dir in os.listdir(main_dir):
#         full_path = os.path.join(main_dir, algo_dir)
        
#         # 确保是目录
#         if os.path.isdir(full_path):
#             csv_path = os.path.join(full_path, 'epoch_results_with_cumtime.csv')
            
#             # 检查CSV文件是否存在
#             if os.path.exists(csv_path):
#                 try:
#                     # 读取CSV文件
#                     df = pd.read_csv(csv_path)
                    
#                     # 确保CSV包含所需列
#                     if 'epoch' in df.columns and 'accuracy' in df.columns and 'epoch_cum_time' in df.columns:
#                         # 存储数据 - 算法名作为键
#                         accuracy_data[algo_dir] = {
#                             'epoch': df['epoch'].values,
#                             'accuracy': df['accuracy'].values,
#                             'cum_time': df['epoch_cum_time'].values
#                         }
#                         results["loaded_algorithms"].append(algo_dir)
#                         print(f"成功加载 {algo_dir} 的数据")
#                     else:
#                         results["skipped_algorithms"].append(f"{algo_dir} (缺少必要的列)")
#                         print(f"警告: {csv_path} 缺少必要的列")
#                 except Exception as e:
#                     results["skipped_algorithms"].append(f"{algo_dir} (读取错误: {str(e)})")
#                     print(f"读取 {csv_path} 时出错: {e}")
#             else:
#                 results["skipped_algorithms"].append(f"{algo_dir} (未找到CSV文件)")
#                 print(f"跳过 {algo_dir}: 未找到 epoch_results_with_cumtime.csv 文件")

#     # 检查是否成功加载了数据
#     if not accuracy_data:
#         print("未找到任何有效数据，请检查目录结构和CSV文件")
#         return results
    
#     # 使用默认样式，启用网格以增强可读性
#     plt.rcParams['axes.grid'] = True
    
#     # 绘制第一张图: 准确率 vs 轮数
#     if save_epoch_plot:
#         plt.figure(figsize=fig_size)
        
#         for algo_name, data in accuracy_data.items():
#             plt.plot(data['epoch'][:total_rounds], data['accuracy'][:total_rounds], 
#                     linewidth=2, marker='o', markersize=4, label=algo_name)
        
#         plt.title(f'{dataset_name} (Dir={dir_alpha}): 准确率 vs 轮数', fontsize=14)
#         plt.xlabel('轮数', fontsize=12)
#         plt.ylabel('准确率', fontsize=12)
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.legend(fontsize=10)
        
#         # 设置刻度字体为Times New Roman
#         plt.xticks(fontproperties='Times New Roman')
#         plt.yticks(fontproperties='Times New Roman')
        
#         plt.tight_layout()
        
#         epoch_plot_path = os.path.join(main_dir, 'accuracy_vs_epoch.png')
#         plt.savefig(epoch_plot_path, dpi=dpi)
#         results["figures"]["epoch_plot"] = epoch_plot_path
#         print(f"保存图表: {epoch_plot_path}")
    
#     # 绘制第二张图: 准确率 vs 累计时间
#     if save_time_plot:
#         plt.figure(figsize=fig_size)
        
#         for algo_name, data in accuracy_data.items():
#             # 使用NumPy布尔索引
#             mask = data['cum_time'] <= max_time
#             plt.plot(data['cum_time'][mask], data['accuracy'][mask], 
#                     linewidth=2, marker='o', markersize=4, label=algo_name)
        
#         plt.title(f'{dataset_name} (Dir={dir_alpha}): 准确率 vs 累计时间', fontsize=14)
#         plt.xlabel('累计时间（秒）', fontsize=12)
#         plt.ylabel('准确率', fontsize=12)
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.legend(fontsize=10)
        
#         # 设置刻度字体为Times New Roman
#         plt.xticks(fontproperties='Times New Roman')
#         plt.yticks(fontproperties='Times New Roman')
        
#         plt.tight_layout()
        
#         time_plot_path = os.path.join(main_dir, 'accuracy_vs_time.png')
#         plt.savefig(time_plot_path, dpi=dpi)
#         results["figures"]["time_plot"] = time_plot_path
#         print(f"保存图表: {time_plot_path}")
    
#     print("绘图完成！")
#     return results

def parse_log_file(path):
    # Initialize lists to store the parsed data
    epochs = []
    selected_clients = []
    accuracies = []
    losses = []
    run_times = []
    log_file_path = f'{path}/output.log'  # Path to the log file
    output_csv_path = f'{path}/epoch_results.csv'  # Path to save the CSV file
    # Regular expressions to match the relevant lines
    epoch_pattern = re.compile(r'\| current_epoch (\d+) \|')
    client_select_pattern = re.compile(r'SchedulerThread (?:schedule (?:Group \$\$ \d+ \$\$|Group \[\s*\d+\s*\]|Tier \[\s*\d+\s*\]), )?select[\$\(]\s*\d+\s*clients[\$\)]:')
    client_id_pattern = re.compile(r'(\d+) \|')
    epoch_result_pattern = re.compile(r'Epoch\(t\): (\d+) accuracy: ([\d.]+) loss ([\d.]+) run_time: ([\d.]+)')

    # Read the log file
    with open(log_file_path, 'r') as file:
        current_epoch = None
        clients = []
        for line in file:
            # Match the current epoch line
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            # Match the client selection line
            client_select_match = client_select_pattern.search(line)
            if client_select_match:
                # Reset clients list for the new epoch
                clients = []
                # Read the next line to extract client IDs
                next_line = next(file, '').strip()
                client_ids = client_id_pattern.findall(next_line)
                clients = [int(client_id) for client_id in client_ids]

            # Match the epoch result line
            epoch_result_match = epoch_result_pattern.search(line)
            if epoch_result_match:
                epoch = int(epoch_result_match.group(1))
                accuracy = float(epoch_result_match.group(2))
                loss = float(epoch_result_match.group(3))
                run_time = float(epoch_result_match.group(4))

                # Ensure the epoch and clients are matched correctly
                if current_epoch == epoch:
                    epochs.append(current_epoch)
                    selected_clients.append(clients)
                    accuracies.append(accuracy)
                    losses.append(loss)
                    run_times.append(run_time)
                else:
                    print(f"Warning: Mismatch between epoch {current_epoch} and result epoch {epoch}")

    # Create a DataFrame from the parsed data
    data = {
        'epoch': epochs,
        'selected_clients': selected_clients,
        'accuracy': accuracies,
        'loss': losses,
        'run_time': run_times
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")

import pandas as pd
import ast

def cal_round_time(x, delay_model):
    """
    计算单个轮次的模拟时间。
    :param x: 当前轮次的数据（包含 selected_clients）。
    :param delay_model: 客户端延迟模型（DataFrame）。
    :return: 当前轮次的最大客户端延迟时间。
    """
    client_ids = x['selected_clients']
    # 提取每个客户端的延迟时间
    time_per_client = []
    for client in client_ids:
        try:
            # 查询客户端的延迟时间
            time = delay_model.loc[client, 'time']
            time_per_client.append(time)
        except KeyError:
            # 如果客户端不存在于延迟模型中，跳过或记录警告
            print(f"Warning: Client {client} not found in delay model.")
            continue
    # 返回最大延迟时间
    return max(time_per_client) if time_per_client else 0

def cal_sim_round_time(delay_model_path, epoch_result_path):
    """
    计算所有轮次的模拟时间。
    :param delay_model_path: 客户端延迟模型文件路径。
    :param epoch_result_path: 轮次结果文件路径。
    :return: 包含模拟时间和累计时间的轮次结果 DataFrame。
    """
    # 读取延迟模型和轮次结果
    delay_model = pd.read_csv(delay_model_path)
    epoch_result = pd.read_csv(epoch_result_path)


    # 将 selected_clients 列从字符串转换为列表
    epoch_result['selected_clients'] = epoch_result['selected_clients'].apply(ast.literal_eval)

    # 将 delay_model 的 client_id 设置为索引，以提高查询效率
    delay_model = delay_model.set_index('client_id')



    # if algo_name == "FedAGSA":
    #     epoch_result['sim_round_time'] = 
    # else:
        # 计算每个轮次的模拟时间
    epoch_result['sim_round_time'] = epoch_result.apply(lambda x: cal_round_time(x, delay_model), axis=1)

    # 计算累计时间
    epoch_result['epoch_cum_time'] = epoch_result['sim_round_time'].cumsum()

    return epoch_result

def process_and_save_results(delay_model_path, epoch_result_path, output_csv_path):
    """
    封装函数：计算模拟时间和累计时间，并保存结果为 CSV 文件。
    :param delay_model_path: 客户端延迟模型文件路径。
    :param fedavg_epoch_result: 轮次结果文件路径。
    :param output_csv_path: 输出 CSV 文件路径。
    """
    # 计算模拟时间和累计时间
    epoch_result = cal_sim_round_time(delay_model_path, epoch_result_path)

    # 保存结果为 CSV 文件
    epoch_result.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

# # Example usage
# log_file_path = 'output.log'  # Path to the log file
# output_csv_path = 'epoch_results.csv'  # Path to save the CSV file
# parse_log_file(log_file_path, output_csv_path)


import pandas as pd
import numpy as np
import os

def analyze_federated_learning_csv(csv_path, accuracy_thresholds, total_rounds):
    """
    分析联邦学习CSV文件以提取性能指标。
    
    参数:
        csv_path (str): 包含轮次结果的CSV文件路径
        accuracy_thresholds (list): 需要查找的准确率阈值列表 (例如, [75, 80, 85])
        total_rounds (int): 要考虑的总轮次数
        
    返回:
        dict: 包含性能指标的字典:
            - epoch_at_threshold: 阈值到首次达到该阈值的轮次的映射
            - time_at_threshold: 阈值到首次达到该阈值的时间的映射（分钟）
            - total_time: 指定总轮次的时间（分钟）
            - best_accuracy: 达到的最佳准确率
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 初始化结果
    results = {
        'epoch_at_threshold': {},
        'time_at_threshold': {},
        'total_time': None,
        'best_accuracy': None
    }
    
    # 限制数据范围到total_rounds
    df = df[df['epoch'] <= total_rounds]
    
    # 找到达到的最佳准确率（仅考虑total_rounds内的数据）
    if not df.empty:
        results['best_accuracy'] = df['accuracy'].max()
    else:
        results['best_accuracy'] = 0
    
    # 找到首次达到每个准确率阈值的时间
    for threshold in accuracy_thresholds:
        threshold_rows = df[df['accuracy'] >= threshold]
        
        if not threshold_rows.empty:
            first_row = threshold_rows.iloc[0]
            results['epoch_at_threshold'][threshold] = int(first_row['epoch'])
            # 将秒转换为分钟
            results['time_at_threshold'][threshold] = float(first_row['epoch_cum_time']) / 60.0
        else:
            # 如果未达到阈值，则设置为None
            results['epoch_at_threshold'][threshold] = None
            results['time_at_threshold'][threshold] = None
    
    # 获取指定总轮次的时间（转换为分钟）
    if total_rounds <= df['epoch'].max():
        total_rounds_row = df[df['epoch'] == total_rounds]
        if not total_rounds_row.empty:
            results['total_time'] = float(total_rounds_row.iloc[0]['epoch_cum_time']) / 60.0
        else:
            # 如果找不到确切的轮次但在范围内，使用最后一个可用的轮次
            results['total_time'] = float(df['epoch_cum_time'].iloc[-1]) / 60.0
    else:
        # 如果total_rounds超出数据范围，使用最后一个可用的轮次
        results['total_time'] = float(df['epoch_cum_time'].iloc[-1]) / 60.0
    
    return results

def format_as_row(method, distribution, results, accuracy_thresholds):
    """
    将结果格式化为表格的单行数据。
    
    参数:
        method (str): 方法名称 (例如 "FedAvg")
        distribution (str): 分布类型 (例如 "IID" 或 "nonIID(0.1)")
        results (dict): analyze_federated_learning_csv的结果
        accuracy_thresholds (list): 准确率阈值列表
        
    返回:
        list: 格式化的数据行，类似图片中的表格
    """
    # 确保分布信息正确显示
    if distribution.lower() == "iid":
        row = ["IID", method]
    else:
        # 确保nonIID格式正确
        if not distribution.lower().startswith("noniid"):
            distribution = f"nonIID({distribution})"
        row = [distribution, method]
    
    # 为每个阈值添加轮次和时间
    for threshold in accuracy_thresholds:
        epoch = results['epoch_at_threshold'].get(threshold)
        time = results['time_at_threshold'].get(threshold)
        
        if epoch is not None and time is not None:
            row.append(epoch)
            row.append(round(time, 2))  # 时间已在analyze_federated_learning_csv中转换为分钟
        else:
            # 如果未达到准确率，则设置为0而不是"-"
            row.append(0)
            row.append(0)
    
    # 添加总时间和最佳准确率
    row.append(round(results['total_time'], 2))  # 时间已在analyze_federated_learning_csv中转换为分钟
    row.append(round(results['best_accuracy'], 2))
    
    return row

def analyze_multiple_algorithms(csv_paths_dict, accuracy_thresholds, total_rounds, output_excel_path):
    """
    分析多个算法在不同数据分布上的性能，并生成Excel文件。
    
    参数:
        csv_paths_dict (dict): 字典结构为 {分布: {算法: csv路径, ...}, ...}
        accuracy_thresholds (list): 准确率阈值列表 (例如 [75, 80, 85])
        total_rounds (int): 要考虑的总轮次数
        output_excel_path (str): 输出Excel文件的路径
        
    返回:
        DataFrame: 包含所有结果的DataFrame
    """
    # 创建标题
    header = ["数据集", "方法"]
    for threshold in accuracy_thresholds:
        header.extend([f"{threshold}%轮次", f"{threshold}%时间(分钟)"])
    header.extend(["总时间(分钟)", "最佳准确率"])
    
    # 收集所有行
    all_rows = []
    
    # 遍历每个分布和算法
    for distribution, algorithms in csv_paths_dict.items():
        for method, csv_path in algorithms.items():
            if not os.path.exists(csv_path):
                print(f"警告: CSV文件 {csv_path} 未找到。")
                continue
            
            # 分析CSV
            results = analyze_federated_learning_csv(csv_path, accuracy_thresholds, total_rounds)
            
            # 格式化为行
            row = format_as_row(method, distribution, results, accuracy_thresholds)
            all_rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(all_rows, columns=header)
    
    # # 保存到Excel
    # df.to_excel(output_excel_path, index=False)
    
    print(f"结果已保存到 {output_excel_path}")
    
    return df



import re
import pandas as pd
import os
import numpy as np
from ast import literal_eval

def parse_fed_log(log_file_path):
    """
    解析联邦学习日志文件，提取每个epoch的信息
    
    参数:
        log_file_path (str): 日志文件的路径
        
    返回:
        pandas.DataFrame: 包含解析结果的DataFrame
    """
    # 读取日志文件内容
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    
    # 提取每个epoch的信息
    results = []
    
    # 使用正则表达式匹配所有的group选择信息
    group_selections = {}
    for group_id in range(5):  # 假设有5个group (0-4)
        pattern = rf'group {group_id} selected \d+ clients: \[([\d, ]+)\]'
        selections = re.findall(pattern, log_content)
        # 将每个选择转换为客户端id列表
        selections = [[int(client_id) for client_id in selection.split(', ')] for selection in selections]
        group_selections[group_id] = selections
    
    # 提取每个epoch的聚合信息
    epoch_pattern = r'group_ready_num: (\d+) .+?\n-+\nEpoch (\d+) tested, accuracy: ([\d\.]+) loss ([\d\.]+) run_time ([\d\.]+)'
    epoch_matches = re.findall(epoch_pattern, log_content, re.DOTALL)
    
    # 构建结果列表
    for match in epoch_matches:
        group_id = int(match[0])
        epoch = int(match[1])
        accuracy = float(match[2])
        loss = float(match[3])
        run_time = float(match[4])
        
        # 找到该epoch对应的客户端选择
        # 假设每个group的选择顺序与epoch的聚合顺序相对应
        # 例如，group 0的第一次选择对应第一次聚合group 0的epoch
        group_epoch_count = sum(1 for m in epoch_matches[:epoch_matches.index(match)] if int(m[0]) == group_id)
        
        # 确保我们有足够的选择记录
        if group_epoch_count < len(group_selections[group_id]):
            selected_clients = group_selections[group_id][group_epoch_count]
        else:
            selected_clients = []  # 如果没有找到对应的选择记录
        
        results.append({
            'Epoch': epoch,
            'Group_ID': group_id,
            'Selected_Clients': selected_clients,
            'Accuracy': accuracy,
            'Loss': loss,
            'Run_Time': run_time
        })
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    return df

def calculate_simulation_time(analysis_df, delay_df, constant_t=5.0):
    """
    计算每个epoch的仿真时间
    
    参数:
        analysis_df (pandas.DataFrame): 解析的epoch信息DataFrame
        delay_df (pandas.DataFrame): 客户端延迟信息DataFrame
        constant_t (float): 常数t值，默认为5.0
        
    返回:
        pandas.DataFrame: 添加了仿真时间的DataFrame
    """
    # 创建客户端ID到系统时间的映射
    client_delay_map = dict(zip(delay_df['client_id'], delay_df['system_time']))
    
    # 创建一个新的DataFrame，以保留原始DataFrame不变
    result_df = analysis_df.copy()
    
    # 添加新列用于存储仿真时间
    result_df['Simulation_Time'] = 0.0
    result_df['edge_round_time'] = 0.0
    
    # 为每个group维护最近一次出现的epoch的仿真时间
    last_sim_time = {group_id: 0.0 for group_id in range(5)}  # 假设有5个group
    
    # 计算每个epoch的仿真时间
    for index, row in result_df.iterrows():
        group_id = row['Group_ID']
        
        # 获取当前epoch的客户端列表
        if isinstance(row['Selected_Clients'], str):
            try:
                # 尝试将字符串转换为列表
                selected_clients = literal_eval(row['Selected_Clients'])
            except (ValueError, SyntaxError):
                # 如果字符串格式不正确，尝试其他方法
                selected_clients = [int(c.strip()) for c in row['Selected_Clients'].strip('[]').split(',') if c.strip()]
        else:
            selected_clients = row['Selected_Clients']
        
        # 获取这些客户端的最大系统时间
        max_system_time = 0.0
        for client_id in selected_clients:
            system_time = client_delay_map.get(client_id, 0.0)
            max_system_time = max(max_system_time, system_time)
        
        # 计算当前epoch的仿真时间
        b = last_sim_time[group_id]  # 上一次该group的仿真时间
        simulation_time = b + max_system_time + constant_t
        
        # 计算当前epoch的边缘轮次时间
        # 更新结果
        result_df.at[index, 'Simulation_Time'] = simulation_time
        result_df.at[index, 'edge_round_time'] = max_system_time
        
        # 更新最近一次出现的仿真时间
        last_sim_time[group_id] = simulation_time
    
    return result_df

def analyze_fed_log_with_simulation_time(log_file_path, delay_file_path, output_path=None, constant_t=5.0):
    """
    分析联邦学习日志文件并计算仿真时间，保存结果
    
    参数:
        log_file_path (str): 日志文件的路径
        delay_file_path (str): 延迟文件的路径
        output_path (str, 可选): 输出CSV文件的路径，如果不提供则使用默认路径
        constant_t (float): 常数t值，默认为5.0
        
    返回:
        pandas.DataFrame: 包含分析结果的DataFrame
    """
    # 解析日志文件
    analysis_df = parse_fed_log(log_file_path)
    
    # 读取延迟文件
    delay_df = pd.read_csv(delay_file_path)
    
    # 计算仿真时间
    results_df = calculate_simulation_time(analysis_df, delay_df, constant_t)
    
    # 如果未提供输出路径，则创建默认路径
    if output_path is None:
        dir_name = os.path.dirname(log_file_path)
        base_name = os.path.basename(log_file_path).split('.')[0]
        output_path = os.path.join(dir_name, f"{base_name}_with_simulation_time.csv")
    
    # 保存结果
    # 将客户端列表转换为字符串，以便在CSV中正确显示
    if not isinstance(results_df['Selected_Clients'].iloc[0], str):
        results_df['Selected_Clients'] = results_df['Selected_Clients'].apply(lambda x: str(x))
    
    results_df.to_csv(output_path, index=False)
    
    print(f"分析结果已保存到: {output_path}")
    
    return results_df

