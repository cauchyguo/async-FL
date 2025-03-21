import pandas as pd
import numpy as np
import os

def analyze_federated_learning_csv(csv_path, accuracy_thresholds, total_rounds):
    """
    分析联邦学习CSV文件，提取性能指标
    
    参数:
        csv_path (str): 包含训练轮次结果的CSV文件路径
        accuracy_thresholds (list): 需要查找的准确率阈值列表（例如：[75, 80, 85]）
        total_rounds (int): 需要考虑的总轮次数
        
    返回:
        dict: 包含性能指标的字典:
            - epoch_at_threshold: 映射阈值到首次达到该阈值的轮次
            - time_at_threshold: 映射阈值到首次达到该阈值的时间
            - total_time: 指定总轮次的累计时间
            - best_accuracy: 达到的最佳准确率
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 初始化结果字典
    results = {
        'epoch_at_threshold': {},
        'time_at_threshold': {},
        'total_time': None,
        'best_accuracy': None
    }
    
    # 找出达到的最佳准确率
    results['best_accuracy'] = df['accuracy'].max()
    
    # 找出首次达到每个准确率阈值的轮次
    for threshold in accuracy_thresholds:
        threshold_rows = df[df['accuracy'] >= threshold]
        
        if not threshold_rows.empty:
            first_row = threshold_rows.iloc[0]
            results['epoch_at_threshold'][threshold] = int(first_row['epoch'])
            results['time_at_threshold'][threshold] = float(first_row['epoch_cum_time'])
        else:
            results['epoch_at_threshold'][threshold] = None
            results['time_at_threshold'][threshold] = None
    
    # 获取指定总轮次的时间
    if total_rounds <= df['epoch'].max():
        total_rounds_row = df[df['epoch'] == total_rounds]
        if not total_rounds_row.empty:
            results['total_time'] = float(total_rounds_row.iloc[0]['epoch_cum_time'])
        else:
            # 如果没有找到确切的轮次但在范围内，使用最后一个可用轮次
            results['total_time'] = float(df['epoch_cum_time'].iloc[-1])
    else:
        # 如果总轮次超过数据范围，使用最后一个可用轮次
        results['total_time'] = float(df['epoch_cum_time'].iloc[-1])
    
    return results

def format_as_row(method, distribution, results, accuracy_thresholds):
    """
    将结果格式化为表格的一行数据
    
    参数:
        method (str): 方法名称（例如："FedAvg"）
        distribution (str): 分布类型（例如："IID"）
        results (dict): analyze_federated_learning_csv函数的结果
        accuracy_thresholds (list): 准确率阈值列表
        
    返回:
        list: 格式化后的数据行，类似于图像中的表格
    """
    row = [distribution, method]
    
    # 添加每个阈值的轮次和时间
    for threshold in accuracy_thresholds:
        epoch = results['epoch_at_threshold'].get(threshold)
        time = results['time_at_threshold'].get(threshold)
        
        if epoch is not None and time is not None:
            row.append(epoch)
            row.append(round(time, 2))
        else:
            row.append('-')
            row.append('-')
    
    # 添加总时间和最佳准确率
    row.append(round(results['total_time'], 2))
    row.append(round(results['best_accuracy'], 2))
    
    return row

def analyze_multiple_experiments(csv_paths_dict, accuracy_thresholds, total_rounds, output_excel):
    """
    分析多个算法的实验结果并生成Excel文件
    
    参数:
        csv_paths_dict (dict): 包含算法信息的字典，格式为：
                              {
                                  'distribution1': {
                                      'method1': 'path/to/csv1',
                                      'method2': 'path/to/csv2',
                                      ...
                                  },
                                  'distribution2': {
                                      ...
                                  }
                              }
        accuracy_thresholds (list): 准确率阈值列表
        total_rounds (int): 需要考虑的总轮次数
        output_excel (str): 输出Excel文件的路径
    """
    # 创建结果列表
    all_rows = []
    
    # 创建表头
    header = ["数据集", "方法"]
    for threshold in accuracy_thresholds:
        header.extend([f"{threshold}%轮数", f"{threshold}%时间"])
    header.extend(["总时间", "最佳准确率"])
    
    # 分析每个分布和方法
    for distribution, methods_dict in csv_paths_dict.items():
        for method, csv_path in methods_dict.items():
            if os.path.exists(csv_path):
                # 分析CSV
                results = analyze_federated_learning_csv(csv_path, accuracy_thresholds, total_rounds)
                
                # 格式化为一行
                row = format_as_row(method, distribution, results, accuracy_thresholds)
                all_rows.append(row)
            else:
                print(f"警告：找不到CSV文件 {csv_path}")
    
    # 创建DataFrame
    df_results = pd.DataFrame(all_rows, columns=header)
    
    # 保存到Excel
    df_results.to_excel(output_excel, index=False)
    print(f"结果已保存到: {output_excel}")

# 使用示例
if __name__ == "__main__":
    # 准确率阈值列表
    accuracy_thresholds = [75, 80, 85]
    # 总轮次
    total_rounds = 500
    # 输出Excel文件路径
    output_excel = "federated_learning_results.xlsx"
    
    # 定义要分析的CSV文件路径
    csv_paths_dict = {
        "IID": {
            "FedAvg": "src/results/Exp2025-FashionMNIST-IID/FedAvg-CNN-IID/epoch_results_with_cumtime.csv",
            "FedProx": "src/results/Exp2025-FashionMNIST-IID/FedProx-CNN-IID/epoch_results_with_cumtime.csv",
            "FedRE": "src/results/Exp2025-FashionMNIST-IID/FedRE-CNN-IID/epoch_results_with_cumtime.csv"
        },
        "nonIID(0.1)": {
            "FedAvg": "src/results/Exp2025-FashionMNIST-Dir0.1/FedAvg-CNN-dir0.1/epoch_results_with_cumtime.csv",
            "FedProx": "src/results/Exp2025-FashionMNIST-Dir0.1/FedProx-CNN-dir0.1/epoch_results_with_cumtime.csv",
            "FedRE": "src/results/Exp2025-FashionMNIST-Dir0.1/FedRE-CNN-dir0.1/epoch_results_with_cumtime.csv"
        },
        "nonIID(0.5)": {
            "FedAvg": "src/results/Exp2025-FashionMNIST-Dir0.5/FedAvg-CNN-dir0.5/epoch_results_with_cumtime.csv",
            "FedProx": "src/results/Exp2025-FashionMNIST-Dir0.5/FedProx-CNN-dir0.5/epoch_results_with_cumtime.csv",
            "FedRE": "src/results/Exp2025-FashionMNIST-Dir0.5/FedRE-CNN-dir0.5/epoch_results_with_cumtime.csv"
        }
    }
    
    # 分析多个实验并生成Excel文件
    analyze_multiple_experiments(csv_paths_dict, accuracy_thresholds, total_rounds, output_excel) 