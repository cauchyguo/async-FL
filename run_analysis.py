import os
from analyze_client_selection import analyze_client_selection

def main():
    # 主实验目录路径
    main_dir = "/home/ypguo/async-FL/PK/Exp2025-FashionMNIST-Dir0.1"
    
    # 客户端时间信息文件路径
    time_info_file = "/home/ypguo/async-FL/src/clients_info/clients_info_dir0.1_cnn_delaymodel.csv"
    
    # 自定义时间区间（可以根据需要修改）
    # 如果不指定，将自动根据数据生成区间
    time_bins = [0, 3, 6, 9, 12]
    
    # 限制分析的最大epoch数（可选）
    max_epoch = 300
    
    # 执行分析
    results = analyze_client_selection(
        main_dir=main_dir,
        time_info_file=time_info_file,
        time_bins=time_bins,
        max_epoch=max_epoch,
        save_plots=True
    )
    
    # 输出每个算法的统计信息
    print("\n==== 各算法客户端选择统计 ====")
    for algo_name, result in results.items():
        print(f"\n算法: {algo_name}")
        print("时间区间统计:")
        print(result['bin_stats'])
        
        # 输出额外统计信息
        total_selections = sum(result['client_selections'].values())
        unique_clients = len(result['client_selections'])
        print(f"被选中的独立客户端数: {unique_clients}")
        print(f"客户端总选择次数: {total_selections}")
        print(f"平均每个客户端被选择次数: {total_selections/unique_clients:.2f}")

if __name__ == "__main__":
    main() 