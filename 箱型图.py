import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import Union

def generate_time_comparison_plot(
    results_dir: Union[str, Path],
    output_path: Union[str, Path],
    max_epoch: int = 500,
    figsize: tuple = (10, 6),
    dpi: int = 300
) -> None:
    """
    生成联邦学习算法耗时对比箱型图
    
    :param results_dir: 包含算法结果的根目录路径
    :param output_path: 图片输出路径（需包含文件名和扩展名）
    :param max_epoch: 最大考虑的训练轮次（默认500）
    :param figsize: 图表尺寸（默认(10,6)）
    :param dpi: 图片分辨率（默认300）
    """
    # 转换为Path对象
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    
    # 算法目录映射
    algorithms = {
        "FedAvg": "fedavg-cnn-dir0.1",
        "FedDocs": "FedDocs-CNN-dir0.1",
        "FedLC": "fedlc-cnn-dir0.1",
        "PyramidFL": "PyramidFL-cnn-dir0.1"
    }

    # 读取数据
    all_data = []
    for algo, path in algorithms.items():
        try:
            df = pd.read_csv(results_dir / path / "epoch_results_with_cumtime.csv")
            df = df.query(f"epoch <= {max_epoch}").assign(algorithm=algo)
            all_data.append(df[['epoch', 'sim_round_time', 'algorithm']])
        except Exception as e:
            print(f"【警告】跳过 {algo}: {str(e)}")
            continue

    if not all_data:
        print("错误：没有成功加载任何数据")
        return

    # 合并数据
    combined = pd.concat(all_data, ignore_index=True)

    # 初始化画布
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    # 绘制箱型图
    ax = sns.boxplot(
        x='algorithm',
        y='sim_round_time',
        data=combined,
        width=0.6,
        showfliers=False
    )

    # 图表装饰
    ax.set_title(f'算法耗时对比 (1-{max_epoch} Epochs)', fontsize=14, pad=15)
    ax.set_xlabel('算法名称', fontsize=12, labelpad=10)
    ax.set_ylabel('单轮耗时（秒）', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.grid(axis='y', alpha=0.6)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"图表已保存至: {output_path.resolve()}")
    plt.close()


# 使用示例
if __name__ == "__main__":
    # 设置中文显示（Windows）
    plt.rcParams['font.sans-serif'] = ['simsun']
    plt.rcParams['axes.unicode_minus'] = False
    
    generate_time_comparison_plot(
        results_dir="/home/ypguo/async-FL/src/results/Exp2025-FashionMNIST-Dir0.1",
        output_path="./sim_time_FashionMNIST-Dir0.1_comparison.png",
        max_epoch=300,
        figsize=(12, 7)
    )
    # generate_time_comparison_plot(
    #     results_dir="/home/ypguo/async-FL/src/results/Exp2025-FashionMNIST-Dir0.5",
    #     output_path="./sim_time_FashionMNIST-Dir0.5_comparison.png",
    #     max_epoch=300,
    #     figsize=(12, 7)
    # )   