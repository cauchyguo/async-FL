import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# Define file paths
results_dir = "src/results/Exp2025-FashionMNIST-Dir0.1"
algorithms = {
    "FedAvg": "fedavg-cnn-dir0.1",
    "FedDocs": "FedDocs-CNN-dir0.1",
    "FedLC": "fedlc-cnn-dir0.1",
    "PyramidFL": "PyramidFL-cnn-dir0.1"
}

# Set a fixed maximum epoch
max_epoch = 50  # Change this value as needed

# Function to read data
def read_data(algo_name, algo_dir):
    file_path = os.path.join(results_dir, algo_dir, "epoch_results_with_cumtime.csv")
    data = pd.read_csv(file_path)
    
    # Filter data up to the specified maximum epoch
    data = data[data['epoch'] <= max_epoch]
    
    data['algorithm'] = algo_name
    return data[['epoch', 'sim_round_time', 'algorithm']]

# Read data from each algorithm
all_data = []
for algo_name, algo_dir in algorithms.items():
    try:
        algo_data = read_data(algo_name, algo_dir)
        all_data.append(algo_data)
        print(f"Successfully loaded data for {algo_name}")
    except Exception as e:
        print(f"Error reading data for {algo_name}: {e}")

if all_data:
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)

    # Calculate statistics for each algorithm
    stats = combined_data.groupby('algorithm')['sim_round_time'].agg(['mean', 'median', 'std', 'min', 'max'])
    print("\nStatistics for sim_round_time by algorithm:")
    print(stats)
    
    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create boxplot
    ax = sns.boxplot(x='algorithm', y='sim_round_time', data=combined_data)
    
    # Add jitter points for better visualization
    sns.stripplot(x='algorithm', y='sim_round_time', data=combined_data, 
                 size=4, color=".3", alpha=0.3)
    
    # Customize plot
    plt.title(f'Simulation Round Time Comparison (Epochs 1-{max_epoch})', fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Simulation Round Time (seconds)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add grid
    ax.yaxis.grid(True)
    
    # Save the plot
    output_file = f"sim_round_time_boxplot_epoch_{max_epoch}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    
    print(f"\nPlot saved as {output_file}")
else:
    print("No data was successfully loaded.") 