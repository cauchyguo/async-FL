import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Dict, Tuple
import re

class FLPlotter:
    def __init__(self, base_dir: str):
        """
        Initialize the FLPlotter with base directory
        Args:
            base_dir: Base directory containing algorithm folders
        """
        self.base_dir = base_dir
        self.algorithms = self._find_algorithms()
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                      '#bcbd22', '#17becf']
        self.linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

    def _find_algorithms(self) -> List[str]:
        """
        Find all algorithm directories in base directory
        Returns:
            List of algorithm directory names
        """
        algorithms = []
        for item in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, item)):
                algorithms.append(item)
        
        # Sort algorithms by name
        algorithms.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])
        return algorithms

    def _load_data(self, file_path: str) -> np.ndarray:
        """
        Load data from a text file
        Args:
            file_path: Path to the data file
        Returns:
            Numpy array containing the data
        """
        with open(file_path, 'r') as f:
            data = f.read().strip()
            if data.startswith('[') and data.endswith(']'):
                data = data[1:-1]
            return np.array([float(x) for x in data.split(',')])

    def _plot_metric(self, ax: plt.Axes, data: np.ndarray, title: str, 
                    ylabel: str, color: str, linestyle: str) -> None:
        """
        Plot a single metric on the given axes
        Args:
            ax: Matplotlib axes object
            data: Data to plot
            title: Plot title
            ylabel: Y-axis label
            color: Line color
            linestyle: Line style
        """
        ax.plot(data, color=color, linestyle=linestyle, linewidth=2)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel('Iterations', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def plot_performance(self, output_file: str = 'performance_comparison.png', 
                        dpi: int = 300) -> None:
        """
        Plot performance metrics for all algorithms
        Args:
            output_file: Output file name
            dpi: DPI for the output image
        """
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle('Federated Learning Performance Comparison', fontsize=14, y=0.95)

        # Plot each algorithm's metrics
        for i, algo in enumerate(self.algorithms):
            try:
                # Load data
                accuracy = self._load_data(os.path.join(self.base_dir, algo, 'accuracy.txt'))
                loss = self._load_data(os.path.join(self.base_dir, algo, 'loss.txt'))
                update_time = self._load_data(os.path.join(self.base_dir, algo, 'update_time.txt'))

                # Plot metrics
                self._plot_metric(ax1, accuracy, 'Model Accuracy', 'Accuracy', 
                                self.colors[i], self.linestyles[i])
                self._plot_metric(ax2, loss, 'Training Loss', 'Loss', 
                                self.colors[i], self.linestyles[i])
                self._plot_metric(ax3, update_time, 'Update Time', 'Time (s)', 
                                self.colors[i], self.linestyles[i])

            except FileNotFoundError:
                print(f"Warning: Data not found for {algo}")

        # Add legend
        lines = [plt.Line2D([0], [0], color=self.colors[i], linestyle=self.linestyles[i]) 
                for i in range(len(self.algorithms))]
        fig.legend(lines, self.algorithms, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.92), ncol=4)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        # Save figure
        plt.savefig(os.path.join(self.base_dir, output_file), 
                   dpi=dpi, bbox_inches='tight')
        plt.close()

def main():
    # Initialize plotter with base directory
    plotter = FLPlotter('PK/Exp2025-V2-FashionMNIST-dir0.1')
    
    # Plot performance metrics
    plotter.plot_performance()

if __name__ == '__main__':
    main() 