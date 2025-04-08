import numpy as np
import pandas as pd
import cvxpy as cp
import math
import matplotlib.pyplot as plt
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


class DelayAdaptiveOptimizer:
    def __init__(self, config):
        # super().__init__(config)
        self.config = config
        self.client_edge_info = pd.read_csv(config["client_edge_info_path"])
        self.total_bandwidth = config.get("total_bandwidth",50)
        self.data_size = config.get("data_size",42.64)
        self.noise_power = config.get("noise_power",-104)
        self.g_dB_coef = config.get("g_dB_coef",[-128.1,-37.6])
        self.data_size = config.get("data_size",42.64)
        

    def optimize(self,group_id,selected_clients_idxs):
        
        selected_clients = self.client_edge_info[self.client_edge_info["client_id"].isin(selected_clients_idxs)]
        selected_clients = selected_clients.set_index("client_id")
        distances = selected_clients[f"server_{group_id}"]
        powers = selected_clients["power"]
        compute_times = selected_clients["compute_time"]


        if self.config.get("algorithm","gp") == "gp":
            result =  self.optimize_bandwidth_gp(distances, powers, compute_times,self.total_bandwidth)
        else:
            result =  self.optimize_bandwidth_simple(distances, powers, compute_times,self.total_bandwidth)
        
        # 将结果与client_id匹配
        if result["success"]:
            print(f"GP算法优化带宽分配结果：")
        else:
            result =  self.optimize_bandwidth_simple(distances, powers, compute_times,self.total_bandwidth)
            print(f"简单迭代算法优化带宽分配结果：")
        # print("优化带宽分配结果：")
        for i, bw in enumerate(result["bandwidths"]):
            # 使用保存的client_id_list来匹配结果
            print(f"客户端ID {selected_clients_idxs[i]}: {bw:.4f} MHz", end=",")
        
        # 延迟同样可以匹配
        print("\n各客户端延迟:",end="")
        for i, delay in enumerate(result["delays"]):
            print(f"客户端ID {selected_clients_idxs[i]}: {delay:.4f} 秒", end=",")
            
        print(f"\n总带宽使用: {np.sum(result['bandwidths']):.4f} MHz , 最大延迟: {result['max_delay']:.4f} 秒")

        return result["delays"]        

    def calculate_data_rate(self, distance, power, bandwidth, noise_poewr=-104,g_dB_coef=[-128.1,37.6]):
        """计算数据传输率 (Mbps)"""
        p_W = 10 ** (power / 10 - 3)  # dBm → 瓦特
        N0_W = 10 ** (noise_poewr / 10 - 3)  # 噪声功率
        
        # 信道增益
        g_dB = g_dB_coef[0] - g_dB_coef[1] * math.log10(distance / 1000)
        g_linear = 10 ** (g_dB / 10)
        
        # 信噪比和频谱效率
        SNR = (g_linear * p_W) / N0_W
        spectral_efficiency = math.log2(1 + SNR)
        
        # 数据传输率 (Mbps)
        rate = bandwidth * 1e6 * spectral_efficiency / 1e6
        return rate

    def optimize_bandwidth_gp(self, distances_series, powers_series, compute_times_series, total_bandwidth=50, data_size=42.64):
        """
        使用几何规划(GP)方法优化带宽分配
        
        GP适合处理有分数项的优化问题，例如1/x形式
        """
        client_num = len(distances_series)
        distances = distances_series.values
        powers = powers_series.values
        compute_times = compute_times_series.values
        
        # 计算基准传输率 (1MHz带宽下)
        base_rates = np.array([self.calculate_data_rate(distances[i], powers[i], 1.0,self.noise_power,self.g_dB_coef) for i in range(client_num)])
        
        # 定义优化变量
        bandwidths = cp.Variable(client_num, pos=True)  # 带宽变量(MHz)，pos=True确保非负
        max_delay = cp.Variable(pos=True)  # 最大延迟变量
        
        # 定义GP问题约束条件
        constraints = [
            cp.sum(bandwidths) <= total_bandwidth  # 总带宽约束
        ]
        
        # 对每个客户端添加延迟约束
        for i in range(client_num):
            # 传输时间 = 数据大小(MB) * 8 / (基准传输率 * 带宽)
            # 重写成: 传输时间 + 计算时间 <= 最大延迟
            trans_time = data_size * 8 / (base_rates[i] * bandwidths[i])
            total_delay = trans_time + compute_times[i]
            constraints.append(total_delay <= max_delay)
        
        # 目标：最小化最大延迟
        objective = cp.Minimize(max_delay)
        
        # 创建并求解GP问题
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(gp=True)  # 使用GP模式求解
            
            if problem.status == "optimal":
                # 计算各客户端的延迟
                delays = []
                # 先获取优化结果，避免变量名冲突
                bandwidth_values = bandwidths.value  # 注意是.value不是.values
                max_delay_value = max_delay.value    # 同上
                
                for i in range(client_num):
                    tx_time = data_size * 8 / (base_rates[i] * bandwidth_values[i])
                    delays.append(tx_time + compute_times[i])
                
                # 创建带有索引的结果
                client_indices = distances_series.index
                bandwidth_series = pd.Series(bandwidth_values, index=client_indices)
                delays_series = pd.Series(delays, index=client_indices)            
                    
                return {
                    "success": True,
                    "bandwidths": bandwidth_series,  # 返回numpy数组而不是pandas Series
                    "max_delay": max_delay_value,    # 返回标量值
                    "delays": delays_series          # 这个可以是Series
                }
            else:
                return {"success": False, "error": f"求解失败: {problem.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        
    def optimize_bandwidth_simple(self, distances_series, powers_series, compute_times_series, total_bandwidth=20, data_size=42.64):
        """
        使用简单迭代法优化带宽分配（不使用凸优化）
        
        适合在CVXPY无法工作时使用
        """
        client_num = len(distances_series)
        distances = distances_series.values
        powers = powers_series.values
        compute_times = compute_times_series.values
        
        # 计算基准传输率
        base_rates = np.array([self.calculate_data_rate(distances[i], powers[i], 1.0, self.noise_power,self.g_dB_coef) for i in range(client_num)])
        
        # 初始均匀分配带宽
        bandwidths = np.ones(client_num) * (total_bandwidth / client_num)
        
        # 计算初始延迟
        delays = []
        for i in range(client_num):
            tx_time = data_size * 8 / (base_rates[i] * bandwidths[i])
            delays.append(tx_time + compute_times[i])
        
        # 迭代优化（100次）
        for _ in range(100):
            # 找出延迟最大的客户端
            max_delay_idx = np.argmax(delays)
            
            # 从其他客户端中找出延迟最小的
            other_indices = list(range(client_num))
            other_indices.remove(max_delay_idx)
            min_delay_idx = other_indices[np.argmin([delays[i] for i in other_indices])]
            
            # 如果最小延迟客户端带宽已经很小，停止迭代
            if bandwidths[min_delay_idx] < 0.5:
                break
                
            # 转移一小部分带宽（0.1MHz）
            transfer = min(0.1, bandwidths[min_delay_idx] * 0.1)
            bandwidths[min_delay_idx] -= transfer
            bandwidths[max_delay_idx] += transfer
            
            # 重新计算延迟
            for i in range(client_num):
                tx_time = data_size * 8 / (base_rates[i] * bandwidths[i])
                delays[i] = tx_time + compute_times[i]


            # 创建带有索引的结果
            client_indices = distances_series.index
            bandwidth_series = pd.Series(bandwidths, index=client_indices)
            delays_series = pd.Series(delays, index=client_indices)   

        return {
            "success": True,
            "bandwidths": bandwidth_series,
            "max_delay": max(delays),
            "delays": delays_series
        }
    
def optimize_bandwidth_binary(self, distances_series, powers_series, compute_times_series, total_bandwidth=20, data_size=42.64):
    """
    使用二分查找方法优化带宽分配
    
    思路：
    1. 设定目标延迟T
    2. 对每个客户端，计算达到延迟T所需的最小带宽
    3. 如果总带宽超过限制，增加目标延迟；否则减少目标延迟
    4. 使用二分查找找到满足总带宽限制的最小目标延迟
    """
    client_num = len(distances_series)
    if client_num == 0:
        return {"success": False, "error": "No clients"}
        
    distances = distances_series.values
    powers = powers_series.values
    compute_times = compute_times_series.values
    
    # 计算基准传输率
    base_rates = np.array([self.calculate_data_rate(d, p, 1.0, self.noise_power, self.g_dB_coef) 
                          for d, p in zip(distances, powers)])
    
    def calculate_required_bandwidth(target_delay):
        """计算达到目标延迟所需的带宽"""
        # 传输时间 = 目标延迟 - 计算时间
        trans_times = np.maximum(target_delay - compute_times, 0)
        # 所需带宽 = 数据大小 * 8 / (基准传输率 * 传输时间)
        required_bw = np.where(
            trans_times > 0,
            data_size * 8 / (base_rates * trans_times),
            float('inf')
        )
        return required_bw
    
    def is_feasible(target_delay):
        """检查目标延迟是否可行"""
        required_bw = calculate_required_bandwidth(target_delay)
        return np.sum(required_bw) <= total_bandwidth and np.all(np.isfinite(required_bw))
    
    # 二分查找合适的目标延迟
    # 下界：最小计算时间
    delay_min = np.max(compute_times)
    # 上界：使用最小带宽时的延迟
    min_bandwidth = 0.1  # 最小分配带宽
    delay_max = np.max(compute_times + data_size * 8 / (base_rates * min_bandwidth))
    
    # 二分查找
    for _ in range(50):  # 限制迭代次数
        target_delay = (delay_min + delay_max) / 2
        if is_feasible(target_delay):
            delay_max = target_delay
        else:
            delay_min = target_delay
            
        if delay_max - delay_min < 1e-4:  # 收敛条件
            break
    
    # 使用最终的目标延迟计算带宽分配
    final_delay = delay_max
    bandwidths = calculate_required_bandwidth(final_delay)
    
    # 处理可能的数值误差，确保总带宽不超过限制
    if np.sum(bandwidths) > total_bandwidth:
        scale = total_bandwidth / np.sum(bandwidths)
        bandwidths *= scale
    
    # 计算实际延迟
    tx_times = data_size * 8 / (base_rates * bandwidths)
    delays = tx_times + compute_times
    
    return {
        "success": True,
        "bandwidths": pd.Series(bandwidths, index=distances_series.index),
        "max_delay": np.max(delays),
        "delays": pd.Series(delays, index=distances_series.index)
    }

    def optimize_batch(self, group_id, base_clients, candidate_clients):
        """批量计算多个候选客户端的延迟"""
        all_delays = {}
        base_clients_set = set(base_clients)
        
        # 预计算基础信息
        selected_clients = self.client_edge_info[self.client_edge_info["client_id"].isin(base_clients)]
        selected_clients = selected_clients.set_index("client_id")
        
        for candidate in candidate_clients:
            if candidate in base_clients_set:
                continue
            
            # 只计算新增的客户端
            temp_clients = selected_clients.copy()
            candidate_info = self.client_edge_info[self.client_edge_info["client_id"] == candidate].set_index("client_id")
            temp_clients = pd.concat([temp_clients, candidate_info])
            
            if self.config.get("algorithm", "gp") == "gp":
                result = self.optimize_bandwidth_gp(
                    temp_clients[f"server_{group_id}"],
                    temp_clients["power"],
                    temp_clients["compute_time"],
                    self.total_bandwidth,
                    self.data_size
                )
            elif self.config.get("algorithm", "gp") == "binary":
                result = self.optimize_bandwidth_binary(
                    temp_clients[f"server_{group_id}"],
                    temp_clients["power"],
                    temp_clients["compute_time"],
                    self.total_bandwidth,
                    self.data_size
                )
            else:
                result = self.optimize_bandwidth_simple(
                    temp_clients[f"server_{group_id}"],
                    temp_clients["power"],
                    temp_clients["compute_time"],
                    self.total_bandwidth,
                    self.data_size
                )
            
            if result["success"]:
                all_delays[candidate] = result["delays"][candidate]
        
        return all_delays

    def calculate_kl_divergence_fast(self, current_dist, current_num, client_info):
        """优化的KL散度计算"""
        client_num, client_dist = client_info
        new_dist = (current_dist * current_num + client_dist * client_num) / (current_num + client_num)
        exp_dist = np.full(10, 0.1)  # 1/10 for each class
        
        # 使用向量化操作
        new_dist = np.maximum(new_dist, 1e-12)
        return np.sum(new_dist * np.log(new_dist / exp_dist))


def simple_plot(result):
    """简单的结果可视化"""
    if not result["success"]:
        print(f"优化失败: {result['error']}")
        return
        
    client_num = len(result["bandwidths"])
    clients = np.arange(client_num) + 1
    
    # 创建简单图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 带宽分配
    ax1.bar(clients, result["bandwidths"])
    ax1.set_xlabel("客户端")
    ax1.set_ylabel("带宽 (MHz)")
    ax1.set_title("带宽分配")
    
    # 延迟
    ax2.bar(clients, result["delays"])
    ax2.axhline(y=result["max_delay"], color='r', linestyle='-', label=f'最大延迟: {result["max_delay"]:.2f}秒')
    ax2.set_xlabel("客户端")
    ax2.set_ylabel("延迟 (秒)")
    ax2.set_title("客户端延迟")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("bandwidth_optimization.png")
    plt.show()