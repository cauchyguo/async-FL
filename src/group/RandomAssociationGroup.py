from group.AbstractGroup import AbstractGroup
import pandas as pd
import numpy as np
import math



class RandomAssociationGroup(AbstractGroup):
    def __init__(self, group_manager, config):
        self.group_manager = group_manager
        self.edge_server_num = config['edge_server_num']
        self.client_edge_distance_path = config["situation_path"]
        self.init = False
        self.nearest_K = config.get('nearest_K', 20)  # 参数K：用于选择最近的客户端数量        self.unique_groups = {}  
        self.total_bandwidth = config["total_bandwidth"]
        self.data_size = config["data_size"]
        self.min_join_clients_num = config["min_join_clients_num"]

        self.mean_bandwidth = self.total_bandwidth / self.min_join_clients_num

        self.group_list = [[] for _ in range(self.edge_server_num)]

    def group(self, client_list, latency_list):
        self.init = True

        # 从CSV文件加载距离矩阵
        client_edge_df = pd.read_csv(self.client_edge_distance_path)
        edge_server_count = len([col for col in client_edge_df.columns if col.startswith('server_')])
        self.edge_server_num = min(edge_server_count, self.edge_server_num)
        self.client_num = client_edge_df.shape[0]
        
        # 计算每个客户端到每个边缘服务器的时延
        delay_matrix = np.zeros((self.client_num, self.edge_server_num))
        for client_idx in range(self.client_num):
            client_compute_delay = client_edge_df['compute_time'].iloc[client_idx]
            for i in range(self.edge_server_num):
                server_col = f'server_{i}'
                trans_rate = self.calculate_data_rate(client_edge_df[server_col].iloc[client_idx], 
                                                    client_edge_df['power'].iloc[client_idx], 
                                                    self.mean_bandwidth)
                trans_time = self.data_size * 8 / trans_rate
                delay_matrix[client_idx, i] = client_compute_delay + trans_time
        
        # 计算每个客户端的平均时延
        avg_delays = np.mean(delay_matrix, axis=1)
        
        # 获取时延的最小值和最大值
        min_delay = np.min(avg_delays)
        max_delay = np.max(avg_delays)
        delay_range = max_delay - min_delay
        
        # 初始化分组列表
        self.group_list = [[] for _ in range(self.edge_server_num)]
        
        # 定义重叠系数（0表示无重叠，值越大重叠越多）
        overlap_factor = 0.2
        
        # 为每个边缘服务器定义一个时延区间，允许有重叠
        for i in range(self.edge_server_num):
            # 定义当前服务器的时延范围
            server_min_delay = min_delay + (i / self.edge_server_num) * (1 - overlap_factor) * delay_range
            server_max_delay = min_delay + ((i + 1) / self.edge_server_num + overlap_factor) * delay_range
            if i == self.edge_server_num - 1:
                server_max_delay = max_delay
            
            # 找出时延在此范围内的客户端
            for client_idx, delay in enumerate(avg_delays):
                if server_min_delay <= delay <= server_max_delay:
                    # 如果客户端已经被分配，检查是否应该重新分配
                    already_assigned = False
                    for group in self.group_list:
                        if client_idx in group:
                            already_assigned = True
                            break
                    
                    # 如果客户端尚未被分配，或者当前服务器更适合（更接近时延中心）
                    if not already_assigned:
                        self.group_list[i].append(client_idx)
        
        # 处理未分配的客户端（如果有）
        all_assigned = set()
        for group in self.group_list:
            all_assigned.update(group)
        
        unassigned = set(range(self.client_num)) - all_assigned
        if unassigned:
            # 将未分配客户端分给时延最匹配的服务器
            for client_idx in unassigned:
                delay = avg_delays[client_idx]
                best_server = 0
                min_distance = float('inf')
                
                for i in range(self.edge_server_num):
                    server_center = min_delay + (i + 0.5) / self.edge_server_num * delay_range
                    distance = abs(delay - server_center)
                    if distance < min_distance:
                        min_distance = distance
                        best_server = i
                
                self.group_list[best_server].append(client_idx)
        
        # 打印各边缘服务器的分配情况和时延范围
        for i in range(self.edge_server_num):
            if len(self.group_list[i]) > 0:
                server_delays = [avg_delays[client_idx] for client_idx in self.group_list[i]]
                print(f"edge_server_{i}: {self.group_list[i]}")
                print(f"  delay range: {min(server_delays):.2f} - {max(server_delays):.2f}")
                print(f"  client count: {len(self.group_list[i])}")
            else:
                print(f"edge_server_{i}: []")
                print(f"  delay range: N/A")
                print(f"  client count: 0")
            
        return self.group_list, self.edge_server_num

    def check_update(self):
        return self.init
    
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