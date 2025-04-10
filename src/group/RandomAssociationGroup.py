from group.AbstractGroup import AbstractGroup
import pandas as pd
import numpy as np
import math
import psutil
import logging



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
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def log_memory_usage(self, stage):
        process = psutil.Process()
        memory_info = process.memory_info()
        self.logger.info(f"Memory usage at {stage}: {memory_info.rss / 1024 / 1024:.2f} MB")

    def group(self, client_list, latency_list):
        self.init = True
        self.log_memory_usage("start")

        # 从CSV文件加载距离矩阵
        client_edge_df = pd.read_csv(self.client_edge_distance_path)
        self.log_memory_usage("after loading csv")
        
        edge_server_count = len([col for col in client_edge_df.columns if col.startswith('server_')])
        self.edge_server_num = min(edge_server_count, self.edge_server_num)
        self.client_num = client_edge_df.shape[0]
        
        self.logger.info(f"Total clients: {self.client_num}, Edge servers: {self.edge_server_num}")
        
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
        
        self.log_memory_usage("after calculating delay matrix")
        
        # 计算每个客户端的平均时延
        avg_delays = np.mean(delay_matrix, axis=1)
        
        # 获取时延的最小值和最大值
        min_delay = np.min(avg_delays)
        max_delay = np.max(avg_delays)
        
        # 初始化分组列表
        self.group_list = [[] for _ in range(self.edge_server_num)]
        
        # 将客户端按平均时延排序
        sorted_clients = np.argsort(avg_delays)
        
        # 确保每个边缘服务器至少有15个客户端
        min_clients_per_server = 15
        
        # 首先进行初始分配，确保每个服务器至少有最小数量的客户端
        for i in range(self.edge_server_num):
            # 计算每个服务器应该分配的基本客户端数量
            base_clients = min_clients_per_server
            # 从排序后的客户端列表中按顺序分配
            for j in range(base_clients):
                if i * base_clients + j < len(sorted_clients):
                    self.group_list[i].append(sorted_clients[i * base_clients + j])
        
        self.log_memory_usage("after initial assignment")
        
        # 剩余的客户端进行轮询分配
        remaining_clients = set(sorted_clients) - set([client for group in self.group_list for client in group])
        remaining_clients = sorted(list(remaining_clients), key=lambda x: avg_delays[x])
        
        # 轮询分配剩余客户端
        current_server = 0
        for client_idx in remaining_clients:
            self.group_list[current_server].append(client_idx)
            current_server = (current_server + 1) % self.edge_server_num
        
        self.log_memory_usage("after final assignment")
        
        # 打印各边缘服务器的分配情况和时延范围
        for i in range(self.edge_server_num):
            if len(self.group_list[i]) > 0:
                server_delays = [avg_delays[client_idx] for client_idx in self.group_list[i]]
                self.logger.info(f"edge_server_{i}: {self.group_list[i]}")
                self.logger.info(f"  delay range: {min(server_delays):.2f} - {max(server_delays):.2f}")
                self.logger.info(f"  client count: {len(self.group_list[i])}")
            else:
                self.logger.info(f"edge_server_{i}: []")
                self.logger.info(f"  delay range: N/A")
                self.logger.info(f"  client count: 0")
            
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