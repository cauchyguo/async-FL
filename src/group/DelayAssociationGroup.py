from group.AbstractGroup import AbstractGroup
import pandas as pd
import numpy as np
import math



class DelayAssociationGroup(AbstractGroup):
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

    def group(self,client_list,latency_list):
        self.init = True

        # 从CSV文件加载距离矩阵
        client_edge_df = pd.read_csv(self.client_edge_distance_path)
        edge_server_count = len([col for col in client_edge_df.columns if col.startswith('server_')])
        self.edge_server_num = min(edge_server_count, self.edge_server_num)
        self.client_num = client_edge_df.shape[0]
        

        print("时延进行匹配边缘服务器") #按照平均带宽产生的时延进行分配
        for client_idx in range(self.client_num):
            system_time = []
            client_compute_delay = client_edge_df['compute_time'].iloc[client_idx]
            for i in range(self.edge_server_num):
                server_col = f'server_{i}'
                trans_rate = self.calculate_data_rate(client_edge_df[server_col].iloc[client_idx], client_edge_df['power'].iloc[client_idx], self.mean_bandwidth)
                trans_time = self.data_size * 8 / trans_rate
                system_time.append(client_compute_delay + trans_time)
            closest_server = np.argmax(system_time)
            self.group_list[closest_server].append(client_idx)
        
        for i in range(self.edge_server_num):
            print(f"edge_server_{i}: {self.group_list[i]}")
            
        return self.group_list,self.edge_server_num

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