from group.AbstractGroup import AbstractGroup
import pandas as pd
import numpy as np
import os

class TiFLGroup(AbstractGroup):
    def __init__(self, group_manager, config):
        self.group_manager = group_manager
        self.clients_info_path = config["clients_info_path"]
        self.tiers = 5  # 将客户端按延迟分为5层
        self.init = False
        
        # 确保输出目录存在
        output_dir = os.path.dirname(self.clients_info_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def group(self, client_list, latency_list):
        self.init = True
        
        # 加载客户端信息
        clients_df = pd.read_csv(self.clients_info_path)
        
        # 按延迟时间排序客户端
        sorted_indices = np.argsort(latency_list)
        sorted_clients = [client_list[i] for i in sorted_indices]
        sorted_latencies = [latency_list[i] for i in sorted_indices]
        
        # 计算每层的客户端数量
        clients_per_tier = len(client_list) // self.tiers
        remainder = len(client_list) % self.tiers
        
        # 创建层级分组
        group_list = [[] for _ in range(self.tiers)]
        
        # 根据排序后的延迟时间将客户端分配到各层
        client_idx = 0
        for tier in range(self.tiers):
            # 计算当前层应有的客户端数量
            tier_size = clients_per_tier + (1 if tier < remainder else 0)
            
            # 将客户端添加到当前层
            for _ in range(tier_size):
                if client_idx < len(sorted_clients):
                    group_list[tier].append(sorted_clients[client_idx])
                    client_idx += 1
        
        # 保存分组信息到CSV以供将来参考
        group_data = []
        for tier in range(self.tiers):
            for client_id in group_list[tier]:
                client_index = client_list.index(client_id)
                # 获取该客户端的数据量(可能需要从原始数据中提取)
                client_data_num = clients_df.loc[clients_df['client_id'] == client_id, 'data_num'].values[0] \
                                  if 'data_num' in clients_df.columns else 1000  # 默认值
                group_data.append({
                    'client_id': client_id,
                    'group_id': tier,
                    'time': latency_list[client_index],
                    'data_num': client_data_num
                })
        
        group_df = pd.DataFrame(group_data)
        group_df.to_csv(self.clients_info_path, index=False)
        
        # 保存层级信息以供调度器使用
        group_info = []
        for tier in range(self.tiers):
            tier_clients = group_list[tier]
            if tier_clients:
                tier_indices = [client_list.index(cid) for cid in tier_clients]
                tier_delays = [latency_list[idx] for idx in tier_indices]
                # 获取数据量，如果信息可用
                try:
                    tier_data = [clients_df.loc[clients_df['client_id'] == cid, 'data_num'].values[0] \
                                for cid in tier_clients]
                except:
                    tier_data = [1000] * len(tier_clients)  # 默认值
                
                group_info.append({
                    'group_id': tier,
                    'client_id_list': tier_clients,
                    'group_data_num': sum(tier_data),
                    'group_delay_time': max(tier_delays)
                })
        
        group_info_df = pd.DataFrame(group_info)
        tifl_group_path = os.path.join(os.path.dirname(self.clients_info_path), 'group_info.csv')
        group_info_df.to_csv(tifl_group_path, index=False)
        
        return group_list, len(group_list)

    def check_update(self):
        return self.init