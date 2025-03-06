from group.AbstractGroup import AbstractGroup
import pandas as pd
import numpy as np

class DualGroup(AbstractGroup):
    def __init__(self, group_manager, config):
        self.group_manager = group_manager
        self.edge_server_num = config['edge_server_num']
        self.client_edge_distance_path = config["situation_path"]
        self.init = False
        self.nearest_K = config.get('nearest_K', 20)  # 参数K：用于选择最近的客户端数量
        self.unique_groups = []  
        self.shared_groups = []  

    def group(self):
        self.init = True

        # 从CSV文件加载距离矩阵
        client_edge_df = pd.read_csv(self.client_edge_distance_path)
        edge_server_count = len([col for col in client_edge_df.columns if col.startswith('server_')])
        self.edge_server_num = min(edge_server_count, self.edge_server_num)
        self.client_num = client_edge_df.shape[0]
        
        # 为每个服务器初始化分组
        for i in range(self.edge_server_num):
            self.unique_groups[i] = set()
            self.shared_groups[i] = set()
        
        # 阶段1：终端主导分配
        # 对每个客户端，找到最近的服务器并分配到该服务器的独占组
        print("阶段1:终端主导分配...")
        for client_idx in range(self.client_num):
            server_distances = [client_edge_df.iloc[client_idx][f'server_{i}'] for i in range(self.edge_server_num)]
            closest_server = np.argmin(server_distances)
            self.unique_groups[closest_server].add(client_idx)
        
        
        # 阶段2：服务器主导分配
        # 对每个服务器，选择K个最近的客户端
        print("阶段2:服务器主导分配...")
        server_candidates = {}
        for server_idx in range(self.edge_server_num):
            # 获取所有客户端到此服务器的距离
            server_col = f'server_{server_idx}'
            distances = client_edge_df[server_col].values
            
            # 获取K个最近客户端的索引
            closest_clients_indices = np.argsort(distances)[:self.nearest_K]
            
            # 创建候选集（K个最近客户端减去已在独占组中的客户端）
            candidates = set(closest_clients_indices) - self.unique_groups[server_idx]
            server_candidates[server_idx] = candidates
        
        # 阶段3：混合分组构建
        print("阶段3:混合分组构建...")
        for server_idx in range(self.edge_server_num):
            # 计算共享组：该服务器候选集中也在其他服务器候选集中的客户端
            shared_clients = set()
            for other_server in range(self.edge_server_num):
                if other_server != server_idx:
                    # 先求交集，再求并集
                    shared_clients |= server_candidates[server_idx] & server_candidates[other_server] 
            
            self.shared_groups[server_idx] = shared_clients
            
            # 更新独占组：添加不在共享组中的候选客户端
            self.unique_groups[server_idx] |= server_candidates[server_idx] - shared_clients
        print("混合分组构建完成")
        print("--------------------------------")
        print("最终分组结果:")
        
        # 将集合转换为列表以提高兼容性
        for server_idx in range(self.edge_server_num):
            self.unique_groups[server_idx] = list(self.unique_groups[server_idx])
            self.shared_groups[server_idx] = list(self.shared_groups[server_idx])
            print(f"服务器{server_idx}的独占组: {self.unique_groups[server_idx]}")
            print(f"服务器{server_idx}的共享组: {self.shared_groups[server_idx]}")
            print("--------------------------------")
            
        return self.unique_groups, self.shared_groups,self.edge_server_num

    def check_update(self):
        return self.init