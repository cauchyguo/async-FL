import random
import pandas as pd
from schedule.AbstractSchedule import AbstractSchedule
import numpy as np
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter
import pandas as pd

class HiFlashSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.min_join_clients_num = config.get("min_join_clients_num", 10)
        self.init_select_clients_num = config.get("init_select_clients_num", 2)
        self.group_kl_divergence = None
        self.group_data_num = 0
        self.group_class_distribution = np.full(10, 1 / 10)
        self.label_columns = None  # 将在schedule方法中初始化
        random.seed(42)

    # 选择客户端
    def schedule(self, client_list, client_label_df):
        # 初始化数据
        self.clients_edge_info_df = client_label_df
        self.label_columns = [col for col in self.clients_edge_info_df.columns if col.startswith('label')]
        
        # 创建客户端列表的副本，避免修改原始列表
        available_clients = client_list.copy()
        
        # 随机选择初始组群
        initial_count = min(max(self.init_select_clients_num, 1), len(available_clients))
        selected_client_threads = random.sample(available_clients, initial_count)
        
        # 从可用客户端中移除已选择的
        for client_id in selected_client_threads:
            available_clients.remove(client_id)

        # 计算初始组群的KL散度
        self.group_kl_divergence = self.cal_selected_clients_kldivergence(selected_client_threads)

        # 贪心选择客户端直到满足最小数量要求
        while len(selected_client_threads) < self.min_join_clients_num and available_clients:
            # 计算每个客户端的适应度
            fitness_list = [(client_id, self.cal_fitness(client_id)) for client_id in available_clients]
            
            # 重新打乱available_clients，避免iid时，选择相同的客户端
            random.shuffle(available_clients)
            # 选择适应度最小的客户端
            best_client_id, _ = min(fitness_list, key=lambda x: x[1])
            
            # 更新组群状态
            self.group_kl_divergence, self.group_data_num, self.group_class_distribution = \
                self.judge_join_client(best_client_id)
                
            # 添加最佳客户端并从可用池中移除
            selected_client_threads.append(best_client_id)
            available_clients.remove(best_client_id)

        return selected_client_threads
    
    def cal_fitness(self,client_id):
        return  self.judge_join_client(client_id)[0]

    def cal_selected_clients_kldivergence(self,selected_clients):
        group_data_num = 0
        group_client_data_distribution = []
        for client_id in selected_clients:
            client_data_num = self.clients_edge_info_df['data_num'].iloc[client_id]
            group_data_num += client_data_num
            self.label_columns = [col for col in self.clients_edge_info_df.columns if col.startswith('label')]
            client_data_distribution = self.clients_edge_info_df[self.label_columns].iloc[client_id].to_list()
            client_data_distribution = np.array(client_data_distribution)
            client_data_distribution = client_data_distribution / sum(client_data_distribution)
            group_client_data_distribution.append(client_data_distribution * client_data_num)
        group_class_distribution = np.sum(group_client_data_distribution,axis=0) / group_data_num
        self.group_data_num = group_data_num
        self.group_class_distribution = group_class_distribution
        return self.calculate_group_kl_divergence(group_class_distribution)
    
    def judge_join_client(self,client_id):
        # 计算加入该客户端后的KL散度
        client_data_num = self.clients_edge_info_df['data_num'].iloc[client_id]
        client_data_distribution = self.clients_edge_info_df[self.label_columns].iloc[client_id].to_numpy()
        client_data_distribution = client_data_distribution / sum(client_data_distribution)
        group_class_distribution = self.group_class_distribution * self.group_data_num + client_data_distribution * client_data_num
        group_class_distribution = group_class_distribution / sum(group_class_distribution)
        group_data_num = self.group_data_num + client_data_num
        return self.calculate_group_kl_divergence(group_class_distribution),group_data_num,group_class_distribution

    
    def calculate_group_kl_divergence(self,group_class_distribution):
        """
        计算两个分布之间的KL散度。
        :param p: 分布P，一个概率分布数组。
        :param q: 分布Q，另一个概率分布数组。
        :return: P和Q之间的KL散度。
        """
        exp_class_distribution = np.full(10, 1 / 10)
        group_dist = np.maximum(group_class_distribution, 1e-12)
        union_dist = np.maximum(exp_class_distribution, 1e-12)
        return np.sum(group_dist * np.log(group_dist / union_dist)) 

