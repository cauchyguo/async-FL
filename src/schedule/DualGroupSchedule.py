import random
import pandas as pd
from schedule.AbstractSchedule import AbstractSchedule
import numpy as np

class DualGroupSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.c_ratio = config["c_ratio"]
        self.clients_edge_info_df = pd.read_csv(config["clients_edge_info_path"])
        self.min_join_clients_num = config.get("min_join_clients_num", 10)
        self.max_join_clients_num = config.get("max_join_clients_num", 20)
        self.target_kl_divergence = config.get("target_kl_divergence", 0.05)
        self.group_kl_divergence = None
        self.theta = config["theta"]

    def schedule(self,edge_server_idx, edge_server_unique_clients,edge_server_shared_clients):
        # 贪心决策
        # 现从独占组中随机选择一个客户端
        # 然后从共享组中选择一个延迟最小的客户端
        # 如果延迟相同，则随机选择一个

        selected_client_threads = []
        self.client_distance = self.clients_edge_info_df[f"server_{edge_server_idx}"]
        # 现从独占组中随机选择若干个客户端
        init_select_client = random.sample(edge_server_unique_clients, int(len(edge_server_unique_clients) * self.c_ratio))
        for client_id in init_select_client:
            edge_server_unique_clients.remove(client_id)
            selected_client_threads.append(client_id)
        print(f"Server {edge_server_idx} init_select_client: {init_select_client}")
        # 计算初始组群的KL散度
        self.group_kl_divergence = self.cal_selected_clients_kldivergence(selected_client_threads)
        # 然后从共享组中选择一个延迟最小的客户端
        available_clients_set = list(set(edge_server_shared_clients + edge_server_unique_clients))
        while len(selected_client_threads) < self.min_join_clients_num or self.group_kl_divergence > self.target_kl_divergence:
            fitness_list = []
            for client_id in available_clients_set:
                client_fitness = self.cal_fitness(client_id)
                fitness_list.append(client_fitness)
            best_fitness_index = fitness_list.index(min(fitness_list))
            self.group_kl_divergence,self.group_data_num,self.group_class_distribution = self.judge_join_client(available_clients_set[best_fitness_index])
            selected_client_threads.append(available_clients_set[best_fitness_index])
            available_clients_set.remove(available_clients_set[best_fitness_index])
            if len(selected_client_threads) >= self.max_join_clients_num:
                break
        print(f"Then, Server {edge_server_idx} selected_client_threads: {[item for item in selected_client_threads if item not in init_select_client]}")
        return selected_client_threads
    
    def cal_fitness(self,client_id):
        return self.theta * self.client_distance[client_id] / max(self.client_distance) +  self.judge_join_client(client_id)[0]

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
        exp_class_distribution = np.full(len(group_class_distribution), 1 / len(group_class_distribution))
        group_dist = np.maximum(group_class_distribution, 1e-12)
        union_dist = np.maximum(exp_class_distribution, 1e-12)
        return np.sum(group_dist * np.log(group_dist / union_dist)) 

