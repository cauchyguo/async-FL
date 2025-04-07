import random
import pandas as pd
from schedule.AbstractSchedule import AbstractSchedule
import numpy as np
from utils import ModuleFindTool

class DualGroupSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.c_ratio = config.get("c_ratio", 0.2)
        self.clients_edge_info_df = pd.read_csv(config["clients_edge_info_path"])
        self.min_join_clients_num = config.get("min_join_clients_num", 10)
        self.max_join_clients_num = config.get("max_join_clients_num", 20)
        self.target_kl_divergence = config.get("target_kl_divergence", 0.05)
        self.group_kl_divergence = None
        self.theta = config["theta"]
        self.init_select_clients_num = config["init_select_clients_num"]

        delay_adaptive_optimizer_class = ModuleFindTool.find_class_by_path(config["delay_adaptive_optimizer"]["path"])
        self.delay_adaptive_optimizer = delay_adaptive_optimizer_class(config["delay_adaptive_optimizer"]["params"])


    def schedule(self,edge_server_idx, edge_server_unique_clients,edge_server_shared_clients):
        # 贪心决策
        self.group_id = edge_server_idx # 边缘服务器编号
        self.selected_client_threads = []
        self.client_distance = self.clients_edge_info_df[f"server_{edge_server_idx}"]
        # 现从独占组中随机选择若干个客户端
        # if len(edge_server_unique_clients) < self.min_join_clients_num:
        #     print("debug")
        init_select_client = random.sample(edge_server_unique_clients, min(self.init_select_clients_num,len(edge_server_unique_clients)))
        for client_id in init_select_client:
            edge_server_unique_clients.remove(client_id)
            self.selected_client_threads.append(client_id)
        print(f"Server {edge_server_idx} init_select_client: {init_select_client}")
        # 计算初始组群的KL散度
        self.group_kl_divergence = self.cal_selected_clients_kldivergence(self.selected_client_threads)
        # 然后从共享组中选择综合延迟和使得KL散度最小的客户端
        available_clients_set = list(set(edge_server_shared_clients + edge_server_unique_clients))
        print(f"candidate {len(available_clients_set)} clients: {available_clients_set}")
        kl_divergence_list = []
        delay_list = []
        while len(self.selected_client_threads) < self.min_join_clients_num:
            if len(self.selected_client_threads) >= self.max_join_clients_num or len(available_clients_set) == 0:
                break
            fitness_list = []
            delay_for_client_list = []
            kl_divergence_for_client_list = []
            for client_id in available_clients_set:
                selected_clients_temp = self.selected_client_threads + [client_id]
                delay_for_selected_clients = self.delay_adaptive_optimizer.optimize(self.group_id,selected_clients_temp)
                delay_for_selected_client = delay_for_selected_clients.loc[client_id]
                delay_for_client_list.append(delay_for_selected_client)
                
                kl_add_client = self.judge_join_client(client_id)[0]
                kl_divergence_for_client_list.append(kl_add_client)

                # client_fitness = self.cal_fitness(client_id)
                fitness_list.append(kl_add_client + self.theta * delay_for_selected_client)
                kl_divergence_for_client_list.append(kl_add_client)
            best_fitness_index = fitness_list.index(min(fitness_list))

            delay_list.append(delay_for_client_list[best_fitness_index])
            kl_divergence_list.append(kl_divergence_for_client_list[best_fitness_index])

            self.group_kl_divergence,self.group_data_num,self.group_class_distribution = self.judge_join_client(available_clients_set[best_fitness_index])
            self.selected_client_threads.append(available_clients_set[best_fitness_index])
            available_clients_set.remove(available_clients_set[best_fitness_index])

        print(f"Then, Server {edge_server_idx} self.selected_client_threads: {[item for item in self.selected_client_threads if item not in init_select_client]}")
        # 输出选择客户端子集的KL散度以及时延
        print(f"KL range:",end='')
        for i in range(len(delay_list)):
            print(f"{kl_divergence_list[i]}",end='')
        print()
        print(f"delay range:",end='')
        for i in range(len(delay_list)):
            print(f"{delay_list[i]}",end='')
        print()
        return self.selected_client_threads
    
    def cal_fitness(self,client_id):
        selected_clients_temp = self.selected_client_threads + [client_id]
        delay_for_selected_clients = self.delay_adaptive_optimizer.optimize(self.group_id,selected_clients_temp)
        delay_for_selected_clients = delay_for_selected_clients.loc[client_id]
        # delay_for_selected_clients = scheduler.delay_adaptive_optimizer.optimize(self.group_id,selected_clients)

        return self.judge_join_client(client_id)[0] + self.theta * delay_for_selected_clients

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

