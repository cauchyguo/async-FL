import random
import pandas as pd
from schedule.AbstractSchedule import AbstractSchedule
import numpy as np
from utils import ModuleFindTool
import time

from algorithm import DelayAdaptiveOptimizer

class DualGroupScheduleDRCS(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        
        self.c_ratio = config.get("c_ratio", 0.2)
        self.clients_edge_info_df = pd.read_csv(config["clients_edge_info_path"])
        self.min_join_clients_num = config.get("min_join_clients_num", 10)
        self.max_join_clients_num = config.get("max_join_clients_num", 20)
        self.target_kl_divergence = config.get("target_kl_divergence", 0.05)
        self.group_kl_divergence = None
        self.theta = config["theta"]
        self.init_select_clients_num = config.get("init_select_clients_num", 5)

        self.init_select_unique_clients_num = config.get("init_select_unique_clients_num", 3)
        self.init_select_shared_clients_num = config.get("init_select_shared_clients_num", 5)

        self.delay_adaptive_optimizer = DelayAdaptiveOptimizer(config["delay_adaptive_optimizer"]["params"])

        self.bandwidth_type = config.get("bandwidth_type", "equal")

        if self.bandwidth_type == "equal":
            self.bandwidth_each = 5


        




        


    def schedule(self,edge_server_idx, edge_server_unique_clients,edge_server_shared_clients,client_label_df):
        # 贪心决策
        start_time = time.time()
        self.group_id = edge_server_idx # 边缘服务器编号
        self.selected_client_threads = []
        self.client_label_df = client_label_df
        self.client_distance = self.clients_edge_info_df[f"server_{edge_server_idx}"]
        # 现从独占组中随机选择若干个客户端
        # if len(edge_server_unique_clients) < self.min_join_clients_num:
        #     print("debug")

        if self.init_select_unique_clients_num == self.min_join_clients_num:
            init_select_client = random.sample(edge_server_unique_clients, min(self.init_select_unique_clients_num,len(edge_server_unique_clients)))
            return init_select_client
        
        if self.bandwidth_type == "equal":
            pass
            

        available_clients_set = list(set(edge_server_shared_clients + edge_server_unique_clients))

        
        # init_select_client = random.sample(edge_server_unique_clients, min(self.init_select_unique_clients_num,len(edge_server_unique_clients)))
        init_select_client = random.sample(available_clients_set, min(self.init_select_clients_num,len(available_clients_set)))


        for client_id in init_select_client:
            available_clients_set.remove(client_id)
            self.selected_client_threads.append(client_id)

        # init_select_shared_client = random.sample(edge_server_shared_clients, min(self.init_select_shared_clients_num,len(edge_server_shared_clients)))
        # for client_id in init_select_shared_client:
        #     edge_server_shared_clients.remove(client_id)
        #     self.selected_client_threads.append(client_id)

        init_select_client = init_select_client 
        # + init_select_shared_client


        print(f"Server {edge_server_idx} init_select_client: {init_select_client}")
        # 计算初始组群的KL散度
        self.group_kl_divergence = self.cal_selected_clients_kldivergence(self.selected_client_threads)

        # 然后从共享组中选择综合延迟和使得KL散度最小的客户端
        # available_clients_set = list(set(edge_server_shared_clients + edge_server_unique_clients))
        print(f"candidate {len(available_clients_set)} clients: {available_clients_set}")

        # 预计算所有客户端的数据分布，避免重复计算
        client_distributions = {}
        for client_id in available_clients_set:
            client_data_num = self.client_label_df['data_num'].iloc[client_id]

            self.label_columns = [col for col in self.client_label_df.columns if col.startswith('label')]
            client_dist = self.client_label_df[self.label_columns].iloc[client_id].to_numpy()
            client_dist = client_dist / sum(client_dist)
            client_distributions[client_id] = (client_data_num, client_dist)


        delay_trans_list = []
        kl_trans_list = []
        while len(self.selected_client_threads) < self.min_join_clients_num:
            if len(self.selected_client_threads) >= self.max_join_clients_num or len(available_clients_set) == 0:
                break

            fiter_available_clients_set = []
            bad_three_clients = []
            for client_id in available_clients_set:
                kl_div = self.calculate_kl_divergence_fast(
                    self.group_class_distribution,
                    self.group_data_num,
                    client_distributions[client_id]
                )
                if kl_div < self.group_kl_divergence:
                    fiter_available_clients_set.append([client_id,kl_div])
                else:
                    if len(bad_three_clients) < 3:
                        bad_three_clients.append([client_id,kl_div])
                    else:
                        if kl_div < min(bad_three_clients,key=lambda x:x[1])[1]:
                            bad_three_clients.remove(min(bad_three_clients,key=lambda x:x[1]))
                            bad_three_clients.append([client_id,kl_div])
                            
            fiter_available_clients_set.extend(bad_three_clients)

            fiter_available_clients_id_set = [item[0] for item in fiter_available_clients_set] 

            all_delays = self.delay_adaptive_optimizer.optimize_batch(
                self.group_id, 
                self.selected_client_threads,
                fiter_available_clients_id_set
            )


            rank_clients = {}
            for client_id,kld in fiter_available_clients_set:
                rank_clients[client_id] = {
                    "kld":kld,
                    "delay":all_delays[client_id]
                }
            
            # 两个排名求和，升序排序

            rank_kld = sorted(rank_clients,key=lambda x:rank_clients[x]["kld"])
            rank_delay = sorted(rank_clients,key=lambda x:rank_clients[x]["delay"])

            # 比较client_id 在rank_kld和rank_delay中的索引和,
            rank_sum = []
            for client_id in fiter_available_clients_id_set:
                rank_clients[client_id]["rank"] = rank_kld.index(client_id) + rank_delay.index(client_id)
                # rank_sum.append(rank_kld.index(client_id) + rank_delay.index(client_id))
            
            # 选择最小索引和的client_id，如果索引和相同，则选择kl_div最小的client_id
            final_rank = sorted(rank_clients,key=lambda x:rank_clients[x]["rank"])
            best_client_id = final_rank[0]


            # 更新状态
            client_data, client_dist = client_distributions[best_client_id]
            self.group_class_distribution = (self.group_class_distribution * self.group_data_num + client_dist * client_data) / (self.group_data_num + client_data)
            self.group_data_num += client_data
            
            self.selected_client_threads.append(best_client_id)
            available_clients_set.remove(best_client_id)

            # delay_trans_list.append(all_delays[best_client_id])
            # kl_trans_list.append(self.calculate_kl_divergence_fast(
            #     self.group_class_distribution,
            #     self.group_data_num,
            #     client_distributions[best_client_id]
            # ))
            # available_clients_set.remove(best_client_id)
            


            # rank_sum = [rank_kld[i][0] for i in range(len(rank_kld))] + [rank_delay[i][0] for i in range(len(rank_delay))]

            # # 选择前三个
            # select_clients = rank_clients[:3]
            

                

            
            # # 批量计算KL散度

            # best_client_id = available_clients_set[0]
            # best_fitness = float('100000')
            # delay_trans_list.append(-1)
            # kl_trans_list.append(-1)
            
            # for client_id in available_clients_set:
            #     delay = all_delays[client_id]
            #     kl_div = self.calculate_kl_divergence_fast(
            #         self.group_class_distribution,
            #         self.group_data_num,
            #         client_distributions[client_id]
            #     )
            #     fitness = kl_div + self.theta * delay
                
            #     if fitness < best_fitness:
            #         best_fitness = fitness
            #         best_client_id = client_id
            #         delay_trans_list[-1] = delay
            #         kl_trans_list[-1] = kl_div


            # fiter_available_clients_set = []
            
            # # 更新状态
            # client_data, client_dist = client_distributions[best_client_id]
            # self.group_class_distribution = (self.group_class_distribution * self.group_data_num + client_dist * client_data) / (self.group_data_num + client_data)
            # self.group_data_num += client_data
            
            # self.selected_client_threads.append(best_client_id)
            # available_clients_set.remove(best_client_id)

        print(f"Then, Server {edge_server_idx} self.selected_client_threads: {[item for item in self.selected_client_threads if item not in init_select_client]}")
        # 输出选择客户端子集的KL散度以及时延

        if len(kl_trans_list) > 0:
            print(f"KL range:",end='')
            for i in range(len(kl_trans_list)):
                print(f"{round(kl_trans_list[i],4)} ->",end='')
            print()
            print(f"delay range:",end='')
            for i in range(len(delay_trans_list)):
                print(f"{round(delay_trans_list[i],2)} -> ",end='')
            print()
            sys_cost = time.time() - start_time
            delay = delay_trans_list[-1] - sys_cost
            print(f"sys_cost: {sys_cost}")
        return self.selected_client_threads
    
    def cal_fitness(self,client_id):
        selected_clients_temp = self.selected_client_threads + [client_id]
        delay_for_selected_clients = self.delay_adaptive_optimizer.optimize(self.group_id,selected_clients_temp)
        delay_for_selected_clients = delay_for_selected_clients.loc[client_id]
        # delay_for_selected_clients = scheduler.delay_adaptive_optimizer.optimize(self.group_id,selected_clients)

        return self.judge_join_client(client_id)[0] + self.theta * delay_for_selected_clients

    def cal_selected_clients_kldivergence(self,selected_clients):

        if len(selected_clients) == 0:
            self.group_kl_divergence = -1
            self.group_data_num = 0
            self.group_class_distribution = np.full(10, 1 / 10)
            return -1

        group_data_num = 0
        group_client_data_distribution = []
        for client_id in selected_clients:
            client_data_num = self.client_label_df['data_num'].iloc[client_id]
            group_data_num += client_data_num
            self.label_columns = [col for col in self.client_label_df.columns if col.startswith('label')]
            client_data_distribution = self.client_label_df[self.label_columns].iloc[client_id].to_list()
            client_data_distribution = np.array(client_data_distribution)
            client_data_distribution = client_data_distribution / sum(client_data_distribution)
            group_client_data_distribution.append(client_data_distribution * client_data_num)
        group_class_distribution = np.sum(group_client_data_distribution,axis=0) / group_data_num
        self.group_data_num = group_data_num
        self.group_class_distribution = group_class_distribution


        return self.calculate_group_kl_divergence(group_class_distribution)
    
    def judge_join_client(self,client_id):
        # 计算加入该客户端后的KL散度
        client_data_num = self.client_label_df['data_num'].iloc[client_id]
        client_data_distribution = self.client_label_df[self.label_columns].iloc[client_id].to_numpy()
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

    def calculate_kl_divergence_fast(self, current_dist, current_num, client_info):
        """优化的KL散度计算"""
        client_num, client_dist = client_info
        new_dist = (current_dist * current_num + client_dist * client_num) / (current_num + client_num)
        exp_dist = np.full(10, 0.1)  # 1/10 for each class
        
        # 使用向量化操作
        new_dist = np.maximum(new_dist, 1e-12)
        return np.sum(new_dist * np.log(new_dist / exp_dist))

