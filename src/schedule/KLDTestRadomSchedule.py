import random

from networkx import union

from schedule.AbstractSchedule import AbstractSchedule
import numpy as np

def calculate_kld(p, q):
    """
    计算两个分布之间的KL散度。
    """
    p = np.maximum(p, 1e-12)  # 避免除以0
    p = p / np.sum(p) # 归一化处理
    q = np.maximum(q, 1e-12)  # 避免除以0
    return np.sum(p * np.log(p / q))

class KLDTestRadomSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.c_ratio = config["c_ratio"]
        self.start = config["start"]

    def schedule(self, client_list,data_dist):
        select_num = int(self.c_ratio * len(client_list))
        
        client_distribution = data_dist
        union_distribution = np.mean(client_distribution, axis=0)
        union_distribution = union_distribution / np.sum(union_distribution)

        print("Current clients:", len(client_list), ", select:", select_num)
        selected_indices = client_list[self.start:self.start+select_num]
        selected_client_distribution = np.sum([client_distribution[i] for i in selected_indices], axis=0)
        selected_client_distribution = selected_client_distribution / np.sum(selected_client_distribution)
        print("KLD of selected_client_distribution:",calculate_kld(selected_client_distribution,union_distribution))
        print("Mean KLD of client_distribution:",np.mean([calculate_kld(client_distribution[i],union_distribution) for i in selected_indices]))
        return selected_indices, selected_client_distribution
    
    def convert_to_distribution(self,data_dist,class_num=10):
        """
        将类别列表转化为概率分布。
        :param categories: 包含类别编号的整型列表。
        :return: 表示概率分布的数组。
        """
        client_distribution = []
        for client_categories in data_dist.values():
            # 初始化概率分布数组
            # distribution = np.zeros(class_num)
            
            # # 更新概率分布
            # for category in client_categories:
            #     distribution[category] = 1
            
            # 归一化概率分布
            distribution /= len(client_categories)
            client_distribution.append(distribution)
        return client_distribution

