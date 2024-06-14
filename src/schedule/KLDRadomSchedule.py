import random
from itertools import combinations
from networkx import union

from schedule.AbstractSchedule import AbstractSchedule
import numpy as np

def calculate_kld(p, q):
    """
    计算两个分布之间的KL散度。
    """
    q = np.maximum(q, 1e-12)  # 避免除以0
    return np.sum(p * np.log(p / q))

class KLDRadomSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.c_ratio = config["c_ratio"]

    def schedule(self, client_list,data_dist):
        select_num = int(self.c_ratio * len(client_list))
        
        client_distribution = self.convert_to_distribution(data_dist,class_num=10)
        # union_distribution = np.mean(client_distribution, axis=0)

        print("Current clients:", len(client_list), ", select:", select_num)
        selected_client_threads = random.sample(client_list, select_num)

        return selected_client_threads,np.mean([client_distribution[k] for k in selected_client_threads], axis=0)
    
    def convert_to_distribution(self,data_dist,class_num=10):
        """
        将类别列表转化为概率分布。
        :param categories: 包含类别编号的整型列表。
        :return: 表示概率分布的数组。
        """
        client_distribution = []
        for client_categories in data_dist.values():
            # 初始化概率分布数组
            distribution = np.zeros(class_num)
            
            # 更新概率分布
            for category in client_categories:
                distribution[category] = 1
            
            # 归一化概率分布
            distribution /= len(client_categories)
            client_distribution.append(distribution)
        return client_distribution

