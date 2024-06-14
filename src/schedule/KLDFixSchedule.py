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

class KLDFixSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.c_ratio = config["c_ratio"]

    def schedule(self, client_list,data_dist):
        select_num = int(self.c_ratio * len(client_list))
        
        client_distribution = self.convert_to_distribution(data_dist,class_num=10)
        union_distribution = np.mean(client_distribution, axis=0)

        # 初始化最佳选择和最小KL散度
        best_choice = None
        min_kld = float('inf')
        
        # 遍历所有可能的20个client的组合
        for combination in combinations(client_list, select_num):
            # 计算当前组合的总分布
            current_distribution = np.mean(client_distribution[list(combination)], axis=0)
            
            # 计算KL散度
            kld = calculate_kld(union_distribution, current_distribution)
            
            # 更新最佳选择和最小KL散度
            if kld < min_kld:
                min_kld = kld
                best_choice = combination

        return best_choice,np.mean(client_distribution[list(combination)], axis=0)
    
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

