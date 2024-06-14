import random

from networkx import union

from schedule.AbstractSchedule import AbstractSchedule
import numpy as np

def calculate_kld(p, q):
    """
    计算两个分布之间的KL散度。
    """
    q = np.maximum(q, 1e-12)  # 避免除以0
    return np.sum(p * np.log(p / q))

class KLDSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.c_ratio = config["c_ratio"]

    def schedule(self, client_list,data_dist):
        select_num = int(self.c_ratio * len(client_list))
        
        client_distribution = self.convert_to_distribution(data_dist,class_num=10)
        union_distribution = np.mean(client_distribution, axis=0)

        print("Current clients:", len(client_list), ", select:", select_num)
        selected_indices = random.sample(client_list, select_num)
        selected_client_distribution = np.mean([client_distribution[i] for i in selected_indices], axis=0)
        improvement = True
        while improvement:
            improvement = False
            for i in selected_indices:
                for j in range(len(client_list)):
                    if j not in selected_indices:
                        # 尝试替换
                        new_indices = selected_indices[:]
                        new_indices.remove(i)
                        new_indices.append(j)
                        new_selected_client_distribution = np.mean([client_distribution[k] for k in new_indices], axis=0)
                        
                        # 计算新旧KL散度
                        old_kld = calculate_kld(union_distribution, selected_client_distribution)
                        new_kld = calculate_kld(union_distribution, new_selected_client_distribution)
                        
                        # 如果新的KL散度更小，则执行替换
                        if new_kld < old_kld:
                            selected_indices = new_indices
                            selected_client_distribution = new_selected_client_distribution
                            improvement = True
                            break
                if improvement:
                    break

        selected_client_threads = [client_list[i] for i in selected_indices]
        return selected_client_threads, selected_client_distribution
    
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

