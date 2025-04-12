import copy
import math
import numpy as np

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter


class PyramidFL(AbstractUpdate):
    def __init__(self, config):
        """
        初始化PyramidFL更新器
        
        参数:
            config: 配置字典，包含算法参数
        """
        self.config = config  # 保存配置信息
        self.global_var = GlobalVarGetter.get()  # 获取全局变量
        # 客户端统计信息，用于跟踪性能指标以进行选择
        self.client_statistics = self.global_var['client_statistics']
        self.current_round = 0  # 当前全局轮次

    def update_server_weights(self, epoch, update_list):
        """
        实现带有PyramidFL客户端选择的FedAvg聚合方法
        
        参数:
            epoch: 当前全局训练轮次
            update_list: 包含权重和元数据的客户端更新列表
            
        返回:
            updated_parameters: 新的全局模型参数
            None: 没有特定的权重需要传递给客户端(使用更新后的参数)
        """
        self.current_round = epoch  # 更新当前轮次
        
        # 根据收到的更新更新客户端统计信息
        for update_dict in update_list:
            client_id = update_dict.get("client_id")  # 获取客户端ID
            if client_id is not None:
                # 构建训练结果字典
                training_results = {
                    'loss': update_dict.get("loss", 0),  # 训练损失
                    'gradient_norm': update_dict.get("gradient_norm", 0),  # 梯度范数
                    'sample_size': update_dict.get("data_sum", 0),  # 样本数量
                    'execution_time': update_dict.get("execution_time", 1),  # 执行时间
                }
                self._update_client_statistics(client_id, training_results)  # 更新客户端统计信息
        
        # 执行标准FedAvg聚合
        total_nums = 0  # 总样本数
        for update_dict in update_list:
            total_nums += update_dict["data_sum"]  # 累加所有客户端的样本数
            
        updated_parameters = {}  # 初始化更新后的参数字典
        # 处理第一个客户端的权重
        for key, var in update_list[0]["weights"].items():
            # 按样本数量加权平均
            updated_parameters[key] = update_list[0]["weights"][key] * update_list[0]["data_sum"] / total_nums
            
        # 处理剩余客户端的权重
        for i in range(len(update_list) - 1):
            update_dict = update_list[i + 1]  # 获取当前客户端更新
            client_weights = update_dict["weights"]  # 获取客户端权重
            for key, var in client_weights.items():
                # 累加加权权重
                updated_parameters[key] += client_weights[key] * update_dict["data_sum"] / total_nums
        
        # 使用客户端统计信息更新调度器
        try:
            from utils.ModuleFindTool import find_instance_by_path
            # 查找调度器实例
            scheduler_instance = find_instance_by_path(self.config['scheduler']['path'])
            # 如果调度器有更新客户端统计信息的方法，则调用它
            if hasattr(scheduler_instance, 'schedule') and hasattr(scheduler_instance.schedule, 'update_client_statistics'):
                scheduler_instance.schedule.update_client_statistics(self.client_statistics)
        except:
            pass  # 忽略可能的异常

        # self.global_var['epoch_avg_loss'][epoch] = 0
        # self.global_var['epoch_group_data_util'][epoch] = 0
        # self.global_var['epoch_group_data_util_accumulate'][epoch] = 0
                
        return updated_parameters, None  # 返回更新后的参数和None


    def _update_client_statistics(self, client_id, training_results):
        """
        根据训练结果更新客户端统计信息
        
        参数:
            client_id: 客户端ID
            training_results: 包含训练指标的字典
        """
        # 如果是新客户端，初始化其统计信息
        if client_id not in self.client_statistics:
            self.client_statistics[client_id] = {
                'count': 0,  # 参与次数
                'history': []  # 历史记录
            }
        
        # 更新统计信息
        self.client_statistics[client_id].update({
            'loss': training_results['loss'],  # 更新损失
            'gradient_norm': training_results['gradient_norm'],  # 更新梯度范数
            'sample_size': training_results['sample_size'],  # 更新样本数量
            'execution_time': training_results['execution_time'],  # 更新执行时间
            'time_stamp': self.current_round,  # 更新时间戳
            'count': self.client_statistics[client_id]['count'] + 1  # 增加参与次数
        })
        
        # 保存历史记录
        self.client_statistics[client_id]['history'].append({
            'round': self.current_round,  # 当前轮次
            'loss': training_results['loss'],  # 损失
            'execution_time': training_results['execution_time']  # 执行时间
        })
        
        return self.client_statistics  # 返回更新后的客户端统计信息