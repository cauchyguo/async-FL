import math
import numpy as np
import random

from schedule.AbstractSchedule import AbstractSchedule
from utils.GlobalVarGetter import GlobalVarGetter


class PyramidSchedule(AbstractSchedule):
    def __init__(self, config):
        """
        初始化PyramidSchedule客户端选择调度器
        
        参数:
            config: 配置字典，包含调度器参数
        """
        super().__init__(config)  # 调用父类初始化方法
        self.c_ratio = config.get("c_ratio", 0.3)  # 客户端选择比例，默认0.3
        self.exploration_factor = config.get("exploration_factor", 0.1)  # 探索因子，控制探索与利用的平衡，默认0.1
        self.cut_off_util = config.get("cut_off_util", 0.9)  # 效用截断比例，默认0.9
        self.round_threshold = config.get("round_threshold", 30)  # 时间阈值百分比，用于惩罚慢客户端，默认30
        
        # 获取全局变量并初始化
        self.global_var = GlobalVarGetter.get()
        if 'client_statistics' not in self.global_var:
            self.global_var['client_statistics'] = {}
        self.current_round = 0
        
    def schedule(self, client_list):
        """
        基于PyramidFL算法选择客户端
        
        参数:
            client_list: 可用客户端ID列表
            
        返回:
            selected_clients: 选择的客户端ID列表
        """
        self.current_round += 1  # 轮次递增
        select_num = int(self.c_ratio * len(client_list))  # 计算需要选择的客户端数量
        print(f"Current clients: {len(client_list)}, select: {select_num}")  # 打印选择信息
        
        # 获取客户端效用
        client_utilities = self._calculate_client_utilities(client_list)  # 计算所有客户端的效用
        
        # 使用PyramidFL算法选择客户端
        selected_clients = self._select_clients(
            client_utilities,  # 客户端效用字典
            client_list,  # 可用客户端列表
            select_num,  # 选择数量
            self.round_threshold,  # 时间阈值
            self.exploration_factor,  # 探索因子
            self.cut_off_util  # 效用截断比例
        )
        
        return selected_clients  # 返回选择的客户端列表
    
    def _calculate_client_utilities(self, client_list):
        """
        计算所有客户端的效用值，直接从全局变量获取统计信息
        
        参数:
            client_list: 可用客户端ID列表
            
        返回:
            client_utilities: 客户端效用字典
        """
        client_utilities = {}
        
        # 确保全局变量中有client_statistics
        if 'client_statistics' not in self.global_var:
            self.global_var['client_statistics'] = {}
            print("警告: 全局变量中未初始化client_statistics")
        
        for client_id in client_list:
            if client_id in self.global_var['client_statistics']:
                # 使用中括号语法获取统计信息
                stats = self.global_var['client_statistics'][client_id]
                
                # 计算统计效用（基于损失和样本数量）
                statistical_utility = math.sqrt(stats.get('loss', 1)) * stats.get('sample_size', 1)
                
                # 计算梯度效用（基于梯度范数和样本数量）
                gradient_norm = stats.get('gradient_norm', 0)
                gradient_utility = math.sqrt(gradient_norm) * stats.get('sample_size', 1) / 100
                
                # 创建客户端效用字典
                client_utilities[client_id] = {
                    'statistical_utility': statistical_utility,
                    'gradient_utility': gradient_utility,
                    'duration': stats.get('execution_time', 1),
                    'time_stamp': stats.get('time_stamp', 0),
                    'count': stats.get('count', 0),
                    'sample_size': stats.get('sample_size', 1)
                }
            else:
                # 为新客户端分配初始效用值
                client_utilities[client_id] = {
                    'statistical_utility': 1,
                    'gradient_utility': 0,
                    'duration': 1,
                    'time_stamp': 0,
                    'count': 0,
                    'sample_size': 1
                }
        
        return client_utilities  # 返回客户端效用字典
    
    def _select_clients(self, client_utilities, feasible_clients, num_to_select, 
                      round_threshold=30, exploration_factor=0.1, cut_off_util=0.9):
        """
        基于效用值选择客户端
        
        参数:
            client_utilities: 客户端效用字典
            feasible_clients: 可用客户端ID列表
            num_to_select: 需要选择的客户端数量
            round_threshold: 时间阈值百分比，用于惩罚慢客户端
            exploration_factor: 探索因子，控制探索与利用的平衡
            cut_off_util: 效用截断比例
            
        返回:
            selected_clients: 选择的客户端ID列表
        """
        # 过滤可用客户端
        blacklist = []  # 黑名单列表（可以根据需要实现黑名单逻辑）
        available_clients = [c for c in client_utilities.keys() 
                            if c in feasible_clients and c not in blacklist]  # 过滤得到可用客户端
        
        # 计算慢客户端的时间阈值
        if round_threshold < 100:  # 如果阈值小于100%
            sorted_durations = sorted([client_utilities[c]['duration'] for c in available_clients])  # 对执行时间排序
            threshold_index = min(int(len(sorted_durations) * round_threshold/100), len(sorted_durations)-1)  # 计算阈值索引
            round_prefer_duration = sorted_durations[threshold_index]  # 获取阈值时间
        else:
            round_prefer_duration = float('inf')  # 无限大，表示不限制执行时间
        
        # 收集奖励和时间间隔信息，用于归一化
        rewards = []  # 奖励列表
        staleness = []  # 时间间隔列表
        current_time = self.current_round  # 当前轮次
        
        for client_id in available_clients:  # 遍历所有可用客户端
            if client_utilities[client_id]['statistical_utility'] > 0:  # 如果统计效用大于0
                rewards.append(client_utilities[client_id]['statistical_utility'])  # 添加到奖励列表
                staleness.append(current_time - client_utilities[client_id]['time_stamp'])  # 计算并添加时间间隔
        
        # 归一化奖励
        if rewards:  # 如果有奖励数据
            max_reward = max(rewards)  # 最大奖励
            min_reward = min(rewards)  # 最小奖励
            range_reward = max(max_reward - min_reward, 1e-4)  # 奖励范围（避免除零）
            clip_value = max_reward  # 截断值
        else:
            max_reward, min_reward, range_reward, clip_value = 0, 0, 1, 0  # 默认值
            
        # 归一化时间间隔
        if staleness:  # 如果有时间间隔数据
            max_staleness = max(staleness)  # 最大间隔
            min_staleness = min(staleness)  # 最小间隔
            range_staleness = max(max_staleness - min_staleness, 1)  # 间隔范围
        else:
            max_staleness, min_staleness, range_staleness = 0, 0, 1  # 默认值
        
        # 计算分数
        scores = {}  # 分数字典
        exploited_clients = []  # 已利用的客户端列表（曾经选择过的）
        unexplored_clients = []  # 未探索的客户端列表（从未选择过的）
        
        for client_id in available_clients:  # 遍历所有可用客户端
            client = client_utilities[client_id]  # 获取客户端效用信息
            
            if client.get('count', 0) > 0:  # 如果客户端曾经被选择过（利用阶段）
                # 归一化奖励
                reward = min(client['statistical_utility'], clip_value)  # 截断奖励
                norm_reward = (reward - min_reward) / float(range_reward)  # 归一化
                
                # 计算时间不确定性（鼓励选择长时间未选择的客户端）
                temporal_uncertainty = math.sqrt(0.1 * math.log(current_time) / max(client['time_stamp'], 1))
                
                # 计算最终分数
                score = norm_reward + temporal_uncertainty  # 奖励加时间不确定性
                
                # 惩罚执行时间长的慢客户端
                if client['duration'] > round_prefer_duration:  # 如果执行时间超过阈值
                    penalty_factor = max((round_prefer_duration / max(1e-4, client['duration'])) ** 2, 0.5)  # 计算惩罚因子,最小为0.5
                    score *= penalty_factor  # 应用惩罚
                    
                scores[client_id] = score  # 保存分数
                exploited_clients.append(client_id)  # 添加到已利用列表
            else:  # 如果客户端从未被选择过（探索阶段）
                # 未探索的客户端
                unexplored_clients.append(client_id)  # 添加到未探索列表
        
        # 利用阶段选择
        exploit_count = min(int(num_to_select * (1.0 - exploration_factor)), len(exploited_clients))  # 计算利用阶段选择数量
        
        # 按分数排序客户端
        sorted_clients = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)  # 降序排序
        
        selected_exploit = []  # 利用阶段选择的客户端
        if sorted_clients:  # 如果有已利用的客户端
            # 计算截断分数
            cutoff_index = min(exploit_count, len(sorted_clients)-1)  # 截断索引
            cut_off_score = scores[sorted_clients[cutoff_index]] * cut_off_util  # 截断分数
            
            # 选择高于截断分数的客户端
            candidates = []  # 候选客户端
            for client_id in sorted_clients:  # 遍历排序后的客户端
                if scores[client_id] >= cut_off_score:  # 如果分数高于截断分数
                    candidates.append(client_id)  # 添加到候选列表
                else:
                    break  # 达到截断分数后停止
            
            # 按概率选择客户端
            if candidates:  # 如果有候选客户端
                probs = np.array([scores[c] for c in candidates])  # 获取分数数组
                probs = probs / np.sum(probs)  # 归一化为概率
                exploit_select = min(exploit_count, len(candidates))  # 选择数量
                
                if exploit_select > 0:  # 如果需要选择
                    # 按概率无放回抽样
                    selected_exploit = np.random.choice(
                        candidates, exploit_select, p=probs, replace=False).tolist()
        
        # 探索阶段选择
        explore_count = min(num_to_select - len(selected_exploit), len(unexplored_clients))  # 计算探索阶段选择数量
        
        selected_explore = []  # 探索阶段选择的客户端
        if unexplored_clients and explore_count > 0:  # 如果有未探索客户端且需要选择
            # 为未探索客户端分配初始奖励（基于样本数量）
            init_rewards = {}  # 初始奖励字典
            for client_id in unexplored_clients:  # 遍历未探索客户端
                init_reward = client_utilities[client_id].get('sample_size', 1)  # 获取样本数量作为初始奖励
                
                # 同样考虑系统效用（执行时间）
                if client_utilities[client_id]['duration'] > round_prefer_duration:  # 如果执行时间超过阈值
                    penalty = (round_prefer_duration / max(1e-4, client_utilities[client_id]['duration'])) ** 2  # 计算惩罚因子
                    init_reward *= penalty  # 应用惩罚
                    
                init_rewards[client_id] = max(init_reward, 1e-4)  # 保存初始奖励（避免为零）
            
            # 按初始奖励排序
            sorted_unexplored = sorted(init_rewards.keys(), key=lambda k: init_rewards[k], reverse=True)  # 降序排序
            
            if sorted_unexplored:  # 如果有排序后的未探索客户端
                probs = np.array([init_rewards[c] for c in sorted_unexplored])  # 获取奖励数组
                probs = probs / np.sum(probs)  # 归一化为概率
                
                # 按概率选择
                selected_explore = np.random.choice(
                    sorted_unexplored, explore_count, p=probs, replace=False).tolist()  # 按概率无放回抽样
        
        # 合并利用和探索阶段的选择
        selected_clients = selected_exploit + selected_explore  # 合并两阶段选择的客户端
        
        # 确保选择足够数量的客户端
        remaining = num_to_select - len(selected_clients)  # 计算还需选择的数量
        if remaining > 0:  # 如果还需选择更多
            remaining_clients = [c for c in available_clients if c not in selected_clients]  # 剩余可选客户端
            if remaining_clients:  # 如果有剩余客户端
                # 随机选择剩余客户端
                additional = np.random.choice(remaining_clients, 
                                            min(remaining, len(remaining_clients)), 
                                            replace=False).tolist()  # 随机无放回抽样
                selected_clients.extend(additional)  # 添加到已选列表
        
        # 更新客户端统计信息中的选择次数
        for client_id in selected_clients:  # 遍历所有选择的客户端
            if client_id in self.global_var['client_statistics']:  # 如果有统计信息
                self.global_var['client_statistics'][client_id]['count'] += 1  # 增加参与次数
        
        return selected_clients  # 返回选择的客户端列表
    
    def _update_client_statistics(self, client_id, training_results):
        """更新客户端统计信息到全局变量"""
        # 如果客户端不在统计信息中，初始化其条目
        if client_id not in self.global_var['client_statistics']:
            self.global_var['client_statistics'][client_id] = {
                'count': 0,
                'history': []
            }
        
        # 更新统计信息
        self.global_var['client_statistics'][client_id].update({
            'loss': training_results['loss'],
            'gradient_norm': training_results['gradient_norm'],
            'sample_size': training_results['sample_size'],
            'execution_time': training_results['execution_time'],
            'time_stamp': self.current_round,
            'count': self.global_var['client_statistics'][client_id]['count'] + 1
        })
        
        # 保存历史记录
        self.global_var['client_statistics'][client_id]['history'].append({
            'round': self.current_round,
            'loss': training_results['loss'],
            'execution_time': training_results['execution_time']
        })
        
        # 打印日志确认更新
        print(f"已将客户端 {client_id} 的统计信息更新到全局变量，当前有 {len(self.global_var['client_statistics'])} 个客户端") 