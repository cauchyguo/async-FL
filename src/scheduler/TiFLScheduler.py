import time
import pandas as pd
import numpy as np
from scheduler.BaseScheduler import BaseScheduler
from utils.GlobalVarGetter import GlobalVarGetter
import random

class TiFLScheduler(BaseScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        BaseScheduler.__init__(self, server_thread_lock, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem

        self.global_var = GlobalVarGetter.get()

        # 使用分层作为调度对象
        self.group_manager = self.global_var['group_manager']
        self.group_num = self.group_manager.get_group_num()
        
        # 初始化历史记录变量
        self.global_var['group_history_improved_loss'] = [[] for i in range(self.group_num)]
        self.global_var['group_history_loss'] = [[] for i in range(self.group_num)]
        self.global_var['history_loss'] = []
        self.global_var['group_selected_at_global_epoch'] = []
        self.global_var['group_history_acc'] = [[] for i in range(self.group_num)]
        self.global_var['group_history_improved_acc'] = [[] for i in range(self.group_num)]
        self.global_var['epoch_avg_loss'] = {}  # 记录每个全局轮次的平均损失
        self.global_var['epoch_group_data_util'] = {}  # 记录每个轮次的数据利用率
        self.global_var['epoch_group_data_util_accumulate'] = {}  # 累积数据利用率
        self.global_var['group_data_util'] = [[] for i in range(self.group_num)]
        self.global_var['group_data_util_accumulate'] = [[] for i in range(self.group_num)]
        self.global_var['group_epoch_avg_loss'] = [[] for i in range(self.group_num)]  # 添加这个关键变量
        self.global_var['epoch_accuacy'] = []  # 记录每个global epoch下的准确率
        
        # TiFL算法的特定变量
        self.credits = [5] * self.group_num  # 每个层的信用值，初始为5
        self.current_tier = 0  # 当前选择的层，从0开始
        self.update_interval = self.config['TiFL_params'].get('update_interval', 5)  # 每隔多少轮更新选择概率
        self.tier_probs = [1.0/self.group_num] * self.group_num  # 初始化为均匀概率分布
        
        # 初始化历史精确度记录
        self.tier_accuracies = [[] for _ in range(self.group_num)]
        
        # 初始化其他全局变量
        for group_id in range(self.group_manager.get_group_num()):
            self.global_var['group_history_loss'][group_id].append((0, 2))
            self.global_var['group_history_acc'][group_id].append((0, 0))
        
        for group_id in range(self.group_manager.get_group_num()):
            if len(self.global_var['group_history_improved_loss'][group_id]) == 0:
                self.global_var['group_history_improved_loss'][group_id].append((0, 0))
                self.global_var['group_history_improved_acc'][group_id].append((0, 0))
        
        # 确保SyncUpdater中使用的所有列表都被初始化
        for group_id in range(self.group_manager.get_group_num()):
            self.global_var['group_data_util'][group_id].append((0, 0))
            self.global_var['group_data_util_accumulate'][group_id].append((0, 0))
            self.global_var['group_epoch_avg_loss'][group_id].append((0, 0))

        # 从配置加载TiFL参数
        self.tifl_group_path = self.config['TiFL_params'].get('TiFL_group_path', '')
        
        # 加载分组信息
        self.group_info_df = self.load_group_info()

    def load_group_info(self):
        """加载层级信息"""
        try:
            return pd.read_csv(self.tifl_group_path)
        except:
            # 如果文件不存在，创建一个默认的DataFrame
            default_info = []
            for i in range(self.group_num):
                default_info.append({
                    'group_id': i,
                    'client_id_list': [],
                    'group_data_num': 0,
                    'group_delay_time': 1.0
                })
            return pd.DataFrame(default_info)

    def run(self):
        while self.current_t.get_time() <= self.T:
            # 调度定期执行
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            self.schedule()
            # 通知更新器聚合权重
            self.mutex_sem.release()
            self.full_sem.release()
            time.sleep(0.01)

    def schedule(self):
        """执行调度"""
        current_time = self.current_t.get_time()
        schedule_time = self.schedule_t.get_time()
        if current_time > self.T:
            return
            
        # 确保epoch_avg_loss和epoch_group_data_util初始化
        if current_time not in self.global_var['epoch_avg_loss']:
            self.global_var['epoch_avg_loss'][current_time] = 0.0
        if current_time not in self.global_var['epoch_group_data_util']:
            self.global_var['epoch_group_data_util'][current_time] = 0.0
        if current_time not in self.global_var['epoch_group_data_util_accumulate']:
            self.global_var['epoch_group_data_util_accumulate'][current_time] = 0.0
        
        # 使用TiFL算法选择层
        selected_tier_id = self.tifl_schedule()
        self.global_var['group_selected_at_global_epoch'].append(selected_tier_id)
        
        # 选择该层的客户端
        selected_client = self.client_select(selected_tier_id)
        
        # 通知被选中的客户端
        self.notify_client(selected_tier_id, selected_client, current_time, schedule_time)
        
        # 等待所有客户端上传更新
        self.queue_manager.receive(len(selected_client))

    def tifl_schedule(self):
        """TiFL自适应层选择算法实现"""
        current_round = self.current_t.get_time()
        
        # 在前几轮，每个层都训练一次以获取初始性能数据
        if current_round <= self.group_num:
            return current_round - 1
        
        # 检查是否需要更新选择概率
        if current_round % self.update_interval == 0 and current_round >= self.update_interval:
            # 检查当前层的精确度是否低于上一轮
            if len(self.global_var['group_history_acc'][self.current_tier]) >= 2:
                current_acc = self.global_var['group_history_acc'][self.current_tier][-1][1]
                prev_acc = self.global_var['group_history_acc'][self.current_tier][-2][1]
                
                if current_acc <= prev_acc:
                    # 性能下降，更新选择概率
                    self.tier_probs = self.change_probs()
        
        # 确保概率和为1且非负
        if sum(self.tier_probs) == 0 or any(p < 0 for p in self.tier_probs):
            self.tier_probs = [1.0 / self.group_num] * self.group_num
        
        # 如果所有层的信用值都为0，重置信用值
        if all(credit == 0 for credit in self.credits):
            self.credits = [5] * self.group_num
            print("所有层信用值已用尽，重置所有层信用值")
        
        # 根据概率选择一个层
        max_attempts = 100  # 防止无限循环
        attempts = 0
        
        while attempts < max_attempts:
            try:
                attempts += 1
                # 从概率分布中选择一个层
                selected_tier = np.random.choice(range(self.group_num), p=self.tier_probs)
                
                # 如果该层还有信用值，则选择它
                if self.credits[selected_tier] > 0:
                    self.current_tier = selected_tier
                    # 减少该层的信用值
                    self.credits[self.current_tier] -= 1
                    break
                    
                # 如果尝试了多次仍找不到有信用值的层，选择第一个有信用的层
                if attempts >= max_attempts // 2:
                    for tier in range(self.group_num):
                        if self.credits[tier] > 0:
                            self.current_tier = tier
                            self.credits[tier] -= 1
                            attempts = max_attempts  # 强制退出循环
                            break
            except ValueError as e:
                # 如果概率出问题，使用均匀分布
                print(f"警告：选择概率出现问题：{e}，使用均匀分布")
                self.tier_probs = [1.0 / self.group_num] * self.group_num
        
        # 如果尝试多次后仍无法选择，随机选择一个层并重置其信用值
        if attempts >= max_attempts:
            self.current_tier = random.randint(0, self.group_num - 1)
            self.credits = [5] * self.group_num
            self.credits[self.current_tier] -= 1
            print(f"警告：无法找到有信用值的层，随机选择层{self.current_tier}并重置所有信用值")
        
        print(f"TiFL算法: 选择层 {self.current_tier}，信用值剩余 {self.credits[self.current_tier]}，选择概率 {self.tier_probs[self.current_tier]:.4f}")
        return self.current_tier

    def change_probs(self):
        """更新各层的选择概率"""
        # 收集各层的平均精确度
        tier_accs = []
        for tier in range(self.group_num):
            if len(self.global_var['group_history_acc'][tier]) > self.update_interval:
                acc_avg = np.mean(self.global_var['group_history_acc'][tier][-self.update_interval:])
                tier_accs.append((tier, acc_avg))
            else:
                tier_accs.append((tier, 0))
        
        print(f"各层当前精度: {tier_accs}")
        print(f"各层当前信用值: {self.credits}")
        
        # 按精确度升序排序
        tier_accs.sort(key=lambda x: x[1])
        
        # 计算仍有信用值的层的数量
        valid_tiers = sum(1 for credit in self.credits if credit > 0)
        if valid_tiers == 0:
            # 如果所有层都没有信用值，重置信用值
            self.credits = [5] * self.group_num
            valid_tiers = self.group_num
            print("所有层信用值已用尽，重置所有信用值")
        
        # 计算分母D
        D = valid_tiers * (valid_tiers - 1) / 2 if valid_tiers > 1 else 1
        
        # 计算新的概率分布
        new_probs = [0] * self.group_num
        for i, (tier, acc) in enumerate(tier_accs):
            if self.credits[tier] > 0:
                # 精确度越低的层获得越高的选择概率
                new_probs[tier] = max(0, (valid_tiers - i) / D)  # 确保概率非负
                print(f"层{tier} 精度={acc:.4f} 排名={i+1}/{valid_tiers} 初始概率={new_probs[tier]:.4f}")
            else:
                new_probs[tier] = 0
        
        # 归一化概率，确保和为1且所有值非负
        sum_probs = sum(new_probs)
        if sum_probs > 0:
            new_probs = [max(0, p / sum_probs) for p in new_probs]  # 确保概率非负
        else:
            # 使用均匀分布
            print("警告：所有概率和为0，使用均匀分布")
            new_probs = [1.0 / self.group_num] * self.group_num
        
        # 再次检查确保所有概率都是非负的
        new_probs = [max(0, p) for p in new_probs]
        
        # 再次归一化
        sum_probs = sum(new_probs)
        if sum_probs > 0:
            new_probs = [p / sum_probs for p in new_probs]
        else:
            new_probs = [1.0 / self.group_num] * self.group_num
        
        print(f"更新后的层选择概率: {[f'{p:.4f}' for p in new_probs]}")
        
        return new_probs

    def client_select(self, group_id, *args, **kwargs):
        """选择指定层中的客户端"""
        client_list = self.group_manager.get_group_list()[group_id]
        selected_clients = self.schedule_caller.schedule(client_list)
        return selected_clients

    def notify_client(self, group_id, selected_client, current_time, schedule_time):
        """通知被选中的客户端"""
        print(f"| current_epoch {current_time} |. Begin client select")
        print(f"\nSchedulerThread schedule Tier [{group_id}], select({len(selected_client)} clients):")
        for client_id in selected_client:
            print(client_id, end=" | ")
            # 向客户端发送服务器的模型参数和时间戳
            self.send_weights(client_id, current_time, schedule_time)
            # 启动客户端线程
            self.selected_event_list[client_id].set()
        print("\n-----------------------------------------------------------------Schedule complete")
        self.schedule_t.time_add()