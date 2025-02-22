import time

from numpy import argmax
from scheduler.BaseScheduler import BaseScheduler
import random
from utils.GlobalVarGetter import GlobalVarGetter
import math

class GroupSyncScheduler(BaseScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        BaseScheduler.__init__(self, server_thread_lock, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem

        self.global_var = GlobalVarGetter.get()

        # 使用group作为调度对象，并融入多臂老虎机机制
        self.group_manager = self.global_var['group_manager']
        self.group_num = self.group_manager.get_group_num()
        self.history_select_group = [] # 记录历史每个global epoch中选择的group
        self.global_var['group_history_improved_loss'] = [[] for i in range(self.group_num)] # 每个group下维持一个二元组列表（global_epoch，delta_loss）
        self.global_var['history_loss'] = []
        self.global_var['group_selected_at_global_epoch'] = []
        
        for group_id in range(self.group_manager.get_group_num()):
            if(len(self.global_var['group_history_improved_loss'][group_id]) == 0):
                self.global_var['group_history_improved_loss'][group_id].append((0, 0))

    def run(self):
        while self.current_t.get_time() <= self.T:
            # Scheduling is performed periodically.
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            self.schedule()
            # Notifying the updater to aggregate weights.
            self.mutex_sem.release()
            self.full_sem.release()
            time.sleep(0.01)

    def schedule(self):
        r"""
            schedule the clients
        """
        current_time = self.current_t.get_time()
        schedule_time = self.schedule_t.get_time()
        if current_time > self.T:
            return
        # if self.current_t.get_time() <= self.group_num:
        #     selected_group_id = int(self.current_t.get_time()-1)
        # else:
        if "method" in self.config and self.config["method"] == "multibandit":
            selected_group_id = self.bandit_schedule()
        else:# 和随机算法做对比
            selected_group_id = self.random_schedule()
        # selected_group_id = self.bandit_schedule()
        self.global_var['group_selected_at_global_epoch'].append(selected_group_id)
        selected_client = self.client_select(selected_group_id)
        self.notify_client(selected_group_id, selected_client, current_time, schedule_time)
        # Waiting for all clients to upload their updates.
        self.queue_manager.receive(len(selected_client))


    def random_schedule(self):
        """随机调度一个分组参与训练"""
        return random.randint(0, self.group_manager.get_group_num() - 1)
    
    def bandit_schedule(self):
        """多臂老虎机调度一个分组参与训练"""
        # CustomGroupManager
        # 衰减指数

        for group_id in range(self.group_manager.get_group_num()):
            if len(self.global_var['group_history_improved_loss'][group_id]) <= 1:
                print(f"Init Selection: Group {group_id} is selected")
                return group_id
        alpha = 0.95
        # 选择评分最高的分组

        score_for_group_current_epoch = []
        for group_i in range(self.group_manager.get_group_num()):
            # 计算当前分组的评分
            weights = [alpha ** (self.current_t.get_time() - epoch_delta_loss[0]) for epoch_delta_loss in self.global_var['group_history_improved_loss'][group_i]]
            group_selected_times = len(self.global_var['group_history_improved_loss'][group_i])
            score = sum([weights[i] * self.global_var['group_history_improved_loss'][group_i][i][1] for i in range(len(weights))]) /(sum(weights) + 1e-6) + math.sqrt(2 * math.log(self.current_t.get_time()) / group_selected_times)
            score_for_group_current_epoch.append(score)
        group_id = argmax(score_for_group_current_epoch)
        # group_id = argmax(self.group_avg_reward)
        print(f"Custom Multi bandit Algo: Group {group_id} is selected")
        return group_id


        return random.randint(0, self.group_manager.get_group_num() - 1)
    
    def client_select(self, group_id, *args, **kwargs):
        client_list = self.group_manager.get_group_list()[group_id]
        selected_clients = self.schedule_caller.schedule(client_list)
        return selected_clients

    def notify_client(self, group_id, selected_client, current_time, schedule_time):
        print(f"| current_epoch {current_time} |. Begin client select")
        print("\nSchedulerThread schedule Group [",group_id, "], select(",len(selected_client), "clients):")
        for client_id in selected_client:
            print(client_id, end=" | ")
            # Sending the server's model parameters and timestamps to the clients
            self.send_weights(client_id, current_time, schedule_time)
            # Starting a client thread
            self.selected_event_list[client_id].set()
        print("\n-----------------------------------------------------------------Schedule complete")
        self.schedule_t.time_add()
