from core.handlers.Handler import Filter
from core.handlers.ServerHandler import ClientSelector
from scheduler.SyncScheduler import SyncScheduler
from utils.GlobalVarGetter import GlobalVarGetter
from core.handlers.Handler import Handler, HandlerChain
from core.handlers.ServerHandler import ContentDispatcher
from scheduler.SyncScheduler import SyncScheduler
from utils import ModuleFindTool
import copy
import time
import math
import pandas as pd
import random
import numpy as np

# this scheduler schedules clients according to the number of aggregations
class AsyncSchedulerPlus(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncScheduler.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.edge_server_num = self.global_var['edge_server_num']
        self.group_num = self.edge_server_num

        client_edge_df = pd.read_csv(self.global_var['config']['server']['scheduler']['schedule']['params']['clients_edge_info_path'])
        self.clients_edge_info_df = client_edge_df
        self.client_num = client_edge_df.shape[0]
        
        self.sys_cost = self.global_var['config']['server']['scheduler']['schedule']['params']['sys_cost']

        edge_bandwidth = self.global_var['config']['server']['scheduler']['schedule']['params']['total_bandwidth']
        # 平均分配带宽
        self.mean_bandwidth = self.edge_server_num * edge_bandwidth / self.client_num
        time.sleep(0.01)

        # 为每个客户端设置delay和group_id

        print("时延进行匹配边缘服务器") #按照平均带宽产生的时延进行分配
        for client_idx in range(self.client_num):
            system_time = []
            client_compute_delay = client_edge_df['compute_time'].iloc[client_idx]
            for i in range(self.edge_server_num):
                server_col = f'server_{i}'
                trans_rate = self.calculate_data_rate(client_edge_df[server_col].iloc[client_idx], client_edge_df['power'].iloc[client_idx], self.mean_bandwidth)
                trans_time = self.global_var['config']['server']['scheduler']['schedule']['params']['data_size'] * 8 / trans_rate
                system_time.append(client_compute_delay + trans_time + self.sys_cost)
            # 选择时延最小的服务器
            client_delay = np.min(system_time)
            # closest_server = np.argmin(system_time)
            self.download_item(client_idx, "delay", client_delay)

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_before(ClientSelectorFilter(), ClientSelector)

    def calculate_data_rate(self, distance, power, bandwidth, noise_poewr=-104,g_dB_coef=[-128.1,37.6]):
        """计算数据传输率 (Mbps)"""
        p_W = 10 ** (power / 10 - 3)  # dBm → 瓦特
        N0_W = 10 ** (noise_poewr / 10 - 3)  # 噪声功率
        
        # 信道增益
        g_dB = g_dB_coef[0] - g_dB_coef[1] * math.log10(distance / 1000)
        g_linear = 10 ** (g_dB / 10)
        
        # 信噪比和频谱效率
        SNR = (g_linear * p_W) / N0_W
        spectral_efficiency = math.log2(1 + SNR)
        
        # 数据传输率 (Mbps)
        rate = bandwidth * 1e6 * spectral_efficiency / 1e6
        return rate

# this scheduler schedules clients according to the nums of update which clients update
class AsyncSchedulerWithUpdate(SyncScheduler):
    def schedule(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_before(ClientSelectorFilterWithUpdate(), ClientSelector)


class ClientSelectorFilter(Filter):
    def __init__(self):
        super().__init__()
        self.last_s_time = -1
        config = GlobalVarGetter.get()['config']['server']['scheduler']
        self.schedule_interval = config.get('schedule_interval', 1)
        self.schedule_delay = config.get('schedule_delay', 1)

    def _handle(self, request):
        scheduler = request.get('scheduler')
        current_t = scheduler.current_t.get_time()
        if (current_t - 1) % self.schedule_interval == 0 and current_t != self.last_s_time and current_t <= scheduler.T:
            print("| current_epoch", current_t, "| last schedule time =",
                  self.last_s_time)
            # scheduling according to the number of aggregations.
            if scheduler.queue_manager.size() <= self.schedule_delay:
                print("| queue.size |", scheduler.queue_manager.size(), "<=", self.schedule_delay)
                self.last_s_time = current_t
                return True
            else:
                print("| queue.size |", scheduler.queue_manager.size(), ">", self.schedule_delay)
                print("\n-----------------------------------------------------------------No Schedule")
                return False


class ClientSelectorFilterWithUpdate(ClientSelectorFilter):
    def _handle(self, request):
        scheduler = request.get('scheduler')
        current_t = scheduler.current_t.get_time()
        # 每隔一段时间进行一次schedule
        if scheduler.queue_manager.get.count % self.schedule_interval == 0 and current_t != self.last_s_time:
            print("| current_time |", current_t % self.schedule_interval, "= 0", current_t, "!=",
                  self.last_s_time)
            print("| queue.size |", scheduler.queue_manager.size(), "<= ", self.schedule_delay)
            # scheduling according to the number of received updates
            if scheduler.queue_manager.size() <= self.schedule_delay:
                self.last_s_time = current_t
                return True
            else:
                print("\n-----------------------------------------------------------------No Schedule")
                return False
