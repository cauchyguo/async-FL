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
class HiFlashScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncScheduler.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.group_ready_num = None
        self.group_manager = self.global_var['group_manager']
        self.edge_server_num = self.global_var['edge_server_num']
        self.group_num = self.edge_server_num
        time.sleep(0.1)

        # 为每个客户端设置delay和group_id
        client_edge_df = pd.read_csv(self.global_var['config']['server']['scheduler']['schedule']['params']['clients_edge_info_path'])
        self.clients_edge_info_df = client_edge_df
        for group_id in range(self.group_manager.get_group_num()):
            for client_id in self.group_manager.get_group_list()[group_id]:
                client_compute_delay = client_edge_df['compute_time'].iloc[client_id]
                server_col = f'server_{group_id}'
                trans_rate = self.calculate_data_rate(client_edge_df[server_col].iloc[client_id], client_edge_df['power'].iloc[client_id], 5)
                trans_time = self.global_var['config']['server']['scheduler']['schedule']['params']['data_size'] * 8 / trans_rate
                system_time = client_compute_delay + trans_time
                self.download_item(client_id, "delay", system_time)
                self.download_item(client_id, "group_id", group_id)

    def create_handler_chain(self):
        self.handler_chain = HandlerChain()
        (self.handler_chain.set_chain(GroupUpdater())
         .set_next(GroupClientSelector())
         .set_next(ContentDispatcher())
         .set_next(GroupUpdateWaiter()))

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


class GroupUpdater(Handler):
    def _handle(self, request):
        return request


class GroupClientSelector(Handler):
    def __init__(self):
        super().__init__()
        self.handler = InnerGroupClientSelector()
        self.first_run = False # 是否是第一次运行，False表示是第一次运行

    def _handle(self, request):
        scheduler = request.get('scheduler')
        self.client_label_df = pd.read_csv("/disk6T/ypguo/async-FL/temp/client_label_df.csv")
        group_manager = scheduler.group_manager
        total_selected_clients = []
        if self.first_run is False:
            print("starting all edge servers training")
            for i in range(group_manager.get_group_num()):
                for j in group_manager.get_group_list()[i]:
                    scheduler.download_item(j, "group_id", i)
                client_list = group_manager.get_group_list()[i]
                selected_clients = self.handler.handle(
                    {'client_list': client_list, 'group_id': i, 'scheduler': scheduler})
                group_manager.group_client_num_list[i] = len(selected_clients)
                group_manager.network_list[i] = scheduler.server_weights
                total_selected_clients.extend(selected_clients)
            self.first_run = True
        else:
            group_id = scheduler.group_ready_num
            client_list = group_manager.get_group_list()[group_id]
            selected_clients = self.handler.handle(
                {'group_id': group_id, 'client_list': client_list, 'scheduler': scheduler})
            group_manager.group_client_num_list[group_id] = len(selected_clients)
            total_selected_clients.extend(selected_clients)
        request['selected_clients'] = total_selected_clients
        return request


class InnerGroupClientSelector(Handler):
    def _handle(self, request):
        client_list = request.get('client_list')
        group_id = request.get('group_id')
        scheduler = request.get('scheduler')
        training_status = scheduler.message_queue.get_training_status()
        client_list = [client_id for client_id in client_list if
                       client_id not in training_status or not training_status[client_id]]
        # # 每个客户端的数据分布
        client_label_df = scheduler.global_var['client_label_df']
        # selected_clients = random.sample(client_list, 5)
        selected_clients = scheduler.schedule_caller.schedule(client_list,client_label_df)
        print(f'group {group_id} selected {len(selected_clients)} clients: {selected_clients}')
        return selected_clients


class GroupUpdateWaiter(Handler):
    def _handle(self, request):
        scheduler = request.get('scheduler')
        scheduler.queue_manager.receive(scheduler.group_manager.group_client_num_list)
        scheduler.group_ready_num = scheduler.queue_manager.group_ready_num
        print(f'group_ready_num: {scheduler.group_ready_num}', scheduler.group_manager.group_client_num_list)
        return request

