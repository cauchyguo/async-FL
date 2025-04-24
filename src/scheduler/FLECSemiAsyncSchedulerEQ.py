from core.handlers.Handler import Handler, HandlerChain
from core.handlers.ServerHandler import ContentDispatcher
from scheduler.SyncScheduler import SyncScheduler
from utils import ModuleFindTool
import copy
import time
from algorithm import DelayAdaptiveOptimizer
import math
import pandas as pd


class FLECSemiAsyncSchedulerEQ(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncScheduler.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.group_ready_num = None
        self.group_manager = self.global_var['group_manager']
        self.edge_server_num = self.global_var['edge_server_num']
        self.edge_unique_groups = self.group_manager.get_edge_group_list()[0]
        self.edge_shared_groups = self.group_manager.get_edge_group_list()[1]
        self.group_num = self.edge_server_num
        self.delay_adaptive_optimizer = DelayAdaptiveOptimizer(config["schedule"]["params"]["delay_adaptive_optimizer"]["params"])

        self.data_size = self.global_var['config']['server']['scheduler']['schedule']['params']['data_size']
        time.sleep(0.01)

        self.client_edge_df = pd.read_csv(self.global_var['config']['server']['scheduler']['schedule']['params']['clients_edge_info_path'])


        # for group_id in range(self.group_num):



    def create_handler_chain(self):
        self.handler_chain = HandlerChain()
        (self.handler_chain.set_chain(GroupUpdater())
         .set_next(GroupClientSelector())
         .set_next(ContentDispatcher())
         .set_next(GroupUpdateWaiter()))


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
        group_manager = scheduler.group_manager
        total_selected_clients = []
        if self.first_run is False:
            print("starting all edge servers training")
            for group_id in range(group_manager.get_group_num()):
                    
                # edge_unique_groups,edge_shared_groups = group_manager.get_group_list()
                edge_unique_group = scheduler.edge_unique_groups[group_id]
                edge_shared_group = [client_id for client_id in scheduler.edge_shared_groups[group_id] if client_id not in total_selected_clients]

                # client_list = group_manager.get_group_list()[group_id]
                selected_clients = self.handler.handle(
                    {'edge_unique_group': edge_unique_group, 'edge_shared_group': edge_shared_group, 'group_id': group_id, 'scheduler': scheduler})
                
                group_manager.group_client_num_list[group_id] = len(selected_clients)

                group_manager.network_list[group_id] = scheduler.server_weights
                total_selected_clients.extend(selected_clients)
                # delay_for_selected_clients = scheduler.delay_adaptive_optimizer.optimize(group_id,selected_clients)
                for client_id in selected_clients: # 将更新的客户端分组信息上传给服务器
                    client_compute_delay = scheduler.client_edge_df['compute_time'].iloc[client_id]
                    server_col = f'server_{group_id}'
                    trans_rate = self.calculate_data_rate(scheduler.client_edge_df[server_col].iloc[client_id], scheduler.client_edge_df['power'].iloc[client_id], 5)
                    trans_time = scheduler.data_size * 8 / trans_rate
                    system_time = client_compute_delay + trans_time

                    scheduler.download_item(client_id, "group_id", group_id)
                    scheduler.download_item(client_id, "delay", system_time)
            self.first_run = True
        else:
            group_id = scheduler.group_ready_num
            # edge_unique_groups,edge_shared_groups = group_manager.get_group_list()
            edge_unique_group = scheduler.edge_unique_groups[group_id]
            edge_shared_group = scheduler.edge_shared_groups[group_id]
            # if len(edge_unique_group) < 10:
            #     print("debug")
            # client_list = group_manager.get_group_list()[group_id]
            selected_clients = self.handler.handle(
                {'edge_unique_group': edge_unique_group, 'edge_shared_group': edge_shared_group,'group_id': group_id,  'scheduler': scheduler})
            group_manager.group_client_num_list[group_id] = len(selected_clients)
            total_selected_clients.extend(selected_clients)
            # 延迟优化
            for client_id in selected_clients:
                client_compute_delay = scheduler.client_edge_df['compute_time'].iloc[client_id]
                server_col = f'server_{group_id}'
                trans_rate = self.calculate_data_rate(scheduler.client_edge_df[server_col].iloc[client_id], scheduler.client_edge_df['power'].iloc[client_id], 5)
                trans_time = scheduler.data_size * 8 / trans_rate
                system_time = client_compute_delay + trans_time

                scheduler.download_item(client_id, "group_id", group_id)
                scheduler.download_item(client_id, "delay", system_time)
        request['selected_clients'] = total_selected_clients
        return request
    
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
    
class InnerGroupClientSelector(Handler):
    def _handle(self, request):
        edge_unique_group = copy.deepcopy(request.get('edge_unique_group'))
        edge_shared_group = copy.deepcopy(request.get('edge_shared_group'))
        group_id = request.get('group_id')
        scheduler = request.get('scheduler')
        client_label_df = scheduler.global_var['client_label_df']
        training_status = scheduler.message_queue.get_training_status()
        edge_shared_group = [client_id for client_id in edge_shared_group if
                       client_id not in training_status or not training_status[client_id]]
        selected_clients= scheduler.schedule_caller.schedule(edge_server_idx=group_id, edge_server_unique_clients=edge_unique_group,edge_server_shared_clients=edge_shared_group,client_label_df=client_label_df)
        print(f'group {group_id} selected_clients: {selected_clients}')
        return selected_clients


class GroupUpdateWaiter(Handler):
    def _handle(self, request):
        scheduler = request.get('scheduler')
        scheduler.queue_manager.receive(scheduler.group_manager.group_client_num_list)
        scheduler.group_ready_num = scheduler.queue_manager.group_ready_num
        print(f'group_ready_num: {scheduler.group_ready_num}', scheduler.group_manager.group_client_num_list)
        return request

