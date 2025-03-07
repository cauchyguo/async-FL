from core.handlers.Handler import Handler, HandlerChain
from core.handlers.ServerHandler import ContentDispatcher
from scheduler.SyncScheduler import SyncScheduler

class FLECSemiAsyncScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncScheduler.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.group_ready_num = None
        self.group_manager = self.global_var['group_manager']
        self.edge_server_num = self.global_var['edge_server_num']
        self.edge_unique_groups = self.global_var['edge_unique_groups']
        self.edge_shared_groups = self.global_var['edge_shared_groups']
        self.group_num = self.edge_server_num

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
                    
                edge_unique_groups,edge_shared_groups = group_manager.get_group_list()
                edge_unique_group = edge_unique_groups[group_id]
                edge_shared_group = [client_id for client_id in edge_shared_groups[group_id] if client_id not in total_selected_clients]

                # client_list = group_manager.get_group_list()[group_id]
                selected_clients = self.handler.handle(
                    {'edge_unique_group': edge_unique_group, 'edge_shared_group': edge_shared_group, 'group_id': group_id, 'scheduler': scheduler})
                
                group_manager.group_client_num_list[group_id] = len(selected_clients)
                for client_id in selected_clients: # 将更新的客户端分组信息上传给服务器
                    scheduler.download_item(client_id, "group_id", group_id)
                group_manager.network_list[group_id] = scheduler.server_weights
                total_selected_clients.extend(selected_clients)
            self.first_run = True
        else:
            group_id = scheduler.group_ready_num
            edge_unique_groups,edge_shared_groups = group_manager.get_group_list()
            edge_unique_group = edge_unique_groups[group_id]
            edge_shared_group = edge_shared_groups[group_id]
            # client_list = group_manager.get_group_list()[group_id]
            selected_clients = self.handler.handle(
                {'edge_unique_group': edge_unique_group, 'edge_shared_group': edge_shared_group,'group_id': group_id,  'scheduler': scheduler})
            group_manager.group_client_num_list[group_id] = len(selected_clients)
            total_selected_clients.extend(selected_clients)
            for client_id in selected_clients: # 将更新的客户端分组信息上传给服务器
                scheduler.download_item(client_id, "group_id", group_id)
        request['selected_clients'] = total_selected_clients
        return request


class InnerGroupClientSelector(Handler):
    def _handle(self, request):
        edge_unique_group = request.get('edge_unique_group')
        edge_shared_group = request.get('edge_shared_group')
        group_id = request.get('group_id')
        scheduler = request.get('scheduler')
        training_status = scheduler.message_queue.get_training_status()
        edge_shared_group = [client_id for client_id in edge_shared_group if
                       client_id not in training_status or not training_status[client_id]]
        selected_clients = scheduler.schedule_caller.schedule(edge_server_idx=group_id, edge_server_unique_clients=edge_unique_group,edge_server_shared_clients=edge_shared_group)
        print(f'group {group_id} selected_clients: {selected_clients}')
        return selected_clients


class GroupUpdateWaiter(Handler):
    def _handle(self, request):
        scheduler = request.get('scheduler')
        scheduler.queue_manager.receive(scheduler.group_manager.group_client_num_list)
        scheduler.group_ready_num = scheduler.queue_manager.group_ready_num
        print(f'group_ready_num: {scheduler.group_ready_num}', scheduler.group_manager.group_client_num_list)
        return request

