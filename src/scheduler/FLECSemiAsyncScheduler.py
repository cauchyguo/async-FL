from scheduler.SyncScheduler import SyncScheduler


class FLECSemiAsyncScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncScheduler.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)
        self.group_ready_num = None
        # self.message_queue = MessageQueueFactory.create_message_queue()
        self.group_manager = self.global_var['group_manager']
        self.edge_server_num = self.global_var['edge_server_num']
        self.edge_unique_groups = self.global_var['edge_unique_groups']
        self.edge_shared_groups = self.global_var['edge_shared_groups']
        # self.group_num = self.group_manager.update()

    def schedule(self):
        current_time = self.current_t.get_time()
        schedule_time = self.schedule_t.get_time()
        if current_time > self.T:
            return
        if self.check_group_update():
            self.edge_server_num = self.group_manager.update()[-1]
        print("| current_epoch |", current_time)
        # the first epoch is to start all groups
        if current_time == 1:
            print("starting all edge servers training")
            for edge_server_idx in range(self.group_manager.get_edge_server_num()):
                # 用于client初始化
                for client_id in self.edge_unique_groups[edge_server_idx]:
                    self.message_queue.put_into_downlink(client_id, "group_id", edge_server_idx)
                for client_id in self.edge_shared_groups[edge_server_idx]:
                    self.message_queue.put_into_downlink(client_id, "group_id", edge_server_idx)
                print(f"\nbegin select edge server {edge_server_idx}")
                selected_clients = self.client_select_edge_server(edge_server_idx)
                # Store the number of clients scheduled.
                self.group_manager.group_client_num_list[edge_server_idx] = len(selected_clients)
                # Global storage of model lists for each group.
                self.group_manager.network_list[edge_server_idx] = (self.server_weights)
                self.notify_client(selected_clients, current_time, schedule_time)
        else:
            print(f"\nbegin select edge server {self.group_ready_num}")
            selected_clients = self.client_select_edge_server(self.group_ready_num)
            self.group_manager.group_client_num_list[self.group_ready_num] = len(selected_clients)
            self.notify_client(selected_clients, current_time, schedule_time)
        # wait for all update from clients of the same group
        self.queue_manager.receive(self.group_manager.group_client_num_list)
        self.group_ready_num = self.queue_manager.group_ready_num

    def client_select(self, group_num, *args, **kwargs):
        client_list = self.group_manager.get_group_list()[group_num]
        selected_clients = self.schedule_caller.schedule(client_list)
        return selected_clients
    
    def client_select_edge_server(self, edge_server_idx, *args, **kwargs):
        # 贪心决策
        edge_server_unique_clients = self.edge_unique_groups[edge_server_idx]
        edge_server_shared_clients = self.edge_shared_groups[edge_server_idx]
        # 判断该client是否够处于训练状态
        training_status = self.message_queue.get_training_status()
        for client_id in edge_server_shared_clients:
            if client_id  in training_status and training_status[client_id]:
                edge_server_shared_clients.remove(client_id)
        selected_clients = self.schedule_caller.schedule(edge_server_idx,edge_server_unique_clients,edge_server_shared_clients)

        # 更新client分组
        for client_id in selected_clients:
            self.global_var['client_group_mapping'][client_id] = edge_server_idx

        return selected_clients

    def check_group_update(self, *args, **kwargs):
        return False
