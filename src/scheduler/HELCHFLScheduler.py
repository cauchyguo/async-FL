import time
import numpy as np
import pandas as pd
from core.handlers.Handler import HandlerChain
from core.handlers.ServerHandler import  ContentDispatcher, UpdateWaiter
from scheduler.BaseScheduler import BaseScheduler
from core.handlers.Handler import Handler, HandlerChain

class HELCHFLScheduler(BaseScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        BaseScheduler.__init__(self, server_thread_lock, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        self.finals = []
        self.edge_server_num = 5
        self.client_num = 100
        self.group_num = self.edge_server_num





        self.sys_cost = self.global_var['config']['server']['scheduler']['schedule']['params']['sys_cost']
        self.eta = self.global_var['config']['server']['scheduler']['schedule']['params']['eta']
        self.edge_select_num = self.global_var['config']['server']['scheduler']['schedule']['params']['edge_select_num']

        self.global_var['client_edge_association_df'] = pd.read_csv(self.global_var['config']['server']['scheduler']['schedule']['params']['client_edge_association_path'])


        self.global_var['client_edge_association_df']['select_times'] = 0
        self.global_var['client_edge_association_df']['utility'] = 0

        self.global_var['client_edge_association_df']['system_time'] = self.global_var['client_edge_association_df']['system_time'].apply(lambda x: x + self.sys_cost)

        # for client_id in range(self.client_num):
        #     self.download_item(client_id, "delay", self.global_var['client_edge_association_df']['system_time'].iloc[client_id])


    def _run_iteration(self) -> None:
        while self.current_t.get_time() <= self.T:
            # Scheduling is performed periodically.
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            self.execute_chain()
            self.schedule_t.time_add()
            # Notifying the updater to aggregate weights.
            self.mutex_sem.release()
            self.full_sem.release()
            time.sleep(0.01)



    def create_handler_chain(self):
        self.handler_chain = HandlerChain()
        (self.handler_chain.set_chain(ClientSelector())
         .set_next(ContentDispatcher())
         .set_next(UpdateWaiter()))

    def execute_chain(self):
        epoch = self.current_t.get_time()
        request = {"epoch": epoch, "updater": self, "global_var": self.global_var,
                   'scheduler': self.global_var['scheduler']}
        self.handler_chain.handle(request)

class ClientSelector(Handler):
    def _handle(self, request):
        global_var = request.get('global_var')
        scheduler = global_var['scheduler']
        epoch = request.get('epoch')

        client_edge_association_df = global_var['client_edge_association_df']

        random_two_server = np.random.randint(0, scheduler.edge_server_num, 2)
        print(f"Random two server: {random_two_server}")
        selected_clients = []
        for group_id in random_two_server:
            client_of_server = client_edge_association_df[client_edge_association_df['group_id'] == group_id].copy()
            for client_id in client_of_server.client_id.tolist():
                client_of_server.loc[client_of_server['client_id'] == client_id, 'utility'] = scheduler.eta ** client_of_server.loc[client_of_server['client_id'] == client_id, 'select_times'] * (1 / client_of_server.loc[client_of_server['client_id'] == client_id, 'system_time'])

            edge_select_clients = client_of_server.sort_values(by='utility', ascending=False).head(scheduler.edge_select_num).client_id.tolist()
            print(f"Edge {group_id} selected clients: {edge_select_clients}")

            selected_clients.extend(edge_select_clients)

        selected_clients = list(set(selected_clients))        
        # 选择时延最小的服务器
        # Update select_times using loc instead of iloc
        client_edge_association_df.loc[selected_clients, 'select_times'] += 1

        request['selected_clients'] = selected_clients
        global_var['selected_clients'] = selected_clients
        print(f"| current_epoch {epoch}, schedule_epoch {scheduler.schedule_t.get_time()} |")
        print("selected_clients:", selected_clients)
        round_time  = max(client_edge_association_df.loc[selected_clients, 'system_time'])
        print(f"Round time: {round_time}")
        return request
