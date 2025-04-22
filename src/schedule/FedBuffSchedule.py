import random

from schedule.AbstractSchedule import AbstractSchedule
from utils.GlobalVarGetter import GlobalVarGetter

class FedBuffSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.select_num = config["select_num"]
        self.c_ratio = config["c_ratio"]
    def schedule(self, client_list):
        self.current_t = GlobalVarGetter.get()['current_t']
        # self.T = GlobalVarGetter.get()
        if self.current_t.get_time() == 1:
            select_num = 95
        else:
            select_num = self.select_num

        print("Current clients:", len(client_list), ", select:", select_num)
        selected_client_threads = random.sample(client_list, select_num)
        return selected_client_threads
