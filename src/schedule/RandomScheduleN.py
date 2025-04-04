import random

from schedule.AbstractSchedule import AbstractSchedule


class RandomScheduleN(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.select_num = config["select_num"]

    def schedule(self, client_list):
        select_num = min(self.select_num, len(client_list))

        print("Current clients:", len(client_list), ", select:", select_num)
        selected_client_threads = random.sample(client_list, select_num)
        return selected_client_threads
