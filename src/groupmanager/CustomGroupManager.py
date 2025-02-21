from group.GroupCaller import GroupCaller
from groupmanager.BaseGroupManager import BaseGroupManager
from utils import ModuleFindTool

class CustomGroupManager(BaseGroupManager):
    def __init__(self, config):
        super().__init__(config)
        self.client_list = self.global_var['client_id_list']
        self.latency_list = self.global_var['client_staleness_list']
        self.network_list = [0] * len(self.client_list)
        self.group_client_num_list = [0] * len(self.client_list)
        self.group_method = ModuleFindTool.find_class_by_path(self.config["group_method"]["path"])(self, self.config[
            "group_method"]["params"])
        self.group_caller = GroupCaller(self)
        self.group_list, self.group_num = self.group_caller.group(self.client_list, self.latency_list)
        self.epoch_list = [0] * len(self.client_list)

        # CustomGroupManager
        self.group_selected_times = [0] * self.group_num
        self.group_avg_reward = [0] * self.group_num
        self.group_history_impoved_acc = [[] for i in range(self.group_num)] # 每个group下维持一个二元组列表（global_epoch，improved_acc）

    def __group(self, client_list, latency_list, *args, **kwargs):
        self.group_list, self.group_num = self.group_caller.group(client_list, latency_list, *args, **kwargs)
        return self.group_list, self.group_num

    def get_group_num(self):
        return self.group_num

    def get_group_list(self):
        return self.group_list

    def update(self, *args, **kwargs):
        return self.__group(self.client_list, self.latency_list, *args, **kwargs)
