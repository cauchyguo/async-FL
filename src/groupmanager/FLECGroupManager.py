from group.GroupCaller import GroupCaller
from groupmanager.BaseGroupManager import BaseGroupManager
from utils import ModuleFindTool

class FLECGroupManager(BaseGroupManager):
    def __init__(self, config):
        super().__init__(config)
        self.client_list = self.global_var['client_id_list']
        self.latency_list = self.global_var['client_staleness_list']
        self.network_list = [0] * self.config["group_method"]["params"]["edge_server_num"] # 5个edge server edge_server_num
        self.group_client_num_list = [0] * self.config["group_method"]["params"]["edge_server_num"] # 5个edge server
        self.group_method = ModuleFindTool.find_class_by_path(self.config["group_method"]["path"])(self, self.config[
            "group_method"]["params"])
        self.group_caller = GroupCaller(self)
        self.edge_unique_groups, self.edge_shared_groups,self.edge_server_num = self.group_caller.group(self.client_list,self.latency_list)
        self.global_var['edge_server_num'] = self.edge_server_num
        self.group_num = self.edge_server_num
        self.global_var['edge_unique_groups'] = self.edge_unique_groups
        self.global_var['edge_shared_groups'] = self.edge_shared_groups
        self.epoch_list = [0] * len(self.client_list)

        # 映射最新的client分组
        self.global_var['client_group_mapping'] = [None] * len(self.client_list)

    def __group(self, *args, **kwargs):
        self.edge_unique_groups, self.edge_shared_groups,self.edge_server_num = self.group_caller.group( *args, **kwargs)
        return self.edge_unique_groups, self.edge_shared_groups,self.edge_server_num

    def get_edge_server_num(self):
        return self.edge_server_num

    def get_edge_group_list(self):
        return self.edge_unique_groups, self.edge_shared_groups

    def update(self, *args, **kwargs):
        return self.__group( *args, **kwargs)
