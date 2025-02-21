from group.AbstractGroup import AbstractGroup
import pandas as pd
class CustomGroup(AbstractGroup):
    def __init__(self, group_manager, config):
        self.group_manager = group_manager
        self.group_info_path = config["clients_info_path"]
        self.init = False

    def group(self, client_list, latency_list): # latency_list即stale_list
        self.init = True

        group_df = pd.read_csv(self.group_info_path)
        group_num = group_df['group_id'].unique().__len__()
        group_list = [[] for i in range(group_num)]
        for i in range(len(client_list)):
            group_id_for_client_i = group_df['group_id'].iloc[i]
            group_list[int(group_id_for_client_i)].append(client_list[i]) # 嵌套列表
        return group_list, len(group_list)

    def check_update(self):
        return self.init