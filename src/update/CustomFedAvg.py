import copy

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter


class CustomFedAvg(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter.get()

    def update_server_weights(self, epoch, update_list):
        total_nums = 0
        total_loss = 0
        group_data_util = 0
        group_data_util_accumulate = 0
        for update_dict in update_list:
            total_nums += update_dict["data_sum"]
            total_loss += update_dict["avg_loss"] * update_dict["data_sum"]
            group_data_util += update_dict["client_data_util"] * update_dict["data_sum"]
            group_data_util_accumulate += update_dict["client_data_util"]
        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            updated_parameters[key] = update_list[0]["weights"][key] * update_list[0]["data_sum"] / total_nums
        for i in range(len(update_list) - 1):
            update_dict = update_list[i + 1]
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key] * update_dict["data_sum"] / total_nums
        avg_loss = total_loss / total_nums
        group_data_util = group_data_util / total_nums # 计算每个group的平均数据利用率
        self.global_var['epoch_avg_loss'][epoch] = avg_loss
        self.global_var['epoch_group_data_util'][epoch] = group_data_util
        self.global_var['epoch_group_data_util_accumulate'][epoch] = group_data_util_accumulate
        return updated_parameters,None


class CustomFedAvgWithPrevious(CustomFedAvg):
    def __init__(self, config):
        super().__init__(config)
        self.beta = config["beta"]

    def update_server_weights(self, epoch, update_list):
        global_model = self.global_var["server_network"].state_dict()
        updated_parameters, _ = super().update_server_weights(epoch, update_list)
        for key, var in updated_parameters.items():
            updated_parameters[key] = self.beta * global_model[key] + (1 - self.beta) * updated_parameters[key]
        return updated_parameters, None
