from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from client.NormalClient import NormalClient
from update.AbstractUpdate import AbstractUpdate
from utils.Algorithm import bernoulli_sampling
from utils.GlobalVarGetter import GlobalVarGetter

class SampleUpdate(AbstractUpdate):
    def __init__(self, config):
        self.global_var = GlobalVarGetter.get()
        self.lr = config.get("lr", 0.01)

    def update_server_weights(self, epoch, update_list):
        global_model = self.global_var["global_model"].state_dict()
        total_nums = 0
        for update_dict in update_list:
            total_nums += update_dict["data_sum"]
        updated_parameters = {k: torch.zeros_like(v) for k, v in global_model.items()}
        for update_dict in update_list:
            if update_dict["data_sum"] == 0:
                continue
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key] / total_nums
        # print(updated_parameters['conv1.weight'][0][0][0], total_nums)
        for key, var in global_model.items():
            updated_parameters[key] = var - self.lr * updated_parameters[key]
        return updated_parameters, None
