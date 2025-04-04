from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter


class FLEC(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter.get()

    def update_server_weights(self, epoch, update_list,group_ready_num,time_stamp):

        
        update_dict = update_list[group_ready_num]
        client_weights = update_dict["weights"]
        # time_stamp = update_dict["time_stamp"]
        staleness_threshold = self.config["staleness_threshold"]

        alpha = self.config["alpha"]
        v = self.config["v"]
        staleness = self.global_var['updater'].current_t.get_time() - time_stamp
        if staleness <= staleness_threshold:
            weight = alpha * v ** staleness
        else:
            weight = 0

        updated_parameters = {}
        server_weights = self.global_var['updater'].model.state_dict()
        for key, var in server_weights.items():
            updated_parameters[key] = (weight * client_weights[key] + (1 - weight) * server_weights[key])
        return updated_parameters, None
