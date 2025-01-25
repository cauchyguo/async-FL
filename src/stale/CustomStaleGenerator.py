import pandas as pd
import client
import numpy as np

def generate_client_stale_list(global_config):
    stale = global_config['stale']
    if isinstance(stale, list):
        client_staleness_list = stale
    elif isinstance(stale, bool):
        client_staleness_list = []
        for i in range(global_config["client_num"]):
            client_staleness_list.append(0)
    elif isinstance(stale, dict) and "path" in stale:
        stale_generator = ModuleFindTool.find_class_by_path(stale["path"])()(stale["params"])
        client_staleness_list = stale_generator.generate_staleness_list()
    else:
        total_sum = sum(stale['list'])
        if total_sum < global_config['client_num']:
            raise Exception("The sum of the client number in stale list must not less than the client number.")
        client_staleness_list = generate_stale_list(stale['step'], stale['shuffle'], stale['list'])
    return client_staleness_list

# "stale": {
#     "path": "stale.stale_generator",
#     "params": {
#         "clients_info_df": "path ",
#         "param2": "value2"
#     }
# }

class CustomStaleGenerator:
    def __init__(self, stale_config):
        clients_dataframe = pd.read_csv(stale_config["clients_info_df"])
        

    def generate_staleness_list(self):
        return "Custom Stale Data"