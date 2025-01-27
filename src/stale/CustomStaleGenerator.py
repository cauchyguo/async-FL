import pandas as pd
import client
import numpy as np

# "stale": {
#     "path": "stale.stale_generator",
#     "params": {
#         "clients_info_df": "path ",
#         "param2": "value2"
#     }
# }

class CustomStaleGenerator:
    def __init__(self, client_num, df_path, stale_config=None):
        self.clients_dataframe = pd.read_csv(df_path)
        if len(self.clients_dataframe) < client_num:
            raise Exception("The number of clients in the dataframe is not equal to the client number.")
        self.client_num = client_num
        

    def generate_staleness_list(self):
        # Generate the stale list based on the clients info
        clients_stale_list = self.clients_dataframe["time"].tolist()
        return clients_stale_list[:self.client_num]