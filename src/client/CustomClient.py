import copy

import torch

from client.NormalClient import NormalClient
import time

class CustomClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.tau = 0

        # self.client_selected_num = 0
        # self.client_avg_reward = 0

    def local_task(self):
        """
        The local task of Client, namely, the detailed process of training a model.
        """
        # The client performs training.
        start_time = time.time()
        data_sum, weights = self.train()

        print("Client", self.client_id, "trained")

        end_time = time.time()

        sys_delay = end_time - start_time

        # Information transmitted from the client to the server has latency.
        self.delay_simulate(max(0, (self.delay - sys_delay)))

        # upload its updates
        self.upload(data_sum, weights)
        


"""     def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp, "tau": self.tau}
        self.message_queue.put_into_uplink(update_dict) """


""" class CustomClientWithGrad(CustomClient):
    def train_one_epoch(self):
        self.tau = 0
        if self.mu != 0:
            global_model = copy.deepcopy(self.model)
        data_sum = len(self.train_dl)
        accumulated_grads = []  # Initialize the list of accumulated gradients

        # Traverse the training data.
        for data, label in self.train_dl:
            self.tau += 1
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.model(data)
            # Calculate the loss function
            loss = self.loss_func(preds, label)
            # Proximal term
            if self.mu != 0:
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = loss + (self.mu / 2) * proximal_term
            # Backpropagate, but do not execute optimization steps.
            loss.backward()
            # Accumulate gradients.
            accumulated_grads = [None if acc_grad is None else acc_grad + param.grad
                                 for acc_grad, param in zip(accumulated_grads, self.model.parameters())]

            # Zero out the gradient to prepare for the next iteration.
            self.model.zero_grad()
        # return accumulate gradients.
        torch.cuda.empty_cache()
        return data_sum, accumulated_grads """
