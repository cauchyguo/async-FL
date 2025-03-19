import copy

import torch

from client.NormalClient import NormalClient
from utils.GlobalVarGetter import GlobalVarGetter
import math
import time
from client.Client import Client
from loss.LossFactory import LossFactory
from utils import ModuleFindTool
from utils.DatasetUtils import FLDataset
from utils.Tools import to_cpu


class CustomClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.tau = 0
        self.global_var = GlobalVarGetter.get()
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

        # server_time_cost = self.global_var['system_time_cost'][-1] - self.global_var['system_time_cost'][-2]

        end_time = time.time()

        sys_delay = end_time - self.global_var['system_time_cost'][-1]

        # Information transmitted from the client to the server has latency.
        self.delay_simulate(max(0, (self.delay - sys_delay)))

        # upload its updates
        self.upload(data_sum, weights)
        
    def train(self):
        data_sum, weights = self.train_one_epoch()
        return data_sum, to_cpu(weights)
    
    def train_one_epoch(self):
        """
        The training function of Client, used for model training.
        """
        if self.mu != 0:
            global_model = copy.deepcopy(self.model)
        data_sum = 0
        total_loss = 0            # 总损失值
        total_samples = 0         # 总样本数
        total_squared_loss = 0     # 损失平方和
        for epoch in range(self.epoch + 1):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.model(data)
                # Calculate the loss function
                loss = self.loss_func(preds, label)
                # 获取当前批次大小
                batch_size = label.size(0)
                # 累加数据量
                data_sum += batch_size
                # 累加损失值
                total_loss += loss.item() * batch_size
                # 累加损失平方值
                total_squared_loss += (loss.item() ** 2) * batch_size
                # 累加样本数
                total_samples += batch_size
                # proximal term
                if self.mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (self.mu / 2) * proximal_term
                # backpropagate
                loss.backward()
                # Update the gradient
                self.opti.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # Zero out the gradient and initialize the gradient.
                self.opti.zero_grad()
        # Return the updated model parameters obtained by training on the client's own data.

        # 训练完后计算每个样本的损失
        # squared_loss_sum = total_squared_loss / max(total_samples, 1)  # 使用平均损失的平方乘以样本数
        self.client_data_util = math.sqrt(total_squared_loss / max(total_samples, 1)) *  max(total_samples, 1)
        self.avg_loss = total_loss / max(total_samples, 1)
        # self.global_var['client_avg_loss']['client_id'].append(avg_loss)
        weights = self.model.state_dict()
        torch.cuda.empty_cache()
        return data_sum, weights
    
    def upload(self, data_sum, weights):
        """
        The detailed parameters uploaded to the server by Client.
        """
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp, "avg_loss": self.avg_loss, "client_data_util": self.client_data_util}
        self.message_queue.put_into_uplink(update_dict)
        print("Client", self.client_id, "uploaded")

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
