import copy
import time
import torch

from client.NormalClient import NormalClient
from utils.Tools import to_cpu

class PyramidClient(NormalClient):
    """
    PyramidFL客户端实现，专门收集和上传训练指标
    包括损失值、梯度范数、样本数量和执行时间
    这些指标将被服务器端的PyramidFL更新器用于客户端选择
    """
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        """
        初始化PyramidClient
        
        参数:
            c_id: 客户端ID
            stop_event: 停止事件
            selected_event: 选择事件
            delay: 模拟延迟（也代表实际执行时间）
            index_list: 数据索引列表
            config: 配置字典
            dev: 设备
        """
        # 调用父类初始化方法
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        # 初始化训练统计数据字典，用于存储客户端的训练指标
        self.training_stats = {
            'loss': 0,              # 存储训练损失值
            'gradient_norm': 0,     # 存储梯度范数，用于衡量梯度大小
            'sample_size': 0,       # 存储样本数量
            'execution_time': 0     # 存储执行时间，用于系统效用评估
        }

    def local_task(self):
        """
        客户端本地训练任务，重写以收集执行时间
        这个方法会被基类的run方法调用，当客户端被选中执行训练时触发
        """
        # 记录开始时间（仅用于日志记录）
        start_time = time.time()
        
        # 执行训练，并获取训练结果和统计数据
        data_sum, weights, stats = self.train()
        
        # 计算实际测量的执行时间（仅用于日志记录）
        measured_time = time.time() - start_time
        
        # 使用self.delay作为执行时间（在仿真环境中，self.delay代表实际执行时间）
        stats['execution_time'] = self.delay
        
        # 打印训练完成信息
        print(f"Client {self.client_id} trained in {measured_time:.2f}s (simulated: {self.delay:.2f}s)")
        
        # 模拟网络延迟（仿真环境中的通信延迟）
        self.delay_simulate(self.delay)
        
        # 上传更新和统计数据到服务器
        self.upload(data_sum, weights, stats)
    
    def train(self):
        """
        执行一轮训练并收集统计数据
        这是对基类train方法的重写，添加了统计数据的收集
        
        返回:
            data_sum: 训练数据量
            weights: 更新后的模型权重
            stats: 训练统计数据
        """
        # 调用训练一个epoch的方法，获取结果和统计数据
        data_sum, weights, stats = self.train_one_epoch()
        # 将模型权重转移到CPU（便于通信）并返回结果
        return data_sum, to_cpu(weights), stats
    
    def train_one_epoch(self):
        """
        训练一个epoch并收集训练指标
        这是对基类train_one_epoch方法的重写，添加了详细的训练指标收集
        
        返回:
            data_sum: 训练数据量
            weights: 更新后的模型权重
            stats: 训练统计数据
        """
        # 如果启用了正则化，保存全局模型副本
        if self.mu != 0:
            global_model = copy.deepcopy(self.model)
            
        # 初始化统计数据累积变量
        data_sum = 0              # 总数据量
        total_loss = 0            # 总损失值
        total_samples = 0         # 总样本数
        gradient_squared_norm = 0 # 梯度平方范数累积值
        
        # 循环训练指定的epoch数
        for epoch in range(self.epoch):
            # 遍历训练数据加载器中的每个批次
            for data, label in self.train_dl:
                # 将数据和标签移动到指定设备（GPU/CPU）
                data, label = data.to(self.dev), label.to(self.dev)
                # 前向传播，获取模型预测
                preds = self.model(data)
                
                # 计算损失函数
                loss = self.loss_func(preds, label)
                # 获取当前批次大小
                batch_size = label.size(0)
                # 累加数据量
                data_sum += batch_size
                
                # 累加加权损失值（乘以批次大小，以便计算加权平均）
                total_loss += loss.item() * batch_size
                # 累加样本数
                total_samples += batch_size
                
                # 如果启用了正则化（μ不为0），添加近端项
                if self.mu != 0:
                    proximal_term = 0.0
                    # 计算当前模型与全局模型参数的距离
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    # 将近端项添加到损失中
                    loss = loss + (self.mu / 2) * proximal_term
                
                # 反向传播计算梯度
                loss.backward()
                
                # 计算当前批次的梯度范数（所有参数梯度的L2范数平方和）
                grad_norm = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        # 累加每个参数梯度的平方范数
                        grad_norm += param.grad.norm(2).item() ** 2
                        
                # 累加批次梯度平方范数到总梯度平方范数
                gradient_squared_norm += grad_norm
                
                # 使用优化器更新模型参数
                self.opti.step()
                # 如果配置了学习率调度器，更新学习率
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                # 清零梯度，为下一批次准备
                self.opti.zero_grad()
        
        # 计算平均损失（总损失除以总样本数）
        avg_loss = total_loss / max(total_samples, 1)
        
        # 计算总梯度范数（平方根）
        gradient_norm = gradient_squared_norm ** 0.5
        
        # 构造统计数据字典
        stats = {
            'loss': avg_loss,            # 平均损失值
            'gradient_norm': gradient_norm,  # 梯度范数
            'sample_size': data_sum      # 样本数量
        }
        
        # 获取更新后的模型参数
        weights = self.model.state_dict()
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 返回数据量、模型权重和统计数据
        return data_sum, weights, stats
    
    def upload(self, data_sum, weights, stats):
        """
        上传训练结果和统计数据到服务器
        这是对基类upload方法的重写，添加了训练统计数据的上传
        
        参数:
            data_sum: 训练数据量
            weights: 更新后的模型权重
            stats: 训练统计数据
        """
        # 构建更新字典，包含客户端ID、模型权重、数据量、时间戳和训练统计数据
        update_dict = {
            "client_id": self.client_id,      # 客户端ID
            "weights": weights,               # 更新后的模型权重
            "data_sum": data_sum,             # 训练数据量
            "time_stamp": self.time_stamp,    # 时间戳
            "loss": stats['loss'],            # 训练损失
            "gradient_norm": stats['gradient_norm'],  # 梯度范数
            "execution_time": stats['execution_time'] # 执行时间（来自self.delay）
        }
        
        # 将更新字典放入上行队列，发送到服务器
        self.message_queue.put_into_uplink(update_dict)
        # 打印上传信息
        print(f"Client {self.client_id} uploaded with stats: loss={stats['loss']:.4f}, grad_norm={stats['gradient_norm']:.4f}, exec_time={stats['execution_time']:.2f}s") 