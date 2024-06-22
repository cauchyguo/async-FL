import random
import copy
import numpy as np
from torch.utils.data import Dataset

from utils import Random
from utils.Tools import dict_to_list, list_to_dict
from utils.GlobalVarGetter import GlobalVarGetter
class CustomizeDistTest():

    def __init__(self, dataset):
        pass
    def test_customize_dist(self):
        # Define the distribution
        class MyDist(Distribution):
            def __init__(self, *args, **kwargs):
                super(MyDist, self).__init__(*args, **kwargs)
                self._dist = stats.norm(loc=0, scale=1)

            def sample(self, size=None):
                return self._dist.rvs(size=size)

            def log_prob(self, x):
                return self._dist.logpdf(x)
            

def generate_non_iid_data(iid_config, dataset, clients, left, right, datasets):
    if "customize" in iid_config.keys() and iid_config["customize"]:
        label_config = iid_config['label']
        data_config = iid_config['data']
        return customize_distribution(label_config, data_config, dataset, clients, left, right, datasets)
def customize_distribution(label_config, data_config, dataset, clients, left, right, datasets):
    # 生成label lists
    # 洗牌算法
    label_lists = []
    shuffle = False
    if "shuffle" in label_config.keys() and label_config["shuffle"]:
        shuffle = True
    if isinstance(label_config, dict):
        # step
        if "step" in label_config.keys():
            label_lists = generate_label_lists_by_step(label_config["step"], label_config["list"], left, right, shuffle)
        # {list:[]}
        elif "list" in label_config.keys():
            label_lists = generate_label_lists(label_config["list"], left, right, shuffle)
        # {[],[],[]}
        else:
            label_lists = dict_to_list(label_config)
    # 生成data lists
    # {}
    if len(data_config) == 0:
        size = dataset.train_data_size // clients
        data_lists = generate_data_lists(size, size, clients, label_lists)
    # max,min
    else:
        data_lists = generate_data_lists(data_config["max"], data_config["min"], clients, label_lists)
    # 保存label至配置文件
    dataset.iid_config['label'] = list_to_dict(label_lists)
    # 生成序列

    global_var = GlobalVarGetter.get()
    global_var['client_data_distribution'] = list_to_dict(label_lists)
    return generate_non_iid_dataset(dataset.train_data, dataset.train_labels, label_lists,
                                    data_lists)
def generate_data_list(max_size, min_size, num):
    ans = []
    for _ in range(num):
        ans.append(random.randint(min_size, max_size))
    return ans


def generate_label_lists(label_num_list, left, right, shuffle=False):
    label_lists = []
    if shuffle:
        label_total = 0
        for label_num in label_num_list:
            label_total = label_total + label_num
        epoch = int(label_total // (right - left)) + 1
        label_all_list = []
        for i in range(epoch):
            label_all_list = label_all_list + Random.shuffle_random(left, right)
        pos = 0
        for label_num in label_num_list:
            label_lists.append(label_all_list[pos: pos + label_num])
            pos += label_num
    else:
        labels = range(left, right)
        for label_num in label_num_list:
            label_list = np.random.choice(labels, label_num, replace=False)
            label_lists.append(label_list.tolist())
    return label_lists

# 生成label lists,step为步长，num_list为每个步长的数量，left为左边界，right为右边界
def generate_label_lists_by_step(step, num_list, left, right, shuffle=False):
    label_lists = []
    bound = 1
    if shuffle:
        label_total = 0
        label_all_lists = []
        for i in num_list:
            label_total += bound * i
            bound += step
        bound = 1
        epoch = int(label_total // (right - left)) + 1
        for i in range(epoch):
            label_all_lists += Random.shuffle_random(left, right)
        pos = 0
        for i in range(len(num_list)):
            for j in range(num_list[i]):
                label_lists.append(label_all_lists[pos: pos + bound])
                pos = pos + bound
            bound += step
    else:
        labels = range(left, right)
        for i in range(len(num_list)):
            for k in range(bound):
                remain_labels = copy.deepcopy(labels)
                for j in range(num_list[i] // bound):
                    s = np.random.choice(remain_labels, min(bound,len(remain_labels)), replace=False)
                    label_lists.append(s.tolist())
                    remain_labels = list(set(remain_labels) - set(s))
            bound += step
    return label_lists


def generate_data_lists(max_size, min_size, num, label_lists):
    data_lists = []
    data_list = generate_data_list(max_size, min_size, num)
    for i in range(len(label_lists)):
        tmp_list = []
        for j in range(len(label_lists[i]) - 1):
            tmp_list.append(data_list[i] // len(label_lists[i]))
        tmp_list.append(data_list[i] - data_list[i] // len(label_lists[i]) * (len(label_lists[i]) - 1))
        data_lists.append(tmp_list)
    return data_lists

def generate_non_iid_dataset(x, y, label_lists, data_lists):
    client_idx_list = []
    for i in range(len(label_lists)):
        index_list = []
        for j in range(len(label_lists[i])):
            ids = np.flatnonzero(y == label_lists[i][j])
            ids = np.random.choice(ids, data_lists[i][j], replace=False)
            index_list.append(ids)
        index_list = np.hstack(index_list)
        client_idx_list.append(index_list)
    return client_idx_list

        # # Create the distribution
        # dist = MyDist()

        # # Test the distribution
        # self.assertTrue(isinstance(dist.sample(), float))
        # self.assertTrue(isinstance(dist.log_prob(0), float))