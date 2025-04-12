from anyio import current_time
import wandb

from updater.BaseUpdater import BaseUpdater
from updater.mixin.MoreTest import TestEachClass, TestMultiTask
import time

class SyncUpdater(BaseUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        BaseUpdater.__init__(self, server_thread_lock, stop_event, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem

    def run(self):
        for _ in range(self.T):
            self.full_sem.acquire()
            self.mutex_sem.acquire()

            update_list = self.get_update_list()
            self.server_thread_lock.acquire()
            epoch = self.current_time.get_time()
            self.server_update(epoch, update_list)
            self.global_var['system_time_cost'].append(time.time())
            self.server_thread_lock.release()

            self.current_time.time_add()
            self.mutex_sem.release()
            self.empty_sem.release()

        print("Average delay =", (self.sum_delay / self.T))

    def server_update(self, epoch, update_list):
        self.update_server_weights(epoch, update_list)
        acc, loss, run_time = self.run_server_test(epoch)

        if 'group_selected_at_global_epoch' in self.global_var and epoch > 1: # 如果是CustomGroupManager，则记录每个global epoch选择的group以及每个group的历史improved loss
            preadv_acc, preadv_loss = self.get_last_accuracy_and_loss()
            selected_group_id = self.global_var['group_selected_at_global_epoch'][-1]
            # 记录每个group的历史improved loss
            self.global_var['group_history_improved_loss'][selected_group_id].append((epoch, preadv_loss - loss))

            # 记录每个group的历史improved acc
            self.global_var['group_history_improved_acc'][selected_group_id].append((epoch, acc - preadv_acc))
            # 记录每个group的平均loss

        if 'group_selected_at_global_epoch' in self.global_var:
            selected_group_id = self.global_var['group_selected_at_global_epoch'][-1]
            # 记录每个group的历史loss
            self.global_var['group_history_loss'][selected_group_id].append((epoch, loss))
            # 记录每个group的历史acc
            self.global_var['group_history_acc'][selected_group_id].append((epoch, acc))
            self.global_var['group_epoch_avg_loss'][selected_group_id].append((epoch, self.global_var['epoch_avg_loss'][epoch]))
            # 记录每个group的平均数据利用率
            self.global_var['group_data_util'][selected_group_id].append((epoch, self.global_var['epoch_group_data_util'][epoch]))
            # 记录每个group的累积数据利用率
            self.global_var['group_data_util_accumulate'][selected_group_id].append((epoch, self.global_var['epoch_group_data_util_accumulate'][epoch]))

            self.global_var['epoch_accuacy'].append((epoch, acc))
        if self.config['enabled']:
            wandb.log({'accuracy': acc, 'loss': loss, 'run_time': run_time})

    def get_update_list(self):
        update_list = []
        # receive all updates
        while not self.queue_manager.empty():
            update_list.append(self.queue_manager.get())
        return update_list


class SyncUpdaterWithDetailedTest(TestEachClass, SyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        TestEachClass.__init__(self)
        SyncUpdater.__init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)

class SyncUpdaterWithTaskTest(TestMultiTask, SyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        TestMultiTask.__init__(self)
        SyncUpdater.__init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)
