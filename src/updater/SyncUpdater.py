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
            self.server_thread_lock.release()

            self.current_time.time_add()
            self.mutex_sem.release()
            self.empty_sem.release()

        print("Average delay =", (self.sum_delay / self.T))

    def server_update(self, epoch, update_list):
        if epoch > 1:
            _, preadv_loss = self.get_last_accuracy_and_loss()
        self.update_server_weights(epoch, update_list)
        acc, loss, run_time = self.run_server_test(epoch)
        if 'group_selected_at_global_epoch' in self.global_var and epoch > 1: # 如果是CustomGroupManager，则记录每个global epoch选择的group以及每个group的历史improved loss
            selected_group_id = self.global_var['group_selected_at_global_epoch'][-1]
            self.global_var['group_history_improved_loss'][selected_group_id].append((epoch, preadv_loss - loss))
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
