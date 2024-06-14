class ScheduleCaller:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def schedule(self, client_list, data_dist=None,*args, **kwargs):
        return self.scheduler.schedule_method.schedule(client_list,data_dist)
