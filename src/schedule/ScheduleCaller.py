class ScheduleCaller:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def schedule(self, client_list=None, *args, **kwargs):
        if client_list is not None:
            return self.scheduler.schedule_method.schedule(client_list,*args, **kwargs)
        else:
            return self.scheduler.schedule_method.schedule(*args, **kwargs)
