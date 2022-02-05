from optimizers import Optimizer
from inspect import signature


class Scheduler(object):
    def __init__(self, scheduler_class, **kwargs):
        self.scheduler_class = scheduler_class
        self.kwargs = {arg: kwargs[arg] for arg in kwargs if arg in dict(signature(scheduler_class).parameters)}
        self.options = {arg: kwargs[arg] for arg in kwargs if arg not in self.kwargs}

    def __call__(self, optimizer: Optimizer):
        scheduler = self.scheduler_class(optimizer=optimizer, **self.kwargs)
        self.options['scheduler'] = scheduler
        return self.options
