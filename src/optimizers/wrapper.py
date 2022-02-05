from torch.optim import Optimizer as TorchOptimizer


class Optimizer(object):
    def __init__(self, optimizer_class, **kwargs):
        self.optimizer_class = optimizer_class
        self.kwargs = kwargs

    def __call__(self, params) -> TorchOptimizer:
        return self.optimizer_class(params=params, **self.kwargs)
