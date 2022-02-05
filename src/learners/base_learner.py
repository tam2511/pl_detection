from pytorch_lightning import LightningModule
import torch

from lr_schedulers import Scheduler
from metrics import MetricsList
from optimizers import Optimizer

MAIN_LOSS_KEY = 'loss'


class BaseLearner(LightningModule):
    def __init__(
            self,
            model: torch.nn.Module = None,
            loss: torch.nn.Module = None,
            optimizer: Optimizer = None,
            lr_scheduler: Scheduler = None,
            train_metrics: list = None,
            val_metrics: list = None,
            return_val_output=False,
            return_train_output=False,
    ):
        """
        :param model: torch.nn.Module model
        :param loss: torch.nn.Module loss function
        :param optimizer: Optimizer wrapper object
        :param lr_scheduler: Scheduler object for lr scheduling
        :param train_metrics: list of train metrics
        :param val_metrics:list of val metrics
        :param return_val_output: if True will return output of model in validation step
        :param return_train_output: if True will return output of model in training step
        """
        super().__init__()
        self.model = model
        self.loss_f = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_metrics = MetricsList()
        if not train_metrics is None:
            for train_metric in train_metrics:
                self.train_metrics.add(metric=train_metric)
        self.val_metrics = MetricsList([])
        if not val_metrics is None:
            for val_metric in val_metrics:
                self.val_metrics.add(metric=val_metric)
        self.return_val_output = return_val_output
        self.return_train_output = return_train_output

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, batch_idx):
        """
        @return: tuple of loss, output, target, return_output
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss, output, target, return_output = self.common_step(batch, batch_idx)
        if isinstance(loss, dict):
            for loss_key in loss:
                self.log('train/{}_loss'.format(loss_key), loss, on_step=True, on_epoch=False)
        else:
            self.log('train/loss', loss, on_step=True, on_epoch=False)
        self.__log_lr()
        self.train_metrics.update(output, target)
        ret = {'loss': loss[MAIN_LOSS_KEY] if isinstance(loss, dict) else loss}
        if self.return_train_output:
            ret['output'] = return_output
        return ret

    def training_epoch_end(self, train_step_outputs):
        train_metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        for metric_name in train_metrics:
            self.log(f'train/{metric_name}', train_metrics[metric_name], on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, output, target, return_output = self.common_step(batch, batch_idx)
        if isinstance(loss, dict):
            for loss_key in loss:
                self.log('val/{}_loss'.format(loss_key), loss, on_step=True, on_epoch=False)
        else:
            self.log('val/loss', loss, on_step=True, on_epoch=False)
        self.val_metrics.update(output, target)
        ret = {'loss': loss[MAIN_LOSS_KEY] if isinstance(loss, dict) else loss}
        if self.return_val_output:
            ret['output'] = return_output
        return ret

    def validation_epoch_end(self, val_step_outputs):
        val_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        for metric_name in val_metrics:
            self.log(f'val/{metric_name}', val_metrics[metric_name], on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))
        lr_scheduler = self.lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]

    def __log_lr(self):
        optimizer = self.optimizers()
        lrs = [group['lr'] for group in optimizer.param_groups]
        grouped_lrs = {}
        for idx, lr in enumerate(lrs):
            if lr not in grouped_lrs:
                grouped_lrs[lr] = []
            grouped_lrs[lr].append(idx)
        if len(grouped_lrs) == 1:
            self.log('lr', lrs[0], on_step=True, on_epoch=False, prog_bar=True)
        else:
            for lr in grouped_lrs:
                ids = ','.join(map(str, grouped_lrs[lr]))
                self.log(f'lr_groups[{ids}]', lr, on_step=True, on_epoch=False, prog_bar=True)
