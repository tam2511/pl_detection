import numpy as np
import torch
from callbacks.mix_callback import MixBaseCallback


class Cutmix(MixBaseCallback):
    """Callback for cutmixing"""

    def __init__(self, on_batch: bool = True, alpha: float = 0.4):
        """
        :param on_batch: If true get samples from batch, else from dataset
        :param alpha: param for cutmix operation
        """
        super().__init__()
        self.on_batch = on_batch
        self.alpha = alpha

    def __random_bbox(self, height, width, alpha):
        ratio = torch.sqrt(1. - alpha)
        w = width * ratio
        h = height * ratio

        # uniform
        cx = torch.rand(1)[0] * w
        cy = torch.rand(1)[0] * h
        x1 = torch.clip(cx - w // 2, torch.tensor(0), w).long()
        y1 = torch.clip(cy - h // 2, torch.tensor(0), h).long()
        x2 = torch.clip(cx + w // 2, torch.tensor(0), w).long()
        y2 = torch.clip(cy + h // 2, torch.tensor(0), h).long()
        return x1, y1, x2, y2

    def __mix(self, sample1, sample2, target1, target2, alpha):
        x1, y1, x2, y2 = self.__random_bbox(sample1.size(1), sample2.size(2), alpha)
        sample1[:, y1:y2, x1:x2] = sample2[:, y1:y2, x1:x2]
        alpha_ = 1 - (x2 - x1) * (y2 - y1) / (sample1.size(1) * sample1.size(2))
        target1 = target1 * alpha_ + target2 * (1 - alpha_)
        return sample1, target1

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        batch_x, batch_y = batch
        if self.device != batch_x.device:
            self.device = batch_x.device
        assert isinstance(batch_x, torch.Tensor)
        assert isinstance(batch_y, torch.Tensor)
        batch_size = batch_x.size(0)
        if self.on_batch:
            batch_x_, batch_y_ = self._generate_batch_sample(batch_x, batch_y)
        else:
            batch_x_, batch_y_ = self._generate_dataset_sample(batch_size, trainer.train_dataloader)
        alpha = torch.from_numpy(np.random.beta(self.alpha, self.alpha, batch_size)).to(self.device)
        for i in range(batch_size):
            x, y = self.__mix(batch_x[i], batch_x_[i], batch_y[i], batch_y_[i], alpha[i])
            batch_x[i] = x
            batch_y[i] = y
