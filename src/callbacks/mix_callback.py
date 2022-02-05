from pytorch_lightning.callbacks import Callback
import numpy as np
import torch


class MixBaseCallback(Callback):
    """Abstract callback for mix data operations"""
    device = torch.device('cpu')

    def _generate_dataset_sample(self, batch_size, dataloader):
        dataset = dataloader.dataset.datasets
        sample = [dataset[i] for i in np.random.randint(0, len(dataset) - 1, batch_size)]
        sample = dataloader.loaders.collate_fn(sample)
        sample_x, sample_y = sample
        return sample_x.to(self.device), sample_y.to(self.device)

    def _generate_batch_sample(self, batch_x, batch_y):
        batch_size = len(batch_x)
        idxs = [np.random.choice([_ for _ in range(batch_size) if _ != i]) for i in range(batch_size)]
        idxs = torch.tensor(idxs, dtype=torch.long)
        sample_x, sample_y = batch_x[idxs], batch_y[idxs]
        return sample_x.to(self.device), sample_y.to(self.device)

    def _unsqueeze(self, tensor, ndims=0, dim=0):
        for _ in range(ndims):
            tensor = tensor.unsqueeze(dim)
        return tensor

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        raise NotImplementedError('You must override method "on_train_batch_start"')
