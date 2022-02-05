import numpy as np
from pytorch_lightning.callbacks import Callback

from albumentations.augmentations.transforms import Normalize
from albumentations.core.composition import Compose


class TTA(Callback):
    """ Callback for test-time-augmentation """
    def __init__(
            self,
            augmentations=None
    ):
        """
        :param augmentations: list of augmentation transforms
        """
        super().__init__()
        self.augmentations = [] if augmentations is None else augmentations
        self.collate_fns = None
        self.transforms = None
        self.inv_transforms = None

    def on_predict_start(self, trainer, pl_module):
        if self.transforms is None:
            self.__init_transform(trainer)
        if self.collate_fns is None:
            self.__init_collate_fn(trainer)

        for dataloader_idx in range(len(trainer.predict_dataloaders)):
            trainer.predict_dataloaders[dataloader_idx].collate_fn = self.__collate_fn(dataloader_idx)

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        for dataloader_idx in range(len(outputs)):
            for batch_idx in range(len(outputs[dataloader_idx])):
                output = outputs[dataloader_idx][batch_idx]
                output = output.view(len(self.augmentations) + 1, -1)
                output = output.mean(0)
                outputs[dataloader_idx][batch_idx] = output

    def __collate_fn(self, dataloader_idx):
        collate_fn = self.collate_fns[dataloader_idx]
        transform = self.transforms[dataloader_idx]
        inv_transform = self.inv_transforms[dataloader_idx]

        def collate_fn_wrapper(batch):
            # TODO collate_fn_wrapper multiprocessing optimization
            images = [np.rint(inv_transform(image=image.permute(1, 2, 0).numpy())['image']) for image in batch]
            images_augmented = []
            for augmentation in self.augmentations:
                for image in images:
                    images_augmented.append(augmentation(image=image)['image'])
            images = images + images_augmented
            images = [transform(image=image)['image'] for image in images]
            batch = collate_fn(images)
            return batch

        return collate_fn_wrapper

    def __get_inv_transform(self, transforms):
        inv_transforms = []
        inv_transforms.append(Normalize(mean=[0, 0, 0], std=[1 / 255.0, 1 / 255.0, 1 / 255.0], max_pixel_value=1.0))
        for transform_idx in range(len(transforms)):
            transform_ = transforms[transform_idx]
            if transform_.__class__.__name__ == 'Normalize':
                inv_transforms.append(
                    Normalize(mean=[- _ for _ in transform_.mean], std=[1, 1, 1], max_pixel_value=1.0))
                inv_transforms.append(
                    Normalize(mean=[0, 0, 0], std=[1 / _ for _ in transform_.std], max_pixel_value=1.0))
        inv_transforms = inv_transforms[::-1]
        return Compose(inv_transforms)

    def __init_transform(self, trainer):
        self.inv_transforms = {}
        self.transforms = {}
        for dataloader_idx in range(len(trainer.predict_dataloaders)):
            transforms = trainer.predict_dataloaders[dataloader_idx].dataset.transform.transforms.transforms
            self.transforms[dataloader_idx] = Compose(transforms)
            self.inv_transforms[dataloader_idx] = self.__get_inv_transform(transforms)

    def __init_collate_fn(self, trainer):
        self.collate_fns = {}
        for dataloader_idx in range(len(trainer.predict_dataloaders)):
            self.collate_fns[dataloader_idx] = trainer.predict_dataloaders[dataloader_idx].collate_fn
