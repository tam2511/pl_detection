from typing import Optional, Any
import torch
import numpy as np
import cv2
import os
import uuid
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from albumentations.augmentations.transforms import Normalize
from albumentations.core.composition import Compose

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.75
prediction_color = (255, 0, 0)
delimiter_color = (0, 255, 0)
target_color = (0, 0, 255)
thickness = 1
extra_padding = 2
padding = 10
delimeter = '-'


class ClassificationImageLogger(Callback):
    """ Callback for image and predictions logging """

    def __init__(
            self,
            mode: str = 'val',
            n_images: int = 1,
            class_names: list = None,
            n_top_classes: int = 1,
            output_dir: str = None
    ):
        """
        :param mode: 'train' or 'val'
        :param n_images: number of log images per epoch
        :param class_names: class names
        :param n_top_classes: number of classes which will be drawing
        :param output_dir: output dir for logging [default is lightning_logs]
        """
        super().__init__()
        self.mode = mode
        self.n_images = n_images
        self.class_names = class_names
        self.n_top_classes = n_top_classes
        self.output_dir = output_dir

        self.idxs = None
        self.inv_transform = None

        self.n_handled = 0 if mode == 'train' else {}

    def __get_inv_transform(self, transform):
        transforms = transform.transforms.transforms
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
        if self.mode == 'train':
            self.inv_transform = self.__get_inv_transform(trainer.train_dataloader.dataset.datasets.transform)
        else:
            self.inv_transform = {}
            for dataloader_idx in range(len(trainer.val_dataloaders)):
                transform = trainer.val_dataloaders[dataloader_idx].dataset.transform
                self.inv_transform[dataloader_idx] = self.__get_inv_transform(transform)

    def __get_random_idxs(self, dataloader):
        bias = len(dataloader.dataset) % len(dataloader) if dataloader.drop_last and len(dataloader) > 0 else 0
        idxs = torch.randperm(len(dataloader.dataset) - bias)[:self.n_images]
        batch_size = dataloader.batch_size
        batch_idxs = idxs // batch_size
        idxs = idxs % batch_size
        return torch.stack([batch_idxs, idxs])

    def __init_idxs(self, trainer):
        if self.mode == 'train':
            if isinstance(trainer.train_dataloader.sampler, torch.utils.data.sampler.SequentialSampler):
                self.idxs = self.__get_random_idxs(trainer.train_dataloader)
        else:
            self.idxs = {}
            for dataloader_idx in range(len(trainer.val_dataloaders)):
                self.idxs[dataloader_idx] = None
                if isinstance(trainer.val_dataloaders[dataloader_idx].sampler,
                              torch.utils.data.sampler.SequentialSampler):
                    self.idxs[dataloader_idx] = self.__get_random_idxs(trainer.val_dataloaders[dataloader_idx])

    def __init_output_dir(self):
        root = os.path.join(os.getcwd(), 'lightning_logs')
        version_logs = [dir_name for dir_name in os.listdir(root) if os.path.isdir(os.path.join(root, dir_name))]
        last_version = max(map(lambda x: int(x.split('_')[-1]), version_logs))
        self.output_dir = os.path.join(root, 'version_{}'.format(last_version), 'images')
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def __text_size(string):
        text_size, _ = cv2.getTextSize(text=string, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontScale,
                                       thickness=thickness)
        return text_size

    def __text_area_size(self, predictions, targets):
        text_sizes = []
        for class_name in predictions:
            text_sizes.append(self.__text_size('{}: {:.2f}'.format(class_name, predictions[class_name].item())))
        for target in targets:
            text_sizes.append(self.__text_size(target))
        text_w = max([_[0] for _ in text_sizes]) + 1
        text_size, _ = cv2.getTextSize(text=delimeter, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1)
        text_h = sum(_[1] + extra_padding for _ in text_sizes) + padding + text_size[1]
        return text_w, text_h

    def __draw_predictions(self, image, bias, classes_info, color):
        for class_name in classes_info:
            row_str = '{}: {:.2f}'.format(class_name, classes_info[class_name].item())
            text_size = self.__text_size(row_str)
            bias = image.shape[1] + text_size[1] + 1 if bias == 0 else bias + text_size[1] + 1
            image = cv2.putText(image, row_str, (1, bias), font, fontScale, color, thickness,
                                cv2.LINE_AA)
        return image, bias

    def __draw_results(self, image, predictions, targets):
        text_w, text_h = self.__text_area_size(predictions, targets)
        new_image = np.zeros((image.shape[0] + text_h, max(image.shape[1], text_w), image.shape[2]), dtype='uint8')
        new_image[:image.shape[0], :image.shape[1], :image.shape[2]] = image
        bias = 0
        new_image, bias = self.__draw_predictions(new_image, bias, predictions, prediction_color)
        text_size, _ = cv2.getTextSize(text=delimeter * image.shape[0], fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                                       thickness=1)
        bias = bias + text_size[1] + 1
        new_image = cv2.putText(new_image, delimeter * image.shape[0], (0, bias), font, fontScale, delimiter_color,
                                thickness, cv2.LINE_AA)
        new_image, bias = self.__draw_predictions(new_image, bias, targets, target_color)
        return new_image

    def __check_part(self, trainer, outputs, dataloader_idx):
        if isinstance(self.n_handled, dict) and dataloader_idx not in self.n_handled:
            self.n_handled[dataloader_idx] = 0
        if self.output_dir is None:
            self.__init_output_dir()
        if self.idxs is None:
            self.__init_idxs(trainer)
        if self.inv_transform is None:
            self.__init_transform(trainer)
        if 'output' not in outputs:
            raise ValueError(
                f'For use the ImageLogger in "{self.mode}" mode, you must pass the "return_{self.mode}_output=True" to the Learner')

    def __common_part1(self, outputs, batch, batch_idx, dataloader_idx):
        output = outputs['output']
        idxs = self.idxs if self.mode == 'train' else self.idxs[dataloader_idx]
        x, y = batch
        x, y, output = x.cpu().float(), y.cpu(), output.cpu().float()
        x = torch.permute(x, (0, 2, 3, 1))
        if idxs is None:
            n_handled = self.n_handled if isinstance(self.n_handled, int) else self.n_handled[dataloader_idx]
            if self.n_images <= self.n_handled:
                return
            x, y = x[:self.n_images - n_handled], y[:self.n_images - n_handled]
            output = output[:self.n_images - n_handled]
            if isinstance(self.n_handled, int):
                self.n_handled += len(x)
            else:
                self.n_handled[dataloader_idx] += len(x)
        else:
            filtered_idxs = idxs[1, torch.where(idxs[0] == batch_idx)[0]]
            if len(filtered_idxs) == 0:
                return
            x, y, output = x[filtered_idxs], y[filtered_idxs], output[filtered_idxs]
        x = x.numpy()
        return x, y, output

    def __common_part2(self, x, y, output, epoch, dataloader_idx):
        transform = self.inv_transform if self.mode == 'train' else self.inv_transform[dataloader_idx]
        x = [np.rint(transform(image=_)['image']).astype('uint8') for _ in x]
        values, indexes = torch.topk(output, dim=1, k=min(self.n_top_classes, output.size(1)))
        results = [{self.class_names[indexes[obj_idx][i]]: values[obj_idx][i] for i in range(len(indexes[obj_idx]))}
                   for obj_idx in range(len(indexes))]
        values, indexes = torch.topk(y, dim=1, k=min(self.n_top_classes, output.size(1)))
        targets = [{self.class_names[indexes[obj_idx][i]]: values[obj_idx][i] for i in range(len(indexes[obj_idx]))}
                   for obj_idx in range(len(indexes))]
        for i in range(len(x)):
            x[i] = self.__draw_results(x[i], results[i], targets[i])
            dest_dir = os.path.join(self.output_dir, f'epoch_{epoch}', self.mode)
            dest_dir = os.path.join(dest_dir, f'dataloader_{dataloader_idx}') if self.mode == 'val' else dest_dir
            os.makedirs(dest_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dest_dir, '{}.jpg'.format(uuid.uuid4())), x[i])

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ):
        if self.mode != 'val':
            return
        self.__check_part(trainer, outputs, dataloader_idx)
        result_part1 = self.__common_part1(outputs, batch, batch_idx, dataloader_idx)
        if result_part1 is None:
            return
        x, y, output = result_part1
        self.__common_part2(x, y, output, trainer.current_epoch, dataloader_idx)

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ):
        if self.mode != 'train':
            return
        self.__check_part(trainer, outputs, dataloader_idx)
        result_part1 = self.__common_part1(outputs, batch, batch_idx, dataloader_idx)
        if result_part1 is None:
            return
        x, y, output = result_part1
        self.__common_part2(x, y, output, trainer.current_epoch, dataloader_idx)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.mode != 'val':
            return
        for dataloader_idx in self.n_handled:
            self.n_handled[dataloader_idx] = 0
        self.__init_idxs(trainer)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None):
        if self.mode != 'train':
            return
        self.n_handled = 0
        self.__init_idxs(trainer)
