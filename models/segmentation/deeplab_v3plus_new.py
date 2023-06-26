import torch
import segmentation_models_pytorch as smp
from models.segmentation import custom_focal_loss
from net.deeplabv3plus_new import deeplabv3plus_resnet101
from net._deeplab import convert_to_separable_conv
import utils
from experiment.deeplabv3_cityscapes.config import cfg

from models.segmentation._abstract import SegmentationModel


class DeeplabV3Plus(SegmentationModel):

    encoder_name = None

    def __init__(self, dataloader_length, **kwargs):
        self.dataloader_length = dataloader_length

        # Model stuff
        self.backbone = kwargs['backbone']
        self.num_classes = kwargs['num_classes']
        self.output_stride = kwargs['output_stride']
        self.separable_conv = kwargs['separable_conv']

        # Loss
        self.loss_type = kwargs['loss_type']

        # Optimizer and schedules
        self.lr = kwargs['lr']
        self.lr_policy = kwargs['lr_policy']
        self.weight_decay = kwargs['weight_decay']
        self.max_steps = kwargs['max_steps']
        self.step_size = kwargs['step_size']

        # Validation scales
        self.val_scales = kwargs['val_dataset']
        self.val_scales = kwargs['val_scales']

        super().__init__(**kwargs, normalize_images=False)

    def initialize_model(self):
        if self.backbone == 'resnet101':
            model = deeplabv3plus_resnet101(num_classes=self.num_classes, output_stride=self.output_stride,
                                            pretrained_backbone=True)
        else:
            raise ValueError(f"Backbone '{self.backbone}' not supported")

        if self.separable_conv:
            convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
        return model

    def initialize_loss(self):
        if self.loss_type == 'focal_loss':
            loss = utils.FocalLoss(ignore_index=255, size_average=True)
        elif self.loss_type == 'cross_entropy':
            loss = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        else:
            raise ValueError(f"loss_type '{self.loss_type}' not supported")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=[
            {'params': self.model.backbone.parameters(), 'lr': 0.1 * self.lr},
            {'params': self.model.classifier.parameters(), 'lr': self.lr},
        ], lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        if self.lr_policy == 'poly':
            scheduler = utils.PolyLR(optimizer, self.max_steps, power=0.9)
        elif self.lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=0.1)
        else:
            raise ValueError(f"lr_policy '{self.lr_policy}' not supported")
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks = batch
        dataset_name = self.val_dataset[dataloader_idx // len(self.val_scales)]
        dataset_scale = self.val_scales[dataloader_idx % len(self.val_scales)]
        if dataset_scale is None:
            dataset_scale = ""
        else:
            dataset_scale = f"_{dataset_scale:.2f}"
        self._step(images, masks, f"val_{dataset_name}" + dataset_scale)


if __name__ == '__main__':

    model = DeeplabV3Plus()