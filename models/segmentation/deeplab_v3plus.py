import torch
import segmentation_models_pytorch as smp
from models.segmentation import custom_focal_loss
from net.deeplabv3plus import deeplabv3plus as dlv3p
from experiment.deeplabv3_voc.config import cfg

from models.segmentation._abstract import SegmentationModel


class DeeplabV3Plus(SegmentationModel):

    encoder_name = None

    def __init__(self, max_epochs, dataloader_length, **kwargs):
        self.max_epochs = max_epochs
        self.dataloader_length = dataloader_length
        # self.loss_mode = kwargs.get("loss_mode", "multiclass")

        super().__init__(**kwargs, normalize_images=False)

    def initialize_model(self):
        model = dlv3p(cfg)
        return model

    def initialize_loss(self):
        # loss = smp.losses.FocalLoss(mode=self.loss_mode, ignore_index=self.ignore_index)
        loss = torch.nn.CrossEntropyLoss(ignore_index=255)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            params=[
                {'params': self.get_params(self.model, key='1x'), 'lr': cfg.TRAIN_LR},
                {'params': self.get_params(self.model, key='10x'), 'lr': 10 * cfg.TRAIN_LR}
            ],
            momentum=cfg.TRAIN_MOMENTUM
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
            lambda x: (1 - x / (self.max_epochs * self.dataloader_length + 1)) ** cfg.TRAIN_POWER,
            lambda x: 10 * (1 - x / (self.max_epochs * self.dataloader_length + 1)) ** cfg.TRAIN_POWER
        ])
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def get_params(self, model, key):
        for m in model.named_modules():
            if key == '1x':
                if 'backbone' in m[0] and isinstance(m[1], torch.nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == '10x':
                if 'backbone' not in m[0] and isinstance(m[1], torch.nn.Conv2d):
                    for p in m[1].parameters():
                        yield p

# class DeeplabV3Resnet50(_DeeplabV3):
#     encoder_name = "resnet50"
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#
# class DeeplabV3Resnet101(_DeeplabV3):
#     encoder_name = "resnet101"
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)


if __name__ == '__main__':

    model = DeeplabV3Plus()
