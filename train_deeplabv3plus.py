import argparse
import os
import matplotlib.pyplot as plt

import torch
import lightning.pytorch as pl
from lightning.pytorch.plugins import TorchSyncBatchNorm
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# from datasets.segmentation.pascal import PascalDataset
# from datasets.segmentation.coco import CocoDatasetPascalClasses
# from datasets.segmentation.synth_pascal import SyntheticPascalDataset
from datasets.VOCDataset import VOCDataset
from models.segmentation.unet import UNetResnet18
from models.segmentation.deeplab_v3 import DeeplabV3Resnet50, DeeplabV3Resnet101
from models.segmentation.deeplab_v3plus import DeeplabV3Plus
import numpy as np
import datetime
from experiment.deeplabv3_voc.config import cfg

MODELS = {
    "UNetResnet18": UNetResnet18,
    "DeeplabV3Resnet50": DeeplabV3Resnet50,
    "DeeplabV3Resnet101": DeeplabV3Resnet101,
    "DeeplabV3Plus": DeeplabV3Plus,
}


def collect_model_kwargs(args, train_loader):
    kwargs = {}
    kwargs['max_epochs'] = args.max_epochs
    kwargs['dataloader_length'] = len(train_loader) / args.num_gpus
    print('dataloader_length: ', kwargs['dataloader_length'])
    return kwargs


def prep_include_classes(args):
    # use default if no val classes are specified
    if args.include_classes_train is None:
        args.include_classes_train = args.include_classes
    if args.include_classes_val is None:
        args.include_classes_val = args.include_classes

    include_classes_train = args.include_classes_train
    # allow user to specify classes with space with an underscore instead
    include_classes_train = [c.replace('_', ' ') for c in include_classes_train]
    if 'all' in include_classes_train:
        include_classes_train = 'all'

    include_classes_val = args.include_classes_val
    # allow user to specify classes with space with an underscore instead
    include_classes_val = [c.replace('_', ' ') for c in include_classes_val]
    if 'all' in include_classes_val:
        include_classes_val = 'all'

    return include_classes_train, include_classes_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--val_every_n_epochs", type=int, default=1)
    parser.add_argument("--wandb_group", type=str, default="FT_baseline_runs")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--wandb_debug", action='store_true', default=False)

    # experiment parameters
    parser.add_argument("--model", type=str, default="DeeplabV3Plus")
    parser.add_argument("--from_scratch", action='store_true', default=False)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_synthetic", action='store_true', default=False)
    parser.add_argument('--include_classes', type=str, nargs='+', default=["Horse"])
    parser.add_argument('--include_classes_train', type=str, nargs='+')
    parser.add_argument('--include_classes_val', type=str, nargs='+')
    parser.add_argument('--use_gpt_synth', action='store_true', default=False)
    parser.add_argument('--use_prob_mask', action='store_true', default=False)
    parser.add_argument('--eval_dataset', type=str, nargs='+', default=['pascal'])
    parser.add_argument('--coco_random_subset', type=float, default=None)

    # synthetic filter parameters
    parser.add_argument("--synth_filter_method", type=str, default=None)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--topk_ratio", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--samples_per_class", type=int, default=None)
    parser.add_argument("--synthetic_data_folder", type=str, default='./data/results/')

    args = parser.parse_args()

    model_name = args.model
    pretrained = not args.from_scratch
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    log_freq = args.log_freq
    log_every_n_steps = args.log_every_n_steps
    wandb_group = args.wandb_group
    wandb_name = args.exp_name
    checkpoint = args.checkpoint
    use_synthetic = args.use_synthetic
    save_topk = 1
    save_last = True
    limit_train_batches = None
    limit_val_batches = None
    if args.debug:
        max_epochs = 4
        os.environ["WANDB_MODE"] = "dryrun"
        num_workers = 0
        batch_size = 16
        log_freq = 1
        save_last = False
        save_topk = 0
    if args.wandb_debug:
        num_workers = 0
        batch_size = 16
        limit_val_batches = 2
        limit_train_batches = 2
        wandb_group = "wandb_debugging"
        wandb_name = f"dummy_{datetime.datetime.now().__str__()}"
        save_last = False
        save_topk = 0

    pl.seed_everything(args.seed)

    # TODO: do selection with multiple classes
    return_prob = False
    if args.use_prob_mask:
        return_prob = True

    include_classes_train, include_classes_val = prep_include_classes(args)

    voc_train_dataset = VOCDataset('VOC2012', cfg, 'train', cfg.DATA_AUG)
    train_dataset = voc_train_dataset

    val_loaders = []
    for v_dset in args.eval_dataset:

        # TODO is return prob needed for val?
        if v_dset == 'pascal':
            val_dataset = VOCDataset('VOC2012', cfg, 'val', False)
        # elif v_dset == 'coco':
        #     val_dataset = CocoDatasetPascalClasses(split="val", resize_mode="center-crop", include_classes=include_classes_val, return_dummy_prob=return_prob)
        #     if args.coco_random_subset is not None:
        #         num_samples = int(args.coco_random_subset * len(val_dataset))
        #         indices = np.arange(len(val_dataset))
        #         np.random.shuffle(indices)
        #         indices = indices[:num_samples]
        #         val_dataset = torch.utils.data.Subset(val_dataset, indices)
        else:
            raise ValueError(f'Unknown eval dataset {args.eval_dataset}')

        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                                drop_last=True)
        val_loaders.append(val_loader)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True)

    model_kwargs = collect_model_kwargs(args, train_loader)
    model = MODELS[model_name](class_names=voc_train_dataset.classes,
                               ignore_index=voc_train_dataset.ignore_index,
                               visualizer_kwargs=voc_train_dataset.visualizer_kwargs,
                               **model_kwargs
                               )

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=True)

    checkpoint_callbacks = []
    for i in range(len(val_loaders)):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=f'val_{i}_loss',
            dirpath=f'./checkpoints/{args.exp_name}/',
            filename=f'model_checkpoint_{args.exp_name}',
            save_top_k=save_topk,  # Save top1 Why?? this is 40GB of checkpoints -->> # Save all checkpoints.
            mode='min',  # Mode for comparing the monitored metric
            save_last=save_last,
        )
        checkpoint_callbacks.append(checkpoint_callback)

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_callback] + checkpoint_callbacks

    logger = pl.loggers.WandbLogger(
        name=wandb_name or "segmentation_test, model={}".format(model_name),
        group=wandb_group or "markusShit",
        project="neuripsV2",
        log_model="all",
        entity="vision-lab",
    )

    # watch model
    logger.watch(model, log="all", log_freq=log_freq)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy="ddp",
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        limit_train_batches=limit_train_batches,  # None unless --wandb_debug flag is set
        limit_val_batches=limit_val_batches,  # None unless --wandb_debug flag is set
        check_val_every_n_epoch=args.val_every_n_epochs,  # None unless --wandb_debug flag is set
        sync_batchnorm=True if args.num_gpus > 1 else False,
    )
    if trainer.global_rank == 0:
        logger.experiment.config.update(args)

    if not args.debug:
        trainer.validate(model, dataloaders=val_loaders)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loaders,
    )


if __name__ == "__main__":
    import sys

    # filter_ops = ["mean_filter", "keep_only_target_classes"]
    # args = [
    #     '--debug', '--eval_dataset', 'pascal', #, 'coco', '--coco_random_subset', '0.1',
    # ]
    # sys.argv.extend(args)
    main()
