import argparse
import os
import matplotlib.pyplot as plt

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from datasets.Cityscapes import Cityscapes
from models.segmentation.deeplab_v3plus_new import DeeplabV3Plus
import numpy as np
import datetime
from experiment.deeplabv3_cityscapes.config import cfg

# Imports from main
from utils import ext_transforms as et

MODELS = {
    "DeeplabV3Plus": DeeplabV3Plus,
}


def collect_model_kwargs(args, train_loader):
    kwargs = {}
    kwargs['dataloader_length'] = len(train_loader) / args.num_gpus

    # Model stuff
    kwargs['backbone'] = args.backbone
    kwargs['num_classes'] = args.num_classes
    kwargs['output_stride'] = args.output_stride
    kwargs['separable_conv'] = args.separable_conv

    # Loss stuff
    kwargs['loss_type'] = args.loss_type

    # Optimizer and scheduler stuff
    kwargs['lr'] = args.lr
    kwargs['lr_policy'] = args.lr_policy
    kwargs['weight_decay'] = args.weight_decay
    kwargs['max_steps'] = args.max_steps
    kwargs['step_size'] = args.step_size

    # For _abstract class
    if args.val_scales is None:
        kwargs['num_val_dataloaders'] = len(args.val_dataset)
    else:
        kwargs['num_val_dataloaders'] = 2 * len(args.val_dataset)


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
    # parser.add_argument("--lr", type=float)
    # parser.add_argument("--weight_decay", type=float)
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

    # args for merge with main

    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--backbone", type=str, default='resnet_101')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")

    parser.add_argument("--max_steps", type=int, default=30e3,
                        help="max number of learning steps (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--train_dataset', type=str, default='cityscapes')
    parser.add_argument('--val_dataset', type=str, nargs='+', default=['cityscapes'])

    parser.add_argument('--val_scales', type=float, nargs='+', default=None)

    args = parser.parse_args()

    model_name = args.model
    pretrained = not args.from_scratch
    max_steps = args.max_steps
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
        max_steps = 100
        os.environ["WANDB_MODE"] = "dryrun"
        num_workers = 0
        batch_size = 8
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

    train_transform = et.ExtCompose([
        et.ExtRandomScale(scale_range=(0.5, 2)),
        et.ExtRandomCrop(size=(args.crop_size, args.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    if args.train_dataset == 'cityscapes':
        cityscapes_train_dataset = Cityscapes(root=args.data_root, split='train',
                                              transform=train_transform)
        train_dataset = cityscapes_train_dataset
        args.num_classes = 19
    else:
        raise ValueError(f'Unknown train dataset {args.train_dataset}')

    val_loaders = []
    for v_dset in args.val_dataset:
        if v_dset == 'cityscapes':
            val_dataset = Cityscapes(root=args.data_root, split='val',
                                     transform=val_transform)
            if args.val_scales is not None:
                scaled_val_sets = []
                for scale in args.val_scales:
                    scaled_val_transform = et.ExtCompose([
                        et.ExtScale(scale=scale),
                        val_transform,
                    ])
                    scaled_dataset = Cityscapes(root=args.data_root, split='val',
                                                transform=scaled_val_transform)
                    scaled_val_sets.append(scaled_dataset)
                scaled_val_set = torch.utils.data.ConcatDataset(scaled_val_sets)
        else:
            raise ValueError(f'Unknown eval dataset {args.val_dataset}')

        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
        val_loaders.append(val_loader)

        if args.val_scales is not None:
            scaled_val_loader = DataLoader(scaled_val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
            val_loaders.append(scaled_val_loader)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True)

    model_kwargs = collect_model_kwargs(args, train_loader)
    model = MODELS[model_name](class_names=cityscapes_train_dataset.class_names,
                               ignore_index=cityscapes_train_dataset.ignore_index,
                               visualizer_kwargs=cityscapes_train_dataset.visualizer_kwargs,
                               **model_kwargs
                               )

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=True)
        # checkpoint = torch.load(checkpoint, map_location=torch.device('cuda'))
        # model.model.load_state_dict(checkpoint["model_state"])
        # model.to('cuda')

    checkpoint_callbacks = []
    for i in range(len(val_loaders)):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=f'val_{i}_mIoU',
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
        max_steps=max_steps,
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

    # args = [
    #     "--train_dataset", "cityscapes",
    #     "--val_dataset", "cityscapes",
    #     "--max_epochs", "1",
    #     "--batch_size", "8",
    #     "--num_gpus", "1",
    #     "--num_workers", "16",
    #     "--val_every_n_epochs", "1",
    #     "--wandb_group", "deeplabv3plus_cityscapes",
    #     "--exp_name", "deeplabv3plus_cityscapes_bs8_epoch350",
    #     "--backbone", "resnet101",
    #     "--lr", "0.1",
    #     "--crop_size", "768",
    #     "--batch_size", "16",
    #     "--output_stride", "16",
    #     "--data_root", "./data/cityscapes",
    #     "--debug"
    # ]
    # sys.argv.extend(args)

    main()
