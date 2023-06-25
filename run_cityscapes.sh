python train_deeplabv3plus_new.py \
--train_dataset cityscapes \
--val_dataset cityscapes \
--max_steps 80000 \
--batch_size 8 \
--num_gpus 4 \
--num_workers 16  \
--val_every_n_epochs 3 \
--wandb_group deeplabv3plus_cityscapes \
--exp_name deeplabv3plus_new_cityscapes_bs8_steps80k_lr1e-2 \
--backbone resnet101 \
--lr 0.01  \
--crop_size 768 \
--output_stride 16 \
--data_root ./data/cityscapes \
