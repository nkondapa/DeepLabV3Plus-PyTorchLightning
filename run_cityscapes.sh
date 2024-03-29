python train_deeplabv3plus_cityscapes.py \
--train_dataset cityscapes \
--val_dataset cityscapes \
--max_steps 80000 \
--batch_size 8 \
--num_gpus 4 \
--num_workers 16  \
--val_every_n_epochs 3 \
--wandb_group deeplabv3plus_cityscapes \
--exp_name deeplabv3plus_new_cityscapes_bs8_steps80k_lr1e-1_rand_scale \
--backbone resnet101 \
--lr 0.1  \
--crop_size 768 \
--output_stride 16 \
--data_root ./data/cityscapes \
#--val_scales 0.75 1.00 1.25 \
