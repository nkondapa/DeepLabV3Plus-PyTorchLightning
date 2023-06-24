#python train_deeplabv3plus_new.py \
#--train_dataset cityscapes \
#--val_dataset cityscapes \
#--max_epochs 350 \
#--batch_size 8 \
#--num_gpus 4 \
#--num_workers 16  \
#--val_every_n_epochs 3 \
#--wandb_group deeplabv3plus_cityscapes \
#--exp_name deeplabv3plus_new_cityscapes_bs8_epoch350 \
#--backbone resnet101 \
#--lr 0.1  \
#--crop_size 768 \
#--output_stride 16 \
#--data_root ./data/cityscapes

python train_deeplabv3plus_new.py \
--train_dataset cityscapes \
--val_dataset cityscapes \
--max_epochs 0 \
--batch_size 16 \
--val_batch_size 16 \
--num_gpus 1 \
--num_workers 16  \
--val_every_n_epochs 3 \
--wandb_group deeplabv3plus_cityscapes \
--exp_name deeplabv3plus_newrepo_cityscapes_val \
--backbone resnet101 \
--lr 0.1  \
--crop_size 768 \
--output_stride 16 \
--data_root ./data/cityscapes \
--checkpoint checkpoints/latest_deeplabv3plus_resnet101_cityscapes_os16.pth
