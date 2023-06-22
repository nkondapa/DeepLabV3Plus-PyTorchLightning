#python train_deeplabv3plus.py --eval_dataset pascal --max_epochs 50 --batch_size 16 --num_gpus 4 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_training
#python train_deeplabv3plus.py --eval_dataset pascal --max_epochs 100 --batch_size 16 --num_gpus 4 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_training
#python train_deeplabv3plus.py --eval_dataset pascal --max_epochs 184 --batch_size 16 --num_gpus 4 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_training
#python train_deeplabv3plus.py --eval_dataset pascal --max_epochs 92 --batch_size 8 --num_gpus 4 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_training --exp_name deeplabv3plus_bs8_epch92
#python train_deeplabv3plus_finetuning.py --eval_dataset pascal --max_epochs 40 --batch_size 8 --num_gpus 4 --num_workers 16 --val_every_n_epochs 3 --wandb_group deeplabv3plus_ft_training --exp_name deeplabv3plus_ft_bs8_epch92_40 --checkpoint checkpoints/deeplabv3plus_bs8_epch92/model_checkpoint_deeplabv3plus_bs8_epch92.ckpt
#python train_deeplabv3plus_finetuning.py --eval_dataset pascal --max_epochs 92 --batch_size 8 --num_gpus 4 --num_workers 16 --val_every_n_epochs 3 --wandb_group deeplabv3plus_ft_training --exp_name deeplabv3plus_ft_bs8_epch92_92 --checkpoint checkpoints/deeplabv3plus_bs8_epch92/model_checkpoint_deeplabv3plus_bs8_epch92.ckpt
#python train_deeplabv3plus_cityscapes.py --eval_dataset cityscapes --max_epochs 1 --batch_size 16 --num_gpus 1 --num_workers 0  --val_every_n_epochs 1 --wandb_group deeplabv3plus_cityscapes --exp_name deeplabv3plus_cityscapes_bs16_epoch100
#python train_deeplabv3plus_cityscapes.py --eval_dataset cityscapes --max_epochs 275 --batch_size 16 --num_gpus 1 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_cityscapes --exp_name deeplabv3plus_cityscapes_bs16_epoch275
python train_deeplabv3plus_cityscapes.py --eval_dataset cityscapes --max_epochs 970 --batch_size 8 --num_gpus 4 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_cityscapes --exp_name deeplabv3plus_cityscapes_bs8_epoch970
