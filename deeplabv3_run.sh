#python train_deeplabv3plus.py --eval_dataset pascal --max_epochs 50 --batch_size 16 --num_gpus 4 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_training
#python train_deeplabv3plus.py --eval_dataset pascal --max_epochs 100 --batch_size 16 --num_gpus 4 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_training
#python train_deeplabv3plus.py --eval_dataset pascal --max_epochs 184 --batch_size 16 --num_gpus 4 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_training
python train_deeplabv3plus.py --eval_dataset pascal --max_epochs 92 --batch_size 8 --num_gpus 4 --num_workers 16  --val_every_n_epochs 3 --wandb_group deeplabv3plus_training --exp_name deeplabv3plus_bs8_epch92
