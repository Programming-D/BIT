CUDA_VISIBLE_DEVICES=1 python train.py --path /home/yijunl/segmentation/Train_Sets/CT_MR/ --save_path ../Weight/  --epochs 5000 --batch_size 1 --pretrain True --weight_path ../Weight/epoch_611_dice_0.6150_.pth