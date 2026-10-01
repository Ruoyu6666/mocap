######################
### Hyperparameter ###
######################

# 1. mask ratio = 0.85
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 256 \
 --depth 6 --decoder_depth 1 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
  --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl --sliding_window 24 \
  --mask_ratio 0.85 --batch_size 128 --epochs 20 --lr 5e-5 --save_dir ./outputs/50/
 
# 2. dim_feat = 256
python trainers/skeletonMAE/pretrain.py --dim_feat 256 --decoder_dim_feat 256 \
 --depth 6 --decoder_depth 1 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
  --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl --sliding_window 19 \
  --mask_ratio 0.8 --batch_size 128 --epochs 20 --lr 5e-5 --save_dir ./outputs/50/

# 4. depth 7
python trainers/skeletonMAE/pretrain.py --dim_feat 192  --decoder_dim_feat 192 \
 --depth 7 --decoder_depth 1 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
 --sliding_window 19 --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl \
  --mask_ratio 0.8 --batch_size 160 --epochs 20 --save_dir outputs/50/ \
  --ckpt_path outputs/50/checkpoints/mae_checkpoint_epoch_10.pth --data_augment True 

# 5. num_frames = 60
python trainers/skeletonMAE/pretrain.py \
  --dim_feat 192 --decoder_dim_feat 256 --depth 6 --decoder_depth 1 \
  --num_frames 60 --num_joints 18 --patch_size 3 --t_patch_size 1 \
  --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl  --lr 5e-5 \
  --sliding_window 25 --mask_ratio 0.75 --batch_size 128 --epochs 20 --save_dir ./outputs/60/ 


# 6. 22 keypoints with patch = 2
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 256 \
 --depth 6 --decoder_depth 1 --num_frames 50 --num_joints 22 --patch_size 2 --t_patch_size 1 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl  --lr 5e-5 \
 --sliding_window 24 --batch_size 128 --epochs 20 --save_dir ./outputs/50/
 

# 7. patch size 1
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 256 \
  --depth 6 --decoder_depth 1 --num_frames 50 --num_joints 18 \
  --patch_size 1 --t_patch_size 1 --attn_drop_rate 0.02 \
  --dataset sdannce --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 24 --mask_ratio 0.75 --batch_size 128 --epochs 20 \
  --lr 5e-5 --weight_decay 5e-4 --save_dir ./outputs/fmr1/50/


# Calculated for swav training. First no augmentation, then with augmentation
python trainers/skeletonMAE/compute_representation.py --dim_feat 192 --depth 6 \
 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 --attn_drop_rate 0.02 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl --sliding_window 50 \
 --data_augment True --batch_size 128 --save_dir swav/finetune/others \
 --model_path outputs/fmr1/50/checkpoints/epoch_15.pth --if_val True


###########################################
### 1. t_patch_size = 3, patch_size = 1 ###
###########################################
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 256 \
  --depth 7 --decoder_depth 1 --num_frames 240 --num_joints 18 \
  --patch_size 1 --t_patch_size 3 --attn_drop_rate 0.01 --dataset \
  --sdannce --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 60 --mask_ratio 0.8 --batch_size 64 --epochs 50 \
  --lr 1e-4 --weight_decay 5e-4 --save_dir ./outputs/ \

ckpt_path: /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18/tpatch3/mae_checkpoint_epoch_30.pth


###########################################
### 2. num_frames = 150，patch_size = 3 ###
###########################################
 --sliding_window 10 is better? longer 30 for example, worse? Frame level very slightly worse, 
 seqeuence level seems like too, with 10 it can reach 78% even, with 30 it seems a bit hard, maybe like 2% difference


###########################################
### 3. t_patch_size = 5, patch_size = 3 ###
###########################################
mask_ratio 0.75
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 256 \
 --depth 7 --decoder_depth 1 --num_frames 900 --num_joints 18 --patch_size 3 --t_patch_size 5 \
 --attn_drop_rate 0.05 --dataset sdannce \
 --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl --sliding_window 50 \
 --batch_size 32 --epochs 20 --lr 5e-5 --weight_decay 1e-3 --save_dir ./outputs/900/


  python trainers/skeletonMAE/compute_representation.py --dim_feat 192 --depth 7 \
  --num_frames 900 --num_joints 18 --patch_size 3 --t_patch_size 5 --attn_drop_rate 0.05 \
  --dataset sdannce --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 50 --batch_size 64 --save_dir ./outputs/ \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/mae_checkpoint_epoch_15.pth\
  --if_val True
  
###########################################
### 3. t_patch_size = 9, patch_size = 1 ###
###########################################
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 256 \
  --depth 7 --decoder_depth 1 --num_frames 450 --num_joints 18 --patch_size 1 \
  --t_patch_size 9  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 63 --mask_ratio 0.8 --batch_size 64 \
  --epochs 50 --weight_decay 5e-4 --save_dir ./outputs/ \

/home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18/tpatch9/mae_checkpoint_epoch_20.pth


###########################################
### 4. t_patch_size = 3, patch_size = 3 ###
###########################################
python trainers/skeletonMAE/pretrain.py --dim_feat 192  --decoder_dim_feat 256 \
  --depth 6 --decoder_depth 1 --num_frames 300 --num_joints 18 \
  --patch_size 3 --t_patch_size 3 --attn_drop_rate 0.01 --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 60 --mask_ratio 0.8 --batch_size 64 --epochs 50 \
  --lr 1e-4 --weight_decay 5e-4 --save_dir ./outputs/ 

/home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18_patch/mae_checkpoint_epoch_30.pth\