###########################################
### 1. t_patch_size = 3, patch_size = 1 ###
###########################################
python trainers/skeletonMAE/pretrain.py \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 7 \
  --decoder_depth 1 \
  --num_frames 240 \
  --num_joints 18 \
  --patch_size 1 \
  --t_patch_size 3 \
  --attn_drop_rate 0.01 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 60 \
  --mask_ratio 0.8 \
  --view_invariant True \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --save_dir ./outputs/ \

ckpt_path: /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18/tpatch3/mae_checkpoint_epoch_30.pth



#########--###############################
### 2. num_frames = 50，patch_size = 3 ###
##########################################
python trainers/skeletonMAE/pretrain.py \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 6 \
  --decoder_depth 1 \
  --num_frames 50 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 1 \
  --attn_drop_rate 0.02 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 25 \
  --mask_ratio 0.75 \
  --view_invariant True \
  --batch_size 128 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --save_dir ./outputs/50/ \
  --ckpt_path


python trainers/skeletonMAE/compute_representation.py \
  --dim_feat 192 \
  --depth 6 \
  --num_frames 50 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 1 \
  --attn_drop_rate 0.01 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 5 \
  --batch_size 128 \
  --save_dir ./outputs/50/ \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/50patch3/checkpoints/mae_checkpoint_epoch_20.pth\
  --if_val True


# mil
python main.py --dataset sdannce --subseq_len 4500 --if_interval True --instance_len 5
--if_extract_feature False 
--batchsize 32
--lr 0.0002
--weight_decay 0.0002
--feats_size 128
--embed 128



###########################################
### 3. t_patch_size = 1, patch_size = 3 ###
###########################################
python trainers/skeletonMAE/pretrain.py \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 7 \
  --decoder_depth 1 \
  --num_heads 8 \
  --mlp_ratio 4 \
  --num_frames 250 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 1 \
  --attn_drop_rate 0.01 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 50 \
  --mask_ratio 0.8 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --save_dir ./outputs/ \
  --ckpt_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/mae_checkpoint_epoch_5.pth


python trainers/skeletonMAE/compute_representation.py \
  --dim_feat 192 \
  --depth 7 \
  --num_frames 250 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 1 \
  --attn_drop_rate 0.01 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 50 \
  --batch_size 64 \
  --save_dir ./outputs/ \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18/tpatch1/mae_checkpoint_epoch_20.pth\
  --if_val True

###########################################
### 3. t_patch_size = 9, patch_size = 1 ###
###########################################
python trainers/skeletonMAE/pretrain.py \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 7 \
  --decoder_depth 1 \
  --num_frames 450 \
  --num_joints 18 \
  --patch_size 1 \
  --t_patch_size 9 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 63 \
  --mask_ratio 0.8 \
  --batch_size 64 \
  --epochs 50 \
  --weight_decay 5e-4 \
  --save_dir ./outputs/ \

/home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18/tpatch9/mae_checkpoint_epoch_20.pth


###########################################
### 4. t_patch_size = 3, patch_size = 3 ###
###########################################
python trainers/skeletonMAE/pretrain.py \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 6 \
  --decoder_depth 1 \
  --num_frames 300 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 3 \
  --attn_drop_rate 0.01 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 60 \
  --mask_ratio 0.8 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --save_dir ./outputs/ 

/home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18_patch/mae_checkpoint_epoch_30.pth\