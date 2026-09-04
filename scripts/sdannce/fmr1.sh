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
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --save_dir ./outputs/ \

ckpt_path: /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18/tpatch3/mae_checkpoint_epoch_30.pth





#### mil ###
python main.py --dataset sdannce --subseq_len 4500 --if_interval True --instance_len 5 --lr 0.0001 
--batchsize 32
--if_extract_feature False
--weight_decay 0.001
--feats_size 128
--embed 128


python main.py --dataset sdannce_kinematic --subseq_len 4500 --if_interval True --instance_len 3



###########################################
### 2. num_frames = 150，patch_size = 3 ###
###########################################
 --sliding_window 10 is better? longer 30 for example, worse? Frame level very slightly worse, seqeuence level seems like too, 
   10 it can reach 78% even, with 30 it seems a bit hard, maybe like 2% difference




###########################################
### 3. t_patch_size = 5, patch_size = 3 ###
###########################################
mask_ratio 0.75
python trainers/skeletonMAE/pretrain.py \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 7 \
  --decoder_depth 1 \
  --num_frames 900 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 5 \
  --attn_drop_rate 0.05 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 50 \
  --batch_size 32 \
  --epochs 20 \
  --lr 5e-5 \
  --weight_decay 1e-3 \
  --save_dir ./outputs/900/


  python trainers/skeletonMAE/compute_representation.py \
  --dim_feat 192 \
  --depth 7 \
  --num_frames 900 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 5 \
  --attn_drop_rate 0.05 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 50 \
  --batch_size 64 \
  --save_dir ./outputs/ \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/mae_checkpoint_epoch_15.pth\
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