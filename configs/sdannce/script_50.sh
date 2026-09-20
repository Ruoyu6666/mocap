############
### RoPe ###
############

#---- cntnap2 -----
python trainers/skeletonMAE/pretrain_rope.py --dim_feat 192 --decoder_dim_feat 192 \
 --depth 7 --decoder_depth 1 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_cntnap2.pkl \
 --sliding_window 25 --batch_size 192 --epochs 15 --lr 5e-5 --weight_decay 5e-4 \
 --save_dir ./outputs/50/ --rope_ratio 1.0 >> 50_cntnap2.txt


python trainers/skeletonMAE/compute_representation_rope.py --dim_feat 192 --depth 7 \
  --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
  --dataset sdannce --path_to_data_dir ../data/sdannce/data_cntnap2.pkl \
  --sliding_window 10 --batch_size 192 --save_dir outputs/50 --rope_ratio 1.0\
  --model_path outputs/50/checkpoints/mae_checkpoint_epoch_10.pth \
  --config configs/sdannce/cntnap2.yaml --if_val True


######### fmr1 #########
#----- decoder = 2 -----
python trainers/skeletonMAE/pretrain_rope.py --dim_feat 192 --decoder_dim_feat 192 \
 --depth 6 --decoder_depth 2 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl \
 --sliding_window 25 --batch_size 128 --epochs 20 --lr 5e-5 --weight_decay 5e-4 \
 --save_dir ./outputs/50/ --rope_ratio 1.0 >> fmr1_decoder_2.txt


python trainers/skeletonMAE/compute_representation_rope.py --dim_feat 192 --depth 6 \
  --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
  --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl \
  --sliding_window 5 --batch_size 192 --save_dir outputs/50 --rope_ratio 1.0\
  --model_path outputs/50/checkpoints/mae_checkpoint_epoch_10.pth \
  --config configs/sdannce/fmr1.yaml --if_val True



--sliding_window 5 
--save_dir outputs/fmr1/50/RoPE
--model_path outputs/fmr1/50/RoPE/checkpoints/mae_checkpoint_epoch_5.pth 


################
### original ###
################
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 256 \
 --depth 6 --decoder_depth 1 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl  --lr 5e-5 \
 --sliding_window 25 --batch_size 128 --epochs 20 --save_dir ./outputs/50/\
 --ckpt_path 
# mask_ratio 0.75

python trainers/skeletonMAE/compute_representation.py --dim_feat 192 --depth 7 \
 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 --dataset sdannce \
 --path_to_data_dir ../data/sdannce/data_fmr1.pkl --sliding_window 10 --batch_size 128 \
 --save_dir outputs/50 --model_path outputs/fmr1/50/checkpoints/epoch_15.pth\
 --if_val True

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
 



# Calculated for swav training. First no augmentation, then with augmentation
python trainers/skeletonMAE/compute_representation.py --dim_feat 192 --depth 6 \
 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 --attn_drop_rate 0.02 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl --sliding_window 50 \
 --data_augment True --batch_size 128 --save_dir swav/finetune/others \
 --model_path outputs/fmr1/50/checkpoints/epoch_15.pth --if_val True






####################
### patch size 1 ###
####################
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 256 \
  --depth 6 --decoder_depth 1 --num_frames 50 --num_joints 18 \
  --patch_size 1 --t_patch_size 1 --attn_drop_rate 0.02 \
  --dataset sdannce --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 24 --mask_ratio 0.75 --batch_size 128 --epochs 20 \
  --lr 5e-5 --weight_decay 5e-4 \
  --save_dir ./outputs/fmr1/50/



###########################################
### first original, then augmented data ###
###########################################
python trainers/skeletonMAE/pretrain.py \
  --dim_feat 192 --decoder_dim_feat 256 --depth 6 --decoder_depth 1 \
  --num_frames 50 --num_joints 18 --patch_size 3  --t_patch_size 1 \
  --dataset sdannce --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 24 --mask_ratio 0.75 --batch_size 128 --epochs 20 \
  --lr 5e-5 --weight_decay 5e-4 --save_dir ./outputs/fmr1/50/ \
  --data_augment False 
  --data_augment True --ckpt_path /home/rguo_hpc/myfolder/mocap/outputs/fmr1/50/checkpoints/mae_checkpoint_epoch_10.pth


python trainers/skeletonMAE/compute_representation.py \
  --dim_feat 192 --depth 6 --num_frames 50 --num_joints 18 \
  --patch_size 3 --t_patch_size 1 --attn_drop_rate 0.02 \
  --dataset sdannce --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 5 --batch_size 128 --save_dir ./outputs/fmr1/50/ \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/fmr1/50/checkpoints/mae_aug_epoch_20.pth\
  --if_val True