################
### original ###
################
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 256 \
 --depth 6 --decoder_depth 1 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl  --lr 5e-5 \
 --sliding_window 25 --batch_size 128 --epochs 20 --save_dir ./outputs/50/\
 --ckpt_path  # mask_ratio 0.75

python trainers/skeletonMAE/compute_representation.py --dim_feat 192 --depth 6 \
 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 --dataset sdannce \
 --path_to_data_dir ../data/sdannce/data_fmr1.pkl --sliding_window 5 --batch_size 128 \
 --save_dir outputs/sdannce/fmr1/50 --model_path outputs/sdannce/fmr1/50/checkpoints/epoch_15.pth\
 --if_val True


#---- chd8 ----
python trainers/skeletonMAE/pretrain.py --dim_feat 192 --decoder_dim_feat 192 \
 --depth 7 --decoder_depth 2 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_chd8.pkl  --lr 5e-5 \
 --sliding_window 25 --batch_size 192 --epochs 20 --save_dir ./outputs/50/

# sample by 2, each sequence len 2280
--num_frames 30 
# Try 8 layers


python trainers/skeletonMAE/compute_representation.py --dim_feat 192 --depth 7 \
 --num_frames 30 --num_joints 18 --patch_size 3 --t_patch_size 1 --dataset sdannce \
 --path_to_data_dir ../data/sdannce/data_chd8.pkl --sliding_window 15 --batch_size 256 \
 --save_dir outputs/30 --model_path outputs/30/checkpoints/mae_checkpoint_epoch_15.pth \
 --config ./configs/sdannce/chd8.yaml --if_val True 


###########
#### mil ##
###########
python main.py --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl  \
--config ../configs/sdannce/fmr1.yaml --if_interval True --instance_len 5  \
--if_extract_feature False --subseq_len 3000 --lr 0.0002


python main.py --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
 --config ../configs/sdannce/fmr1.yaml --if_interval True --instance_len 5  --batchsize 32  --if_extract_feature False 
 --lr 0.0001 
--if_extract_feature False 
--batchsize 32
--weight_decay 0.001
--feats_size 256
--embed 256

python main.py --dataset sdannce_kinematic --subseq_len 4500 --if_interval True --instance_len 3






############
### RoPe ###
############
#---- fmr1 ----
python trainers/skeletonMAE/pretrain_rope.py --dim_feat 192 --decoder_dim_feat 256 \
 --depth 6 --decoder_depth 2 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_fmr1.pkl --sliding_window 25 \
 --batch_size 128 --epochs 20 --lr 5e-5 --weight_decay 5e-4 \
 --save_dir ./outputs/50/ --rope_ratio 1.0


#---- cntnap2 -----
python trainers/skeletonMAE/pretrain_rope.py --dim_feat 192 --decoder_dim_feat 256 \
 --depth 7 --decoder_depth 1 --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 \
 --dataset sdannce --path_to_data_dir ../data/sdannce/data_cntnap2.pkl \
 --sliding_window 25 --batch_size 192 --epochs 15 --lr 5e-5 --weight_decay 5e-4 \
 --save_dir ./outputs/50/ --rope_ratio 1.0 >> cntnap2.txt


python trainers/skeletonMAE/compute_representation_rope.py --dim_feat 192 --depth 7 \
  --num_frames 50 --num_joints 18 --patch_size 3 --t_patch_size 1 --dataset sdannce \
  --path_to_data_dir ../data/sdannce/data_cntnap2.pkl --sliding_window 10 \
  --batch_size 192 --save_dir outputs/50 --rope_ratio 1.0\
  --model_path outputs/50/checkpoints/mae_checkpoint_epoch_10.pth \
  --config configs/sdannce/cntnap2.yaml --if_val True

#----- chd8 ----




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