# normal config t_patch_size = 3, patch_size = 1
python trainers/skeletonMAE/pretrain.py \
  --dim_in 3 \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 7 \
  --decoder_depth 1 \
  --num_heads 8 \
  --mlp_ratio 4 \
  --num_frames 240 \
  --num_joints 18 \
  --patch_size 1 \
  --t_patch_size 3 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.01 \
  --drop_path_rate 0.0 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 60 \
  --mask_ratio 0.8 \
  --view_invariant True \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --blr 1e-3 \
  --min_lr 0.0 \
  --weight_decay 5e-4 \
  --save_dir ./outputs/ \



python trainers/skeletonMAE/compute_representation.py \
  --dim_in 3 \
  --dim_feat 192 \
  --depth 7 \
  --num_heads 8 \
  --mlp_ratio 4 \
  --num_frames 240 \
  --num_joints 18 \
  --patch_size 1 \
  --t_patch_size 3 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 60 \
  --sampling_rate 1 \
  --view_invariant True \
  --num_workers 8 \
  --batch_size 64 \
  --log_interval 100 \
  --save_dir ./outputs/ \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18/tpatch3/mae_checkpoint_epoch_30.pth\
  --if_val True







# patch = 3, t_patch=1
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
  --drop_rate 0.0 \
  --attn_drop_rate 0.01 \
  --drop_path_rate 0.0 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 50 \
  --mask_ratio 0.8 \
  --view_invariant True \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --blr 1e-3 \
  --min_lr 0.0 \
  --weight_decay 5e-4 \
  --save_dir ./outputs/ \
  --ckpt_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/mae_checkpoint_epoch_5.pth



python trainers/skeletonMAE/compute_representation.py \
  --dim_feat 192 \
  --depth 7 \
  --num_heads 8 \
  --mlp_ratio 4 \
  --num_frames 250 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 1 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.01 \
  --drop_path_rate 0.0 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 50 \
  --sampling_rate 1 \
  --view_invariant True \
  --num_workers 8 \
  --batch_size 64 \
  --log_interval 100 \
  --save_dir ./outputs/ \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/mae_checkpoint_epoch_5.pth\
  --if_val True

/home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18/tpatch1/mae_checkpoint_epoch_20.pth



python main.py --dataset sdannce --subseq_len 4500 --if_extract_feature False --lr 0.001
--batchsize 32 
--lr 0.001
--embed 128 
--weight_decay 0.0001




# Try t_patch_size = 9
python trainers/skeletonMAE/pretrain.py \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 7 \
  --decoder_depth 1 \
  --num_frames 450 \
  --num_joints 18 \
  --patch_size 1 \
  --t_patch_size 9 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.0 \
  --drop_path_rate 0.0 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 63 \
  --mask_ratio 0.8 \
  --view_invariant True \
  --batch_size 64 \
  --epochs 50 \
  --weight_decay 5e-4 \
  --save_dir ./outputs/ \





#ä With patch_size = 3
python trainers/skeletonMAE/pretrain.py \
  --dim_in 3 \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 6 \
  --decoder_depth 1 \
  --num_heads 8 \
  --mlp_ratio 4 \
  --num_frames 300 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 3 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.01 \
  --drop_path_rate 0.0 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 60 \
  --mask_ratio 0.8 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --blr 1e-3 \
  --min_lr 0.0 \
  --weight_decay 5e-4 \
  --log_interval 100 \
  --save_dir ./outputs/  >> log_patch.txt


python trainers/skeletonMAE/compute_representation.py \
  --dim_in 3 \
  --dim_feat 192 \
  --depth 6 \
  --num_heads 8 \
  --mlp_ratio 4 \
  --num_frames 300 \
  --num_joints 18 \
  --patch_size 3 \
  --t_patch_size 3 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.01 \
  --drop_path_rate 0.0 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 60 \
  --batch_size 64 \
  --log_interval 100 \
  --save_dir ./outputs/ \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/18_patch/mae_checkpoint_epoch_30.pth\
  --if_val True