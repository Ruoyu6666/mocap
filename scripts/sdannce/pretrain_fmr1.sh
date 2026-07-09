python trainers/skeletonMAE/pretrain.py \
  --dim_in 3 \
  --dim_feat 192 \
  --decoder_dim_feat 256 \
  --depth 6 \
  --decoder_depth 1 \
  --num_heads 8 \
  --mlp_ratio 4 \
  --num_frames 240 \
  --num_joints 23 \
  --patch_size 1 \
  --t_patch_size 3 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.01 \
  --drop_path_rate 0.0 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 60 \
  --sampling_rate 1 \
  --interp_holes False \
  --mask_ratio 0.8 \
  --view_invariant True \
  --num_workers 8 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --blr 1e-3 \
  --min_lr 0.0 \
  --weight_decay 5e-4 \
  --log_interval 100 \
  --save_dir ./outputs/ \
  --if_test False \
  --job pretrain \
  --ckpt_path



python trainers/skeletonMAE/compute_representation.py \
  --dim_in 3 \
  --dim_feat 192 \
  --depth 6 \
  --num_heads 8 \
  --mlp_ratio 4 \
  --num_frames 150 \
  --num_joints 23 \
  --patch_size 1 \
  --t_patch_size 3 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.0 \
  --drop_path_rate 0.0 \
  --dataset sdannce \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl \
  --sliding_window 50 \
  --sampling_rate 1 \
  --view_invariant True \
  --num_workers 8 \
  --batch_size 64 \
  --epochs 60 \
  --log_interval 100 \
  --save_dir ./outputs/ \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/fmr1/61/mae_epoch_30_3600_150.pth\
  --if_val True



python main.py --dataset sdannce --subseq_len 2000  -if_extract_feature False
--lr 0.001 --batchsize 32
--embed 128 
--weight_decay 0.0001
--batchsize 32


2000: 60.74