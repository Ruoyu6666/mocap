python trainers/skeletonMAE/pretrain.py \
  --dim_in 2 \
  --dim_feat 128 \
  --decoder_dim_feat 64 \
  --depth 2 \
  --decoder_depth 1 \
  --num_heads 4 \
  --mlp_ratio 4 \
  --num_frames 71 \
  --num_joints 13 \
  --patch_size 1 \
  --t_patch_size 1 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.01 \
  --drop_path_rate 0.0 \
  --dataset eyetract \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/eye/eyetrack.pkl \
  --sliding_window 71 \
  --mask_ratio 0.7 \
  --view_invariant True \
  --batch_size 4 \
  --epochs 100 \
  --lr 2e-4 \
  --blr 1e-3 \
  --min_lr 0.0 \
  --weight_decay 2e-4 \
  --save_dir ./outputs/eyetrack/


python trainers/skeletonMAE/compute_representation.py \
  --dim_in 2 \
  --dim_feat 128 \
  --depth 2 \
  --num_heads 4 \
  --mlp_ratio 4 \
  --num_frames 71 \
  --num_joints 13 \
  --patch_size 1 \
  --t_patch_size 1 \
  --drop_rate 0.0 \
  --attn_drop_rate 0.01 \
  --drop_path_rate 0.0 \
  --path_to_data_dir /home/rguo_hpc/myfolder/data/eye/eyetrack.pkl \
  --sliding_window 71 \
  --batch_size 4 \
  --save_dir ./outputs/eyetrack/ \
  --fast_inference True \
  --model_path /home/rguo_hpc/myfolder/mocap/outputs/eyetrack/checkpoints/mae_checkpoint_epoch_100.pth\
  --dataset eyetract


python main.py --dataset eyetract --lr 0.0001 --batchsize 4 --embed 128 --weight_decay 0.0001 --if_extract_feature False
python main.py --dataset eyetract --lr 0.0001 --batchsize 4 --feats_size 26 --embed 128 --weight_decay 0.0001