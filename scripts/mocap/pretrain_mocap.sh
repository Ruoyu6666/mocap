python trainers/skeletonMAE/pretrain.py \
    --job pretrain \
    --path_to_data_dir /home/rguo_hpc/myfolder/data/mocap/data_FL2.pkl  \
    --sliding_window 60 \
    --batch_size 32

python trainers/skeletonMAE/compute_representation.py \
    --job compute_representations \
    --path_to_data_dir /home/rguo_hpc/myfolder/data/mocap/data_FL2_v0.pkl  \
    --sliding_window 60 \
    --batch_size 32 \
    --model_path /home/rguo_hpc/myfolder/mocap/outputs/checkpoints/FL2/whole_vi_checkpoint_30.pth \
    --if_val True
