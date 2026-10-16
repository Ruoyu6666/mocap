# 30
python trainers/skeletonMAE/pretrain_rope.py --dataset mabe_mice --dim_in 2 \
 --dim_feat 192 --decoder_dim_feat 256 --depth 7 --decoder_depth 1 \
 --num_frames 30 --num_joints 12 --patch_size 3 --t_patch_size 1 \
 --path_to_data_dir ../data/mabe_mice/mouse_triplet_train.npy --sliding_window 30 \
 --epochs 20 --save_dir ./outputs/mabe/ --rope_ratio 1.0 --batch_size 96 --lr 5e-5 \
 --num_workers 8 --include_testdata False >> mabe30.txt

python trainers/skeletonMAE/compute_representation_rope.py --dataset mabe_mice --dim_in 2 \
 --dim_feat 192 --depth 7 --num_frames 30 --num_joints 12 --patch_size 3 --t_patch_size 1 \
 --path_to_data_dir ../data/mabe_mice/mouse_triplet_train.npy \
 --sliding_window 15 --batch_size 256 --save_dir ./outputs/mabe/ \
 --model_path /home/rguo_hpc/myfolder/mocap/outputs/mabe/checkpoints/mae_checkpoint_epoch_15.pth \
 --mode test

# 300
python trainers/skeletonMAE/pretrain_rope.py --dataset mabe_mice --dim_in 2 \
 --dim_feat 192 --decoder_dim_feat 256 --depth 7 --decoder_depth 1 \
 --num_frames 300 --num_joints 12 --patch_size 3 --t_patch_size 3 \
 --path_to_data_dir ../data/mabe_mice/mouse_triplet_train.npy --sliding_window 49 \
 --epochs 20 --save_dir ./outputs/mabe/ --rope_ratio 1.0 --batch_size 80 --lr 5e-5 \
 --num_workers 6 --include_testdata False
 


python trainers/skeletonMAE/compute_representation_rope.py --dataset mabe_mice --dim_in 2 \
 --dim_feat 192 --depth 7 --num_frames 300 --num_joints 12 --patch_size 3 --t_patch_size 3 \
 --path_to_data_dir ../data/mabe_mice/mouse_triplet_train.npy \
 --sliding_window 60 --batch_size 128 --save_dir ./outputs/mabe/ \
 --model_path /home/rguo_hpc/myfolder/mocap/outputs/mabe/trainonly/300/mae_checkpoint_epoch_15.pth \
 --mode test
 
