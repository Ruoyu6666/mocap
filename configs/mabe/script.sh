python trainers/skeletonMAE/pretrain_rope.py --dataset mabe_mice --dim_in 2 \
 --dim_feat 192 --decoder_dim_feat 256 --depth 7 --decoder_depth 1 \
 --num_frames 60 --num_joints 12 --patch_size 1 --t_patch_size 3 \
 --path_to_data_dir ../data/mabe_mice/mouse_triplet_train.npy --sliding_window 30 \
 --epochs 20 --save_dir ./outputs/mabe/ --rope_ratio 1.0 --batch_size 128 \
 --num_workers 6 --include_testdata False  >> mabe.txt
 

python trainers/skeletonMAE/compute_representation_rope.py --dataset mabe_mice --dim_in 2 \
 --dim_feat 192 --depth 7 --num_frames 60 --num_joints 12 --patch_size 1 --t_patch_size 3 \
 --path_to_data_dir ../data/mabe_mice/mouse_triplet_train.npy \
 --sliding_window 30 --batch_size 128 --save_dir ./outputs/mabe/ \
 --model_path outputs/mabe/checkpoints/mae_checkpoint_epoch_20.pth --mode test
 
