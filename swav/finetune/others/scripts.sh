# With RoPe
python main.py --mode finetune_last_n --encoder_module models.skeletonMAE.model.encoder_rope \
 --encoder_ckpt ../../outputs/fmr1/50/RoPE/mae_checkpoint_epoch_10.pth \
 --num_frames 50 --batch_size 128

python main.py --mode finetune_last_n --encoder_module models.skeletonMAE.model.encoder_rope \
 --compute_representations --checkpoint_path swav_output/checkpoint_epoch_15.pt \
 --num_frames 50 --batch_size 128


# Experiment 1: finetune last 2 layers
python main.py --mode finetune_last_n

# with seq_level loss
python main.py --mode finetune_last_n
python main.py --mode finetune_last_n --compute_representations --checkpoint_path ./swav_output/checkpoint_epoch_20.pt
# EXperiment 2: num_prototypes 64
python main.py --mode finetune_last_n --num_prototypes 64
# Experiment 3: finetune last 4 layer
python main.py --mode finetune_last_n --unfreeze_n 4 --num_prototypes 128 --batch_size 128
# Experiment 4 finetune last 2 with encoder checkpointepoch_15
python main.py --mode finetune_last_n --num_prototypes 128  --unfreeze_n 2 --epochs 5 \
    --encoder_ckpt /home/rguo_hpc/myfolder/mocap/outputs/fmr1/50/checkpoints/epoch_15.pth
# Experiment 5: finetune last 1 layer
python main.py --mode finetune_last_n --num_prototypes 128 --unfreeze_n 1
# Experiment 6: finetune last 2 layers with batch size 32
python main.py --mode finetune_last_n --num_prototypes 128 --batch_size 32
# Experiment 7: init with GMM
python main.py --mode finetune_last_n --num_prototypes 128 --gmm_means_path ./others/gmm_centers.npy
# Experiment 9: finetune last 3 layers
python main.py --mode finetune_last_n --num_prototypes 128 --unfreeze_n 3
# Expoeriment 10; num_prototypes 192
python main.py --mode finetune_last_n --num_prototypes 192 

# Experiment 8: batch size 128
python main.py --mode finetune_last_n --num_prototypes 128 --batch_size 128
--checkpoint_path ./swav_output/checkpoint_epoch_20.pt
python main.py --mode finetune_last_n --num_prototypes 128 --batch_size 128 \
    --compute_representations --checkpoint_path ./swav_output/checkpoint_epoch_20.pt
