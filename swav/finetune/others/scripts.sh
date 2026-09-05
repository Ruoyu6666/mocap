# Experiment 1: finetune last 2 layers
python main.py --mode finetune_last_n --num_prototypes 128 

python main.py --mode finetune_last_n --compute_representations --representation_which raw \
    --checkpoint_path ./swav_output/finetune_last_2/checkpoint_epoch_15.pt --num_prototypes 128 --batch_size 128


# EXperiment 2: num_prototypes 64
python main.py --mode finetune_last_n --num_prototypes 64

# Experiment 3: finetune last 4 layer: not better
python main.py --mode finetune_last_n --unfreeze_n 4 --compute_representations --representation_which raw \
    --checkpoint_path ./swav_output/checkpoint_epoch_10.pt --num_prototypes 128 --batch_size 128

# Experiment 4 finetune last 2 with encoder checkpointepoch_15
python main.py --mode finetune_last_n --num_prototypes 128  --unfreeze_n 2 --epochs 5 \
    --encoder_ckpt /home/rguo_hpc/myfolder/mocap/outputs/fmr1/50/checkpoints/epoch_15.pth

# Experiment 5: finetune last 1 layer with lr 5e-4
python main.py --mode finetune_last_n --num_prototypes 128 --unfreeze_n 1 --epochs 25 --lr 5e-4


# Experiment 6: finetune last 2 layers with batch size 32
python main.py --mode finetune_last_n --num_prototypes 128 --batch_size 32


# Experiment 7: init with GMM
python main.py --mode finetune_last_n --num_prototypes 128 --gmm_means_path ./others/gmm_centers.npy

python main.py --mode finetune_last_n --num_prototypes 128 --compute_representations --representation_which raw \
    --checkpoint_path ./swav_output/checkpoint_epoch_10.pt

 #--lr 5e-4