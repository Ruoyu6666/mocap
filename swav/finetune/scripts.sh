##### With RoPe #####

#----------------
#--- sdannce ---
#----------------


#---  chd8 ---

#---  cntnap2 ---
python main.py --mode finetune_last_n --unfreeze_n 2 --batch_size 192\
 --encoder_ckpt ../../outputs/sdannce/cntnap2/checkpoints/mae_checkpoint_epoch_5.pth \
 --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_cntnap2.pkl \
 --config ../../configs/sdannce/cntnap2.yaml


python main.py --mode finetune_last_n --unfreeze_n 2 --batch_size 192 \
 --compute_representations --checkpoint_path swav_output/checkpoint_epoch_10.pt \
 --path_to_data_dir /home/rguo_hpc/myfolder/data/sdannce/data_cntnap2.pkl \
 --config ../../configs/sdannce/cntnap2.yaml --shuffle False --sliding_window 10
 

#---  fmr1 ---
python main.py --mode finetune_last_n --batch_size 128 \
 --encoder_ckpt ../../outputs/sdannce/fmr1/50/RoPE/mae_checkpoint_epoch_10.pth
# --sliding_window 25
# --semisup True --swav_weight 0.7 --cls_weight 0.3 --shuffle True # with semisup 


python main.py --mode finetune_last_n --encoder_module models.skeletonMAE.model.encoder_rope \
 --compute_representations --checkpoint_path swav_output/len50/rope/10ckpt/128batch/checkpoint_epoch_15.pt \
 --num_frames 50 --batch_size 128 --shuffle False


#--------------------------------
# ----- Supervised finetune -----
#--------------------------------
python main.py --mode finetune_last_n --encoder_module models.skeletonMAE.model.encoder_rope \
--encoder_ckpt ../../outputs/sdannce/fmr1/50/RoPE/mae_checkpoint_epoch_10.pth  --num_frames 50 \
--pairing_mode classification_only --unfreeze_n 4 --classifier_hidden_dim 192 --sliding_window 25


python main.py --mode finetune_last_n --encoder_module models.skeletonMAE.model.encoder_rope \
 --compute_representations --checkpoint_path swav_output/checkpoint_epoch_10.pt \
 --num_frames 50 --shuffle False --unfreeze_n 4 --classifier_hidden_dim 192


#---------------------------
# ----- Hyperparameter -----
#---------------------------
# Experiment 1: finetune last 2 layers
python main.py --mode finetune_last_n

# with seq_level loss
python main.py --mode finetune_last_n
python main.py --mode finetune_last_n --compute_representations --checkpoint_path ./swav_output/checkpoint_epoch_20.pt
# EXperiment 2: num_prototypes 64
python main.py --mode finetune_last_n --num_prototypes 64
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
