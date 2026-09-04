# Experiment 1: finetune last 2 layers
python main.py --mode finetune_last_n --num_prototypes 128

python main.py --mode finetune_last_n --compute_representations --representation_which raw \
    --checkpoint_path ./swav_output/finetune_last_2/checkpoint_epoch_15.pt --num_prototypes 128 --batch_size 128

# EXperiment 2: num_prototypes 64
python main.py --mode finetune_last_n --num_prototypes 64

python main.py --mode finetune_last_n --compute_representations --representation_which raw \
    --checkpoint_path ./swav_output/checkpoint_epoch_15.pt --num_prototypes 64 --batch_size 128

# Experiment 3: finetune last 4 layers
python main.py --mode finetune_last_n --num_prototypes  --unfreeze_n 4

python main.py --mode finetune_last_n --compute_representations --representation_which raw \
    --checkpoint_path ./swav_output/finetune_last_4/checkpoint_epoch_15.pt --num_prototypes  --batch_size 128  --unfreeze_n 4