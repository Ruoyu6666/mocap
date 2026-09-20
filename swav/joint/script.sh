python main.py --shuffle True --num_prototypes 64 --batch_size 256
python main.py --protocol compute_representations \
--checkpoint_path pretrain_output/checkpoint_epoch_5.pt --num_prototypes 64 --batch_size 256