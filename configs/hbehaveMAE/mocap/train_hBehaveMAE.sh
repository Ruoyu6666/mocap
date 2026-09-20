
# number of gpus to use
GPUS=$1

common_args="--dataset mocap \
    --path_to_data_dir /home/rguo_hpc/myfolder/mocap/data/mocap/data_CLB.pkl \
    --batch_size 64 \
    --model hbehavemae \
    --input_size 300 1 30 \
    --stages 7 \
    --q_strides 1,1,1 \
    --mask_unit_attn True \
    --patch_kernel 3 1 30 \
    --init_embed_dim 192 \
    --init_num_heads 2 \
    --out_embed_dims 192 \
    --epochs 100 \
    --num_frames 300 \
    --decoding_strategy multi \
    --decoder_embed_dim 192 \
    --decoder_depth 1 \
    --decoder_num_heads 1 \
    --pin_mem \
    --num_workers 8 \
    --sliding_window 99 \
    --blr 1.6e-4 \
    --warmup_epochs 40 \
    --masking_strategy random \
    --mask_ratio 0.75 \
    --clip_grad 0.02 \
    --checkpoint_period 5 \
    --fill_holes False \
    --data_augment False \
    --norm_loss True \
    --seed 0 \
    --output_dir outputs/checkpoints/hbehave/experiment2 \
    --log_dir logs/mocap/experiment1"

python trainers/behaveMAE/pretrain_behaveMAE.py $common_args


if [[ $GPUS == 1 ]]; then
    OMP_NUM_THREADS=1 python run_pretrain.py $common_args
else
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --node_rank 0 --master_addr=127.0.0.1 --master_port=2999 \
        run_pretrain.py --distributed $common_args
fi
