
experiment=experiment1

python run_test.py \
    --path_to_data_dir /home/rguo_hpc/myfolder/code/mocap/data/data_CLB.pkl \
    --dataset mocap \
    --embedsum True \
    --fast_inference False \
    --batch_size 128 \
    --model gen_hiera \
    --input_size 300 1 30 \
    --stages 7 \
    --q_strides "1,1,1" \
    --mask_unit_attn True \
    --patch_kernel 3 1 30 \
    --init_embed_dim 128 \
    --init_num_heads 2 \
    --out_embed_dims 128 \
    --distributed \
    --num_frames 300 \
    --pin_mem \
    --num_workers 8 \
    --fill_holes False \
    --output_dir outputs/mocap/${experiment}


cd hierAS-eval
            
nr_submissions=$(ls ../outputs/mice/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))

parallel --line-buffer \
    python evaluator.py \
        --task mabe_mice --output-dir results \
        --labels ../data/MABe22/mouse_triplets_test_labels.npy \
        --submission ../outputs/mice/${experiment}/test_submission_{}.npy \
    ::: "${files[@]}"
