bsz=32

lr=1e-4
task=$task
bridge_weight=${bridge_weight}
bridge_type=${bridge_type}

for seed in 42 43 44; do
    save_path=checkpoint/bias/bert/$task/${bridge_type}/bridge_weight_${bridge_weight}/seed_${seed}
    if [ ! -d $save_path ]; then
        mkdir -p $save_path
    fi
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 src/run_glue_bert.py \
        --learning_rate $lr \
        --batch_size $bsz \
        --bridge_weight $bridge_weight \
        --eval_step 1000 \
        --project_dim 1024 256 32 \
        --eval_batch_size 128 \
        --prompt_length 0 \
        --apply_bias \
        --task_name $task \
        --model_type bridge_bert \
        --bridge_type ${bridge_type} \
        --load_bridge_path checkpoint/${bridge_type}/bert/iter_0200000/mp_rank_00/model_optim_rng.pt \
        --original_model_path ./pretrained_ckpt/bert \
        --disable_wandb \
        --seed $seed \
        --save \
        --save_path $save_path \
            2>&1 | tee ${save_path}/train.log
done
