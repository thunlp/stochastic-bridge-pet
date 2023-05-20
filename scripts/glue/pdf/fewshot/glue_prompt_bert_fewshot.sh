bsz=2

lr=1e-3
task=$task
bridge_weight=${bridge_weight}
bridge_type=${bridge_type}
num_shot=${num_shot}

for fewshot_seed in 42 43 44 45 46; do
    save_path=checkpoint/prompt/bert/fewshot/$task/${bridge_type}/bridge_weight_${bridge_weight}/seed_${fewshot_seed}
    if [ ! -d $save_path ]; then
        mkdir -p $save_path
    fi
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 src/run_glue_bert.py \
        --learning_rate $lr \
        --batch_size $bsz \
        --bridge_weight $bridge_weight \
        --training_step 1000 \
        --eval_step 50 \
        --log_step 5 \
        --project_dim 1024 256 128 \
        --eval_batch_size 128 \
        --prompt_length 20 \
        --task_name $task \
        --model_type bridge_bert \
        --bridge_type ${bridge_type} \
        --load_bridge_path checkpoint/${bridge_type}/bert/iter_0500000/mp_rank_00/model_optim_rng.pt \
        --original_model_path ./pretrained_ckpt/bert \
        --disable_wandb \
        --save \
        --save_path $save_path \
        --fewshot \
        --num_shot ${num_shot} \
        --fewshot_seed ${fewshot_seed} \
        --disable_wandb \
            2>&1 | tee ${save_path}/train.log
done