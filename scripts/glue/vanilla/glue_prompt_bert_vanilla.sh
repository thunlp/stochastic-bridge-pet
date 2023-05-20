bsz=32

lr=1e-3
task=$task

for seed in 42 43 44; do
    save_path=checkpoint/prompt/bert/$task/vanilla/seed_${seed}
    if [ ! -d $save_path ]; then
        mkdir -p $save_path
    fi
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 src/run_glue_bert.py \
        --learning_rate $lr \
        --batch_size $bsz \
        --eval_step 1000 \
        --project_dim 1024 256 128 \
        --eval_batch_size 128 \
        --prompt_length 20 \
        --task_name $task \
        --model_type bert \
        --original_model_path ./pretrained_ckpt/bert \
        --disable_wandb \
        --seed $seed \
        --save \
        --save_path $save_path \
            2>&1 | tee ${save_path}/train.log
done
