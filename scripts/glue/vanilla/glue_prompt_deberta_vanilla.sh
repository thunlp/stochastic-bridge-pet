bsz=16

lr=1e-3
task=$task

for seed in 42 43 44; do
    save_path=checkpoint/prompt/deberta/$task/vanilla/seed_${seed}
    if [ ! -d $save_path ]; then
        mkdir -p $save_path
    fi
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 src/run_glue_deberta.py \
        --output_dir ${save_path} \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --save_strategy steps \
        --save_steps 10000 \
        --logging_strategy steps \
        --logging_steps 100 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --max_steps 50000 \
        --lr_scheduler_type constant \
        --learning_rate $lr \
        --per_device_train_batch_size $bsz \
        --per_device_eval_batch_size 32 \
        --project_dim 1024 256 128 \
        --prompt_length 20 \
        --task_name $task \
        --model_type deberta \
        --original_model_path ./pretrained_ckpt/deberta-xlarge \
        --dataloader_num_workers 2 \
        --fp16 \
        --seed $seed \
        --report_to none \
            | tee ${save_path}/train.log
done
