bsz=2

lr=1e-4
task=$task
bridge_weight=${bridge_weight}
bridge_type=${bridge_type}
num_shot=${num_shot}

declare -A metrics=( ['MNLI']='Accuracy' ['SST-2']='Accuracy' ['QNLI']='Accuracy' ['RTE']='Accuracy' ['MRPC']="F1" ['QQP']="F1" ['CoLA']="MatthewsCorrelation" )
metric=${metrics[$task]}

for fewshot_seed in 42 43 44 45 46; do
    save_path=checkpoint/adapter/deberta/fewshot/$task/${bridge_type}/bridge_weight_${bridge_weight}/seed_${fewshot_seed}
    if [ ! -d $save_path ]; then
        mkdir -p $save_path
    fi
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 src/run_glue_deberta.py \
        --output_dir ${save_path} \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --evaluation_strategy steps \
        --eval_steps 50 \
        --save_strategy steps \
        --save_steps 50 \
        --logging_strategy steps \
        --logging_steps 5 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --max_steps 1000 \
        --lr_scheduler_type constant \
        --learning_rate $lr \
        --per_device_train_batch_size $bsz \
        --per_device_eval_batch_size 16 \
        --bridge_weight ${bridge_weight} \
        --project_dim 1024 256 128 \
        --prompt_length 0 \
        --task_name $task \
        --model_type bridge_deberta \
        --bridge_type ${bridge_type} \
        --load_bridge_path checkpoint/${bridge_type}/deberta/checkpoint-100000/pytorch_model.bin \
        --original_model_path ./pretrained_ckpt/deberta-xlarge \
        --dataloader_num_workers 2 \
        --fp16 \
        --report_to none \
        --apply_adapter \
        --adapter_r 8 \
        --load_best_model_at_end \
        --fewshot \
        --fewshot_seed $fewshot_seed \
        --num_shot $num_shot \
        --metric_for_best_model $metric \
        --greater_is_better True \
            | tee ${save_path}/train.log
done
