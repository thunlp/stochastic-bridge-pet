#!/bin/bash

lr=1e-3
wd=0
gc=$(( 8 / ${GPU_NUM} ))

save_path=checkpoint/${bridge_type}/deberta/
data_path=data/pretrain_data/deberta/wiki_bookcorpus_text_sentence
checkpoint_path=pretrained_ckpt/deberta-xlarge

if [ ! -d ${save_path} ]; then
    mkdir -p ${save_path}/src
fi

cp -r src/models ${save_path}/src
cp src/fit_g_deberta.py ${save_path}/src
cp -r src/dataloader ${save_path}/src
cp scripts/fit_g/deberta.sh ${save_path}/src

python3 -m torch.distributed.launch --master_port $(( $RANDOM % 99000 + 1000 )) --nproc_per_node ${GPU_NUM} src/fit_g_deberta.py \
    --output_dir ${save_path} \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --bridge_type ${bridge_type} \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $gc \
    --learning_rate $lr \
    --weight_decay $wd \
    --max_grad_norm 1.0 \
    --max_steps 100000 \
    --warmup_ratio 0.1 \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 10000 \
    --data_path ${data_path} \
    --pretrained_path ${checkpoint_path} \
    --ddp_find_unused_parameters False \
    --dataloader_num_workers 4 \
    --fp16 \
    --report_to none \
    --disable_tqdm True 2>&1 |& tee ${SAVE_PATH}/train.log

