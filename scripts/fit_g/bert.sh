#!/bin/bash

GPUS_PER_NODE=${GPU_NUM}
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=$(($RANDOM % 99000 + 1000))
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=./data/pretrain_data/bert/wiki_bookcorpus_text_sentence
SAVE_PATH=./checkpoint/${bridge_type}/bert/
CHECKPOINT_PATH=./pretrained_ckpt/bert/
VOCAB_FILE=${CHECKPOINT_PATH}/bert-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ ! -d ${SAVE_PATH} ]; then
	mkdir -p ${SAVE_PATH}
	mkdir ${SAVE_PATH}/src
fi

cp src/fit_g_bert.py ${SAVE_PATH}/src
cp scripts/fit_g/bert.sh ${SAVE_PATH}/src
cp -r src/models ${SAVE_PATH}/src
cp -r src/dataloader ${SAVE_PATH}/src

MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       src/fit_g_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 16 \
       --global-batch-size 128 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 500000 \
       --lr-decay-iters 490000 \
       --save $SAVE_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file ${CHECKPOINT_PATH}/bert-vocab.txt \
       --data-impl mmap \
       --split 973,26,1 \
       --lr 0.001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 0 \
       --clip-grad 1.0 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --model bridge_bert \
       --bridge-type ${bridge_type} \
       --project-dim 1024 256 32 \
       --only-train-bridge \
       --continue-pretraining \
       --num-workers 4 \
       --bridge-weight 1.0 2>&1 |& tee ${SAVE_PATH}/train.log
