#!/usr/bin/env bash
set -e 
set -x 


GPU='1'
DATASET='KAIROS'
CKPT_NAME='baseline_bert_crf_'${DATASET}

python src/train_baseline_bert_crf.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=4 \
    --gpus ${GPU} \
    --data_file 'preprocessed/'${CKPT_NAME}