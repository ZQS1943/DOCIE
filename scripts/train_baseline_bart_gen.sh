#!/usr/bin/env bash
set -e 
set -x 

GPU='0'
DATASET='KAIROS'
CKPT_NAME='baseline_bart_gen_'${DATASET}

python src/train_baseline_bart_gen.py \
    --ckpt_name=${CKPT_NAME} \
    --dataset=${DATASET} \
    --gpus ${GPU} \
    --data_file 'preprocessed/'${CKPT_NAME}