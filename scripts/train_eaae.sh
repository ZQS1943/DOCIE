#!/usr/bin/env bash
set -e 
set -x 


GPU='1'
TRG_DIS='40'
alpha='0.5'
DATASET='KAIROS'
CKPT_NAME='eaae_'${TRG_DIS}'_'${alpha}'_'${DATASET}

python src/train_eaae.py \
    --ckpt_name=${CKPT_NAME} \
    --gpus ${GPU} \
    --accumulate_grad_batches=4 \
    --alpha=${alpha} \
    --trg_dis ${TRG_DIS} \
    --dataset ${DATASET} \
    --data_file 'preprocessed/'${CKPT_NAME}