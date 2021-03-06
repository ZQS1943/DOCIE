#!/usr/bin/env bash
set -e 
set -x 


GPU='0'
TRG_DIS='40'
DATASET='KAIROS'
CKPT_NAME='eaae_no_at_'${TRG_DIS}'_'${DATASET}'_'${SEED}

python src/train_eaae_no_at.py \
    --ckpt_name=${CKPT_NAME} \
    --gpus ${GPU} \
    --accumulate_grad_batches=4 \
    --trg_dis ${TRG_DIS} \
    --dataset ${DATASET} \
    --data_file 'preprocessed/'${CKPT_NAME}