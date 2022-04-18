#!/usr/bin/env bash
set -e 
set -x 

GPU='1'
DATASET='KAIROS'
CKPT_NAME='bert_12'
EPOCH='5'

python test_bert_crf.py \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${EPOCH}.ckpt \
     --gpus ${GPU} \
     --dataset=${DATASET} \
     --data_file 'preprocessed/'${CKPT_NAME}