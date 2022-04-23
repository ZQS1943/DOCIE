#!/usr/bin/env bash
set -e 
set -x 


GPU='0'
TRG_DIS='40'
DATASET='KAIROS'
CKPT_NAME='comparing_0.5_40_42'
EPOCH='5'

python src/test_bart_gen.py \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${EPOCH}.ckpt \
     --num_iterative_epochs 6 \
     --gpus ${GPU} \
     --trg_dis ${TRG_DIS} \
     --dataset ${DATASET} \
     --data_file 'preprocessed/'${CKPT_NAME}