#!/usr/bin/env bash
set -e 
set -x 


GPU='2'
SEED=('42')

for zone in "${SEED[@]}"
do
CKPT_NAME='bert_'${zone}

python train_bert_crf.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=4 \
    --gpus ${GPU} \
    --learning_rate 3e-5 \
    --data_file=preprocessed/preprocessed_${CKPT_NAME} \
    --seed ${zone} \
    # --seed 12 \
done    


for zone in "${SEED[@]}"
do
CKPT_NAME='bert_ace_'${zone}

python train_bert_crf.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=4 \
    --gpus ${GPU} \
    --learning_rate 3e-5 \
    --data_file=preprocessed/preprocessed_${CKPT_NAME} \
    --seed ${zone} \
    --dataset=ACE \
    --role_num=25
done   
