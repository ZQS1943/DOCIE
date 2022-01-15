#!/usr/bin/env bash
set -e 
set -x 

GPU='0'
SEED=('42' '9' '12' '21')

for zone in "${SEED[@]}"
do
CKPT_NAME='iterative_decoding_info_'${zone}
python train_iterative_decoding.py \
    --ckpt_name=${CKPT_NAME} \
    --gpus ${GPU} \
    --learning_rate 3e-5 \
    --data_file=preprocessed/preprocessed_${CKPT_NAME} \
    --accumulate_grad_batches=4 \
    --use_info \
    --seed ${zone}
done

for zone in "${SEED[@]}"
do
CKPT_NAME='iterative_decoding_ace_'${zone}
python train_iterative_decoding.py \
    --ckpt_name=${CKPT_NAME} \
    --gpus ${GPU} \
    --learning_rate 3e-5 \
    --data_file=preprocessed/preprocessed_${CKPT_NAME} \
    --accumulate_grad_batches=4 \
    --seed ${zone} \
    --dataset "ACE"
done
