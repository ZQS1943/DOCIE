#!/usr/bin/env bash
set -e 
set -x 


GPU='3'
TRG_DIS='70'

SEED=('42' '9' '12' '21')
alpha='0.5'

for zone in "${SEED[@]}"
do
CKPT_NAME='comparing_0.5_'${TRG_DIS}'_'${zone}

python train_comparing.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=2 \
    --gpus ${GPU} \
    --learning_rate 3e-5 \
    --data_file=preprocessed/preprocessed_${CKPT_NAME} \
    --accumulate_grad_batches=4 \
    --alpha=${alpha} \
    --seed ${zone} \
    --trg_dis ${TRG_DIS} \
    --use_info
done


for zone in "${SEED[@]}"
do
CKPT_NAME='comparing_ace_'${zone}
python train_comparing_dataset.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=2 \
    --gpus ${GPU} \
    --learning_rate 3e-5 \
    --data_file=preprocessed/preprocessed_${CKPT_NAME} \
    --accumulate_grad_batches=4 \
    --alpha=${alpha} \
    --seed ${zone} \
    --trg_dis ${TRG_DIS} \
    --dataset "ACE"
done