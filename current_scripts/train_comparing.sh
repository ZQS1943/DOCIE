#!/usr/bin/env bash
set -e 
set -x 


GPU='0'
TRG_DIS='40'

# SEED=('42' '9' '25')
# alpha=('0.1' '0.5' '1')
# SEED=('9' '25')
# alpha=('0.5' '1')
SEED=('25')
alpha=('0.5' '1')

for zone in "${SEED[@]}"
do
for al in "${alpha[@]}"
do
CKPT_NAME='comparing_'${TRG_DIS}'_'${al}'_'${zone}

python train_comparing.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=2 \
    --gpus ${GPU} \
    --learning_rate 3e-5 \
    --data_file=preprocessed/preprocessed_${CKPT_NAME} \
    --accumulate_grad_batches=4 \
    --alpha=${al} \
    --seed ${zone} \
    --trg_dis ${TRG_DIS}
done
done


# for zone in "${SEED[@]}"
# do
# CKPT_NAME='comparing_ace_'${zone}
# python train_comparing_dataset.py \
#     --ckpt_name=${CKPT_NAME} \
#     --train_batch_size=2 \
#     --gpus ${GPU} \
#     --learning_rate 3e-5 \
#     --data_file=preprocessed/preprocessed_${CKPT_NAME} \
#     --accumulate_grad_batches=4 \
#     --alpha=${alpha} \
#     --seed ${zone} \
#     --trg_dis ${TRG_DIS} \
#     --dataset "ACE"
# done