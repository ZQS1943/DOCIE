#!/usr/bin/env bash
set -e 
set -x 


GPU='0'
TRG_DIS='40'

SEED=('42')
# CKPT_NAME='iterative_decoding_3e-5_'${SEED}
for s in "${SEED[@]}"
do
CKPT_NAME='iterative_decoding_info_'${s}
CKPT_NUM=('5')
# CKPT_NUM=('5')

for zone in "${CKPT_NUM[@]}"
do
python test_iterative_test.py \
     --ckpt_name=${CKPT_NAME}-pred \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${zone}.ckpt \
     --num_iterative_epochs 6 \
     --gpus ${GPU} \
     --data_file=preprocessed/preprocessed_${CKPT_NAME} \
     --use_info \
     --trg_dis ${TRG_DIS} \
     --seed ${s}
done 
# checkpoints/iterative_fast_5e-5/epoch_5.ckpt

done