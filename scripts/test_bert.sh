#!/usr/bin/env bash
set -e 
set -x 


GPU='0'
SEED=('42')
# CKPT_NAME='iterative_decoding_3e-5_'${SEED}
for s in "${SEED[@]}"
do
CKPT_NAME='bert_ace_'${s}
CKPT_NUM=('5')
# CKPT_NUM=('5')

for zone in "${CKPT_NUM[@]}"
do
python test_bert_crf.py \
     --ckpt_name=${CKPT_NAME}-pred \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${zone}.ckpt \
     --gpus ${GPU} \
     --data_file=preprocessed/preprocessed_${CKPT_NAME} \
     --role_num=25\
     --dataset="ACE"\
     --seed ${s}
done 
# checkpoints/iterative_fast_5e-5/epoch_5.ckpt

done