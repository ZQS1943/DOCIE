#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME='two_event_simicolon'
CKPT_NUM=('5')
# CKPT_NUM=('5')

for zone in "${CKPT_NUM[@]}"
do
python test_two_event.py \
     --ckpt_name=${CKPT_NAME}-pred \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${zone}.ckpt \
     --gpus 0 \
     --data_file=preprocessed/preprocessed_two_event_simicolon \
     --score_th 0.0
     # --use_info
done 
# checkpoints/iterative_fast_5e-5/epoch_5.ckpt
