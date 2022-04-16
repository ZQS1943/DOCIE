#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME='sentence_selection_info'
CKPT_NUM=('0' '1' '2' '3' '4' '5')

for zone in "${CKPT_NUM[@]}"
do
python train_sentence_selection.py \
     --ckpt_name=${CKPT_NAME}-pred \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${zone}.ckpt \
     --gpus 2 \
     --data_file=preprocessed_sentence_selection_info \
     --eval_only
     # --use_info
done 
# checkpoints/iterative_fast_5e-5/epoch_5.ckpt
