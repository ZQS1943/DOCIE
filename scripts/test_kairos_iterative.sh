#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME='debug_iter'
CKPT_NUM=('5')

for zone in "${CKPT_NUM[@]}"
do
python test_iterative_test_tmp.py \
     --ckpt_name=${CKPT_NAME}-pred \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${zone}.ckpt \
     --num_iterative_epochs 6 \
     --gpus 2 \
     --data_file=preprocessed_iterative_fast_5e_5 \
     # --use_info
done 
# checkpoints/iterative_fast_5e-5/epoch_5.ckpt
