#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'
CKPT_NAME='iterative_fast_5e_5_40_info_s'
# rm -rf checkpoints/${CKPT_NAME}
CKPT_NUM=('2' '3' '4' '5')

for zone in "${CKPT_NUM[@]}"
do
# does not use informative mentions 
python train_iterative_decoding_fast.py \
    --ckpt_name=${CKPT_NAME} \
    --gpus 1 \
    --data_file=preprocessed_iterative_fast_5e_5_40_info \
    --eval_only \
    # --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${zone}.ckpt \
    
done
# Event-level identification: P: 48.35
# Event-level : P: 48.35
# gold arg num: 561
# Role identification: P: 66.45, R: 54.01, F: 59.59
# Role: P: 58.33, R: 47.42, F: 52.31
# Coref Role identification: P: 67.98, R: 55.26, F: 60.96
# Coref Role: P: 59.87, R: 48.66, F: 53.69