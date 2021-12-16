#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'
CKPT_NAME='iterative_fast_5e_5_200_info'
# rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
python train_iterative_decoding_fast.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=4 \
    --gpus 1 \
    --learning_rate 5e-5 \
    --data_file=preprocessed_iterative_fast_5e_5_200_info \
    --accumulate_grad_batches=4 \
    # --load_ckpt=checkpoints/iterative_tag_other_finetune/epoch_2.ckpt \
    # --eval_only

# Event-level identification: P: 48.35
# Event-level : P: 48.35
# gold arg num: 561
# Role identification: P: 66.45, R: 54.01, F: 59.59
# Role: P: 58.33, R: 47.42, F: 52.31
# Coref Role identification: P: 67.98, R: 55.26, F: 60.96
# Coref Role: P: 59.87, R: 48.66, F: 53.69