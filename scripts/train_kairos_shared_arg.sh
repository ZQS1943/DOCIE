#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'
CKPT_NAME='try'
rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
python train_shared_arg.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=2 \
    --gpus 2 \
    --learning_rate 5e-5 \
    --lambda_value 1 \
    --lambda_value_3 0 \
    --data_file=preprocessed_m_copy_3_move
    # --load_ckpt=checkpoints/gen-KAIROS-finetune/default/version_59/checkpoints/epoch=0-step=42.ckpt
