#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'
CKPT_NAME='dis_20_m_0_1'
rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
python train_event_aware.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=2 \
    --gpus 1 \
    --learning_rate 3e-5 \
    --lambda_value 0 \
    --lambda_value_3 1 \
    --data_file=preprocessed_event_aware_KAIROS
    # --load_ckpt=checkpoints/gen-KAIROS-finetune/default/version_59/checkpoints/epoch=0-step=42.ckpt
