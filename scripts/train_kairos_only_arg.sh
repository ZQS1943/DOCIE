#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'
CKPT_NAME='only_arg_all_args'
rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
python train_only_arg.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=4 \
    --gpus 1 \
    --learning_rate 3e-5 \
    --lambda_value 1 \
    --lambda_value_3 0 \
    --data_file=preprocessed_only_arg_all_args
    # --load_ckpt=checkpoints/gen-KAIROS-finetune/default/version_59/checkpoints/epoch=0-step=42.ckpt
