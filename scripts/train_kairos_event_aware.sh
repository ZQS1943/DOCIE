#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'
CKPT_NAME='m-20-10-3e-5'
rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
python train_event_aware.py --model=constrained-gen --ckpt_name=${CKPT_NAME} \
    --dataset=KAIROS \
    --train_file=data/wikievents/train.jsonl \
    --val_file=data/wikievents/dev.jsonl \
    --test_file=data/wikievents/test.jsonl \
    --train_batch_size=2 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=8 \
    --num_train_epochs=10 \
    --mark_trigger \
    --coref_dir=data/wikievents/coref \
    --gpus 2 \
    --lambda_value=10
    # --load_ckpt=checkpoints/gen-KAIROS-finetune/default/version_59/checkpoints/epoch=0-step=42.ckpt
