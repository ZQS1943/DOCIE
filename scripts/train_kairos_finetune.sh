#!/usr/bin/env bash
set -e 
set -x 

CKPT_NAME='gen-KAIROS-WFinetune'
# CKPT_NAME='gen-KAIROS'
rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
python train.py --model=constrained-gen --ckpt_name=${CKPT_NAME} \
    --dataset=KAIROS \
    --train_file=data/wikievents/train.jsonl \
    --val_file=data/wikievents/dev.jsonl \
    --test_file=data/wikievents/test.jsonl \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=8 \
    --num_train_epochs=1 \
    --mark_trigger \
    --coref_dir=data/wikievents/coref
    --load_ckpt=checkpoints/gen-KAIROS-finetune/default/version_62/checkpoints/epoch=9-step=859.ckpt
