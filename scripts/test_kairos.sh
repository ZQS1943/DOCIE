#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=s
MODEL=constrained-gen

# rm -rf checkpoints/${CKPT_NAME}-pred 
# python train.py --model=$MODEL --ckpt_name=${CKPT_NAME}-pred \
#      --load_ckpt=checkpoints/${CKPT_NAME}/epoch=4-v0.ckpt \
#      --dataset=KAIROS \
#      --eval_only \
#      --train_file=data/wikievents/train.jsonl \
#      --val_file=data/wikievents/dev.jsonl \
#      --test_file=data/wikievents/test.jsonl \
#      --train_batch_size=4 \
#      --eval_batch_size=4 \
#      --learning_rate=3e-5 \
#      --accumulate_grad_batches=4 \
#      --num_train_epochs=3 \
#      --gpus 2

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/wikievents/test.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/coref/test.jsonlines \
--head-only \
--coref 
