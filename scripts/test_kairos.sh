#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=tmp
MODEL=constrained-gen

rm -rf checkpoints/${CKPT_NAME}-pred 
python train.py --model=$MODEL --ckpt_name=${CKPT_NAME}-pred \
     --load_ckpt=checkpoints/new_gen-KAIROS-Multitask-best/default/version_57/checkpoints/epoch=2-step=512.ckpt \
     --dataset=KAIROS \
     --eval_only \
     --train_file=data/wikievents/train.jsonl \
     --val_file=data/wikievents/dev.jsonl \
     --test_file=data/wikievents/test.jsonl \
     --train_batch_size=4 \
     --eval_batch_size=4 \
     --learning_rate=3e-5 \
     --accumulate_grad_batches=4 \
     --num_train_epochs=3 \
     --gpus 1 

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/wikievents/test.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/coref/test.jsonlines \
--head-only \
--coref 

     
     # --load_ckpt=checkpoints/gen-KAIROS-multievent/default/version_59/checkpoints/epoch=1-step=173.ckpt \
     # --load_ckpt=checkpoints/new_gen-KAIROS-Multitask/default/version_57/checkpoints/epoch=1-step=341.ckpt \