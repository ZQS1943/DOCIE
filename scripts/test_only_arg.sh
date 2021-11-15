#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=dis_20_m_move_1_0
DATA_NAME=preprocessed_KAIROS
GPU=4
MODEL=constrained-gen
CKPT_NUM=('1-v0' '2-v0' '2' '3' '4')

for zone in "${CKPT_NUM[@]}"
do
rm -rf checkpoints/${CKPT_NAME}-pred 
python train_only_arg.py --model=$MODEL --ckpt_name=${CKPT_NAME}-pred \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch=${zone}.ckpt \
     --dataset=KAIROS \
     --eval_only \
     --train_file=data/wikievents/train.jsonl \
     --val_file=data/wikievents/dev.jsonl \
     --test_file=data/wikievents/test.jsonl \
     --train_batch_size=4 \
     --eval_batch_size=4 \
     --accumulate_grad_batches=4 \
     --gpus ${GPU} \
     --data_file=${DATA_NAME}

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/wikievents/test.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/coref/test.jsonlines \
--head-only \
--coref 
done 
