#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=s_info
DATA_NAME=preprocessed_KAIROS_info
GPU=1
MODEL=constrained-gen
CKPT_NUM=('2' '3-v0' '3')

for zone in "${CKPT_NUM[@]}"
do
rm -rf checkpoints/${CKPT_NAME}-pred 
python train.py --model=$MODEL --ckpt_name=${CKPT_NAME}-pred \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch=${zone}.ckpt \
     --dataset=KAIROS \
     --eval_only \
     --train_batch_size=4 \
     --eval_batch_size=4 \
     --accumulate_grad_batches=4 \
     --gpus ${GPU} \
     --data_file=${DATA_NAME}

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/wikievents/test_info_no_ontology.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/coref/test.jsonlines \
--head-only \
--coref 
done 


