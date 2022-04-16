#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=sentence_selection_nb_debug
DATA_NAME=preprocessed_sentence_selection_nb_debug
GPU=1
MODEL=constrained-gen
CKPT_NUM=('0' '1' '2' '3' '4' '5')

for zone in "${CKPT_NUM[@]}"
do
rm -rf checkpoints/${CKPT_NAME}-pred 
python train.py --model=$MODEL --ckpt_name=${CKPT_NAME}-pred \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${zone}.ckpt \
     --dataset=KAIROS \
     --eval_only \
     --train_batch_size=4 \
     --eval_batch_size=4 \
     --accumulate_grad_batches=4 \
     --gpus ${GPU} \
     --data_file=${DATA_NAME}

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/wikievents/test_no_ontology.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/coref/test.jsonlines \
--head-only \
--coref 
done 


