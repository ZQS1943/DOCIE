#!/usr/bin/env bash
set -e 
set -x 
CKPT_NUM=('0' '1' '1-v0' '2' '2-v0' '2-v1')

for fold_num in {0..9}
do
for zone in "${CKPT_NUM[@]}"
do
rm -rf checkpoints/fold_${fold_num}_normal-pred
python train.py --ckpt_name=fold_${fold_num}_normal-pred \
     --load_ckpt=checkpoints/fold_${fold_num}_normal/epoch=${zone}.ckpt \
     --eval_only \
     --gpus 2 \
     --data_file=preprocessed_fold_${fold_num}_normal \
     --fold_num ${fold_num} \

python src/genie/scorer.py --gen-file=checkpoints/fold_${fold_num}_normal-pred/predictions.jsonl \
--test-file=data/wikievents/10fold/fold_${fold_num}/test.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/10fold/fold_${fold_num}/test_coref.jsonl \
--head-only \
--coref 
done 
done