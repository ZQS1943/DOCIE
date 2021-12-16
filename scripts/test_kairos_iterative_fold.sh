#!/usr/bin/env bash
set -e 
set -x 
CKPT_NUM=('5')
fold_num=6
# for fold_num in {0..9}
# do
for zone in "${CKPT_NUM[@]}"
do
python test_iterative_test.py \
     --ckpt_name=fold_check_${fold_num}-pred \
     --load_ckpt=checkpoints/fold_check_${fold_num}/epoch_${zone}.ckpt \
     --num_iterative_epochs 6 \
     --gpus 3 \
     --data_file=preprocessed_fold_${fold_num} \
     --fold_num ${fold_num} \
     # --use_info
done 
# done
# checkpoints/iterative_fast_5e-5/epoch_5.ckpt
