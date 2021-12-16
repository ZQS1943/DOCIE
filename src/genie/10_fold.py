import json
import os
from utils import load_ontology
from sklearn.model_selection import KFold

data = []
dev_data = []
with open("./data/wikievents/test_no_ontology.jsonl", 'r') as test, open("./data/wikievents/train_no_ontology.jsonl", 'r') as train, open("./data/wikievents/dev_no_ontology.jsonl", 'r') as dev:
    for line in test:
        data.append(line.strip())
    for line in train:
        data.append(line.strip())
    for line in dev:
        dev_data.append(line.strip())
    # data.extend(dev)

kf = KFold(n_splits=10,shuffle=False)
file_dir = "./data/wikievents/10fold"
# for fold_num in range(10):
#     fold_dir = f'{file_dir}/fold_{fold_num}'
#     if not os.path.exists(fold_dir):
#         os.mkdir(fold_dir)
for fold_num, (train_index, test_index) in enumerate(kf.split(data)):
    fold_dir = f'{file_dir}/fold_{fold_num}'
    with open(f'{fold_dir}/train.jsonl', 'w') as writer:
        for index in train_index:
            writer.write(data[index] + '\n')
    with open(f'{fold_dir}/test.jsonl', 'w') as writer:
        for index in test_index:
            writer.write(data[index] + '\n')
    with open(f'{fold_dir}/dev.jsonl', 'w') as writer:
        for item in dev_data:
            writer.write(item + '\n')
            