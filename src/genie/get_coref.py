import json
coref_lines = []
with open('./data/wikievents/coref/train.jsonlines','r') as f:
    coref_lines.extend(f.readlines())

with open('./data/wikievents/coref/test.jsonlines','r') as f:
    coref_lines.extend(f.readlines())

coref_dict = {}
for line in coref_lines:
    line = json.loads(line)
    coref_dict[line['doc_key']] = line


for fold_num in range(10):
    file_name = f'./data/wikievents/10fold/fold_{fold_num}'
    test_lines = []
    with open(f'{file_name}/test.jsonl', 'r') as reader, open(f'{file_name}/test_coref.jsonl','w') as writer:
        for line in reader:
            line = json.loads(line)
            coref = coref_dict[line['doc_id']]
            writer.write(json.dumps(coref) + '\n')
            