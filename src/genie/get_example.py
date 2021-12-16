import json
from os import stat
from utils import load_ontology
filename = "./data/wikievents/test_info"
cnt = 0
total_cnt =0
with open(f'{filename}.jsonl','r') as reader:
    ontology_dict = load_ontology("KAIROS")
    for line in reader:
        doc  = json.loads(line)
        id2entity = {entity['id']:entity for entity in doc['entity_mentions']}
        for event in doc['event_mentions']:
            total_cnt += 1
            starts = [event['trigger']['start']]
            for arg in event['arguments']:
                starts.append(id2entity[arg['entity_id']]['start'])
            starts = sorted(starts)
            dis = starts[-1] - starts[0]
            # if dis > 300 and len(event['arguments'])>=3 and 'Die' in event['event_type']:
            #     print(event)
            #     print(starts)
            #     # tokens = [f"({x[0]}, {x[1]})" for x in zip(doc['tokens'], range(len(doc['tokens'])))]
            #     # print(' '.join(tokens))
            #     assert 1==0
            print(dis)
            if dis < 300:
                cnt += 1
print(cnt / total_cnt)