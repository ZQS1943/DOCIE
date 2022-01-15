import json
from os import stat
from utils import load_ontology

from collections import defaultdict

cluster2sent = defaultdict(set)
for stage in ['train','dev','test']:
    with open(f'./data/wikievents/{stage}_info.jsonl','r') as data, open(f'./data/wikievents/coref/{stage}.jsonlines') as coref_file:
        for line, coref_line in zip(data, coref_file):
            ex = json.loads(line)
            coref = json.loads(coref_line)
            assert ex['doc_id'] == coref['doc_key']

            

            for event in ex['event_mentions']:
                for arg in event['arguments']:
                    for idx, cluster in enumerate(coref['clusters']):
                        if arg['entity_id'] in cluster:
                            cluster2sent[f"{ex['doc_id']}:{idx}"].add((event['trigger']['sent_idx'], arg['role'], event['event_type']))
            
            # if ex['doc_id'] == 'backpack_ied_10':
            #     sentence = set(x[0] for x in cluster2sent['backpack_ied_10:7'])
            #     for sen in sentence:
            #         print(f'[S{sen}]: {ex["sentences"][sen][1]}')

            #     for event in ex['event_mentions']:
            #         if event['trigger']['sent_idx'] in sentence:
            #             print(f"[S{event['trigger']['sent_idx']}]: {event['event_type']}", event['trigger']['text'], event['arguments'])

            #     print(set(x[1] for x in cluster2sent['backpack_ied_10:7']))
            #     print(sentence)
            #     assert 1==0

            # if ex['doc_id'] == 'scenario_en_kairos_74':
            #     sentence = set(x[0] for x in cluster2sent['scenario_en_kairos_74:2'])
            #     for sen in sentence:
            #         print(f'[S{sen}]: {ex["sentences"][sen][1]}')

            #     for event in ex['event_mentions']:
            #         if event['trigger']['sent_idx'] in sentence and len(event['arguments']):
            #             print(f"[S{event['trigger']['sent_idx']}]: {event['event_type']}", event['trigger']['text'], event['arguments'], len(event['arguments']))

            #     print(set(x[1] for x in cluster2sent['scenario_en_kairos_74:2']))
            #     print(sentence)
            #     assert 1==0

            if ex['doc_id'] == '9_VOA_EN_NW_2016.09.19.3515911':
                sentence = set(x[0] for x in cluster2sent['9_VOA_EN_NW_2016.09.19.3515911:3'])
                for sen in sentence:
                    print(f'[S{sen}]: {ex["sentences"][sen][1]}')

                for event in ex['event_mentions']:
                    if event['trigger']['sent_idx'] in sentence and len(event['arguments']):
                        print(f"[S{event['trigger']['sent_idx']}]: {event['event_type']}", event['trigger']['text'], event['arguments'], len(event['arguments']))

                print(set(x[1] for x in cluster2sent['9_VOA_EN_NW_2016.09.19.3515911:3']))
                print(sentence)
                assert 1==0

print(cluster2sent)
cluster2sent = list(cluster2sent.items())
print(cluster2sent)
print('-'*10)
cluster2sent = list(filter(lambda x:len(set(a[1] for a in x[1]))>5 and len(set(a[0] for a in x[1]))>5 and len(set(a[2] for a in x[1]))>5,cluster2sent))
for item in cluster2sent:
    print(item[0])
    print(set(x[1] for x in item[1]))
# print(cluster2sent, len(cluster2sent))
