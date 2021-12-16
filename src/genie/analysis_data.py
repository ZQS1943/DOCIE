import json
from utils import load_ontology
filename = "./data/wikievents/train_info"
with open(f'{filename}.jsonl','r') as reader:
    ontology_dict = load_ontology("KAIROS")



    role_set = set()
    for line in reader:
        doc  = json.loads(line)
        actual_event_mentions = []
        for event in doc['event_mentions']:
            if "Explode" in event['event_type']:
                print(event['event_type'])
                role_set.update(set(arg['role'] for arg in event['arguments']))
    print(role_set)