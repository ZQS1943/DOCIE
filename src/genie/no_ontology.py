import json
from utils import load_ontology
filename = "./data/wikievents/test_info"
with open(f'{filename}.jsonl','r') as reader , open(f'{filename}_no_ontology.jsonl','w') as writer:
    ontology_dict = load_ontology("KAIROS")
    for line in reader:
        doc  = json.loads(line)
        actual_event_mentions = []
        for event in doc['event_mentions']:
            if event['event_type'] not in ontology_dict:
                print(event['event_type'])
                continue
            actual_event_mentions.append(event)
        doc['event_mentions'] = actual_event_mentions
        writer.write(json.dumps(doc) + '\n')