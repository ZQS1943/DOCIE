import json
with open('./event_role_ACE.json','r') as reader:
    ace = json.loads(reader.read())
    print(ace.keys())