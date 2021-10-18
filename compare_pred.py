import json

f1 = open("./checkpoints/tmp-pred/predictions.jsonl",'r')
lines_1 = f1.readlines()
f2 = open("./checkpoints/gen-KAIROS-m-s-pred/predictions.jsonl",'r')
lines_2 = f2.readlines()


assert len(lines_1) == len(lines_2)

for i in range(len(lines_1)):
    tmp_1 = json.loads(lines_1[i])
    pred_1 = tmp_1["predicted"]    
    tmp_2 = json.loads(lines_2[i])
    pred_2 = tmp_2["predicted"]
    gold = tmp_2["gold"]
    doc = tmp_2["doc_key"]
    if pred_1==gold and pred_2!= gold:
        print(doc)
        print(pred_1)        
        print(pred_2)
        print(gold)