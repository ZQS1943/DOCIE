from unittest import result


ALPHA = ['0.5', '0.4','0.3','1','0.7','10','0.1','0.6']
DIS = ['100','40','50','60','70','80']



def get_results(lines):
    # print(lines)
    results = {}
    for line in lines:
        tmp = line.split(': ')
        # print(tmp)
        results[tmp[0]] = [float(tmp[2][:-3]), float(tmp[3][:-3]), float(tmp[4])]
        # print(line.split(': '))
    # print(results)
    return results


with open('./eval_result','r') as f:
    content = f.read()

content = content.split('--------------------------------------------------------------------------------\n')[:-1]

para_results = {}

for model in content:
    first_line  = model.splitlines()[0]
    print(first_line)
    epoch_num = first_line[-6]
    
    alpha = 1
    for a in ALPHA:
        if '_'+a in first_line:
            alpha = a
            break
    dis = 0
    for d in DIS:
        if '_'+d in first_line:
            dis = d
            break 
    print(epoch_num, alpha, dis)

    # print(model)
    steps = model.split('start scoring')
    for step in steps[1:]:
        # print(step)
        lines = step.splitlines()
        step = lines[1][-len('_predictions.jsonl') - 1]
        lines = list(filter(lambda x: 'Role' in x, lines))
        results = get_results(lines)
        # assert 1==0
    # assert 1==0
        para_results[(epoch_num, alpha, dis, step, first_line)] = results
print(para_results)
para_results_list = sorted(list(para_results.items()),key = lambda x:sum(x[1][s][-1] for s in x[1]))
print(para_results_list)
# print(models[0])
# print(content)
