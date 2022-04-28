import numpy as np
def convert_results(result):
    lines = result.splitlines()
    ans = []
    for line in lines:
        tmp = line.split(', ')
        tmp = [float(x[-5:]) for x in tmp]
        ans.append(tmp)
    new_ans = []
    new_ans.extend(ans[0])
    new_ans.extend(ans[2])
    new_ans.extend(ans[1])
    new_ans.extend(ans[3])
    print(new_ans)
    return new_ans

# results = ["""Role identification: P: 74.59, R: 72.47, F: 73.51
# Role: P: 68.56, R: 66.61, F: 67.57
# Coref Role identification: P: 75.87, R: 73.71, F: 74.77
# Coref Role: P: 69.84, R: 67.85, F: 68.83""",
# """Role identification: P: 77.11, R: 73.00, F: 75.00
# Role: P: 71.48, R: 67.67, F: 69.53
# Coref Role identification: P: 78.61, R: 74.42, F: 76.46
# Coref Role: P: 72.80, R: 68.92, F: 70.80""",
# """Role identification: P: 77.84, R: 73.00, F: 75.34
# Role: P: 71.02, R: 66.61, F: 68.74
# Coref Role identification: P: 78.60, R: 73.71, F: 76.08
# Coref Role: P: 71.78, R: 67.32, F: 69.48"""]

# results = ["""Role identification: P: 74.36, R: 72.11, F: 73.22
# Role: P: 67.95, R: 65.90, F: 66.91
# Coref Role identification: P: 75.82, R: 73.53, F: 74.66
# Coref Role: P: 69.41, R: 67.32, F: 68.35""",
# """Role identification: P: 75.75, R: 72.11, F: 73.89
# Role: P: 69.78, R: 66.43, F: 68.06
# Coref Role identification: P: 77.05, R: 73.36, F: 75.16
# Coref Role: P: 70.90, R: 67.50, F: 69.15""",
# """Role identification: P: 77.76, R: 72.65, F: 75.11
# Role: P: 71.10, R: 66.43, F: 68.69
# Coref Role identification: P: 78.52, R: 73.36, F: 75.85
# Coref Role: P: 71.86, R: 67.14, F: 69.42"""]

results = ["""Role identification: P: 77.26, R: 71.23, F: 74.12
Role: P: 71.10, R: 65.54, F: 68.21
Coref Role identification: P: 78.61, R: 72.47, F: 75.42
Coref Role: P: 72.25, R: 66.61, F: 69.32""",
"""Role identification: P: 77.26, R: 71.23, F: 74.12
Role: P: 71.10, R: 65.54, F: 68.21
Coref Role identification: P: 78.61, R: 72.47, F: 75.42
Coref Role: P: 72.25, R: 66.61, F: 69.32""",
"""Role identification: P: 77.26, R: 71.23, F: 74.12
Role: P: 71.10, R: 65.54, F: 68.21
Coref Role identification: P: 78.61, R: 72.47, F: 75.42
Coref Role: P: 72.25, R: 66.61, F: 69.32"""]


ans = []
for result in results:
    ans.append(convert_results(result))

avg_ans = np.array(ans)
avg_ans = np.average(avg_ans, axis=0)
print('&'.join('{:.2f}'.format(x) for x in avg_ans))
