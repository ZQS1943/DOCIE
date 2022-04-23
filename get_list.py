import os

ckpts = []

for item in os.walk('./checkpoints'):
    ckpt_name = item[0][len('./checkpoints/'):]
    if ckpt_name.startswith('comparing_'):
        print(ckpt_name)
        if 'ace' not in ckpt_name:
            if '40' in ckpt_name:
                ckpts.append(ckpt_name)

print(' '.join(f"'{ckpt}'" for ckpt in ckpts))