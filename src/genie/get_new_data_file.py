import argparse
import json

def get_new_data_file(args, epoch):
    original_data = f'{args.data_file}/test.jsonl'
    with open(original_data, 'r') as reader:
        for line in reader:
            print(line)
            break
    return 0    

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train-file',type=str,default='data/wikievents/train.jsonl')
    parser.add_argument('--val-file', type=str, default='data/wikievents/dev.jsonl')
    parser.add_argument('--test-file', type=str, default='data/wikievents/test.jsonl')
    parser.add_argument('--coref-dir', type=str, default='data/wikievents/coref')

    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='KAIROS')
    parser.add_argument('--mark-trigger', action='store_true', default=True)
    parser.add_argument('--data_file', default='../../preprocessed_type',type=str)
    args = parser.parse_args() 

    get_new_data_file(args,0)