import argparse

def define_arguments(parser):
    parser.add_argument("--model", type=str, default='constrained-gen', choices=['gen','constrained-gen'])
    parser.add_argument("--dataset", type=str, default='KAIROS', choices=['RAMS', 'ACE', 'KAIROS'])
    parser.add_argument('--tmp_dir', type=str)
    parser.add_argument("--ckpt_name", default=None, type=str, help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--train_file", default='data/wikievents/train_no_ontology.jsonl', type=str, help="The input training file. If a data dir is specified, will look for the file there" + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument("--val_file", default='data/wikievents/dev_no_ontology.jsonl', type=str, help="The input evaluation file. If a data dir is specified, will look for the file there" + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument('--test_file', type=str, default='data/wikievents/test_no_ontology.jsonl')
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--coref_dir', type=str, default='data/wikievents/coref')
    parser.add_argument("--data_file", type=str, required=True, help='dir to cache the preprocessed data')
    parser.add_argument('--fold_num', type=int, default=0)


    parser.add_argument('--use_info', action='store_true', default=False, help='use informative mentions instead of the nearest mention.')
    parser.add_argument('--mark_trigger', default=True, action='store_true')
    parser.add_argument('--sample-gen', action='store_true', help='Do sampling when generation.')

    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--num_iterative_epochs", default=2, type=int, help='The number of iterative epochs.')
    parser.add_argument('--save_top_k', default=3, type=int, help="save top k ckpts.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    parser.add_argument("--gpus",type=str, default=1, help='-1 means train on all the gpus')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")


    parser.add_argument("--lambda_value", type=float, default=1, help="loss = loss_extraction + lambda_value * loss_dis, -1 means automatic")
    parser.add_argument("--alpha", type=float, default=1, help="loss = loss_extraction + alpha * loss_dis, -1 means automatic")
    parser.add_argument("--score_th", type=float, default=0.0)
    parser.add_argument("--trg_dis", type=int, default=40)
    parser.add_argument("--role_num", type=int, default=85)

def parse_arguments():
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    args = parser.parse_args()
    return args