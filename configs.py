import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42). if seed=0, seed is not fixed.')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--window_width', type=int, default=0, help='window width')
parser.add_argument('--normalize', action='store_true', default=False, help='normalize signal based on mean/std of training samples')
parser.add_argument('--pretrain', action='store_true', default=False)

# optimization
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--trial', type=str, default='default', help='trial id')

# dataset and model
parser.add_argument('--model', type=str, default='EarlyFusion', choices=['BaselineCNN', 'DeepConvLSTM', 'LSTMConvNet', 'EarlyFusion', 'CAGE'])
parser.add_argument('--dataset', type=str, default='UCI_HAR', choices=['UCI_HAR', 'WISDM', 'Opportunity', 'USC_HAD', 'PAMAP2', 'mHealth', 'MobiAct'])
parser.add_argument('--no_clean', action='store_false', default=False)
parser.add_argument('--no_null', action='store_false', default=True)
parser.add_argument('--train_portion', type=float, default=1.0, help='use portion of trainset')
parser.add_argument('--model_path', type=str, default='save', help='path to save model')
parser.add_argument('--load_model', type=str, default='', help='load the pretrained model')
parser.add_argument('--lambda_cls', type=float, default=1.0, help='loss weight for classification loss')
parser.add_argument('--lambda_ssl', type=float, default=1.0, help='loss weight for reconstruction loss')
parser.add_argument('--proj_dim', type=int, default=64)

# contrastive learning arguments
parser.add_argument('--loss_type', type=str, default='default',
                    choices=['default', 'nt_xent', 'triplet'],
                    help='Type of contrastive loss to use')
parser.add_argument('--temperature', type=float, default=0.00,
                    help='Temperature parameter for NT-Xent loss')
parser.add_argument('--margin', type=float, default=0.00,
                    help='Margin for triplet loss')

# -----------------------------------------------------

# IDEA at testing 5 : encoder adding with skip-connection
parser.add_argument('--num_encoders', type=int, default=1, 
                    help='number of stacked encoders (default: 1)')
parser.add_argument('--use_skip', action='store_true', default=False,
                    help='use skip connections between encoders')

# -----------------------------------------------------

args = parser.parse_args()

# 기존 로직 유지
if args.pretrain:
    args.lambda_cls = 0.0
    args.lambda_ssl = 1.0

# contrastive learning을 위한 설정
if args.loss_type in ['nt_xent', 'triplet']:
    args.lambda_cls = 0.0
    args.lambda_ssl = 1.0

def dict_to_markdown(d, max_str_len=120):
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()