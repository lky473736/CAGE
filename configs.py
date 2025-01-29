import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()


######## clustering
parser.add_argument('--clustering_method', type=str, default='kmeans',
                    choices=['kmeans', 'dbscan', 'spectral', 'gmm', 'fastcluster', 'birch'],
                    help='Clustering method to use')

parser.add_argument('--gmm_components', type=int, default=2,
                    help='Number of components for Gaussian Mixture Model')

parser.add_argument('--fastcluster_linkage', type=str, default='average',
                    choices=['ward', 'complete', 'average', 'single'],
                    help='Linkage method for hierarchical clustering')

parser.add_argument('--birch_threshold', type=float, default=0.5,
                    help='Threshold for Birch clustering')

parser.add_argument('--birch_branching_factor', type=int, default=50,
                    help='Branching factor for BIRCH clustering')

parser.add_argument('--dbscan_eps', type=float, default=0.5,
                    help='DBSCAN eps parameter')
parser.add_argument('--dbscan_min_samples', type=int, default=5,
                    help='DBSCAN min_samples parameter')

parser.add_argument('--spectral_affinity', type=str, default='rbf',
                    choices=['rbf', 'nearest_neighbors'],
                    help='Affinity type for spectral clustering')

####### ResNet - transformer-encoder
parser.add_argument('--num_resnet_blocks', type=int, default=3,
                    help='Number of ResNet blocks')
parser.add_argument('--resnet_channels', type=int, default=64,
                    help='Number of channels in ResNet blocks')


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

# PCA arguments
parser.add_argument('--use_pca', action='store_true', default=False,
                    help='Whether to use PCA for dimension reduction')
parser.add_argument('--pca_components', type=int, default=64,
                    help='Number of PCA components (should be less than embedding dim)')

# Encoder selection arguments
parser.add_argument('--encoder_type', type=str, default='default',
                   choices=['default', 'transformer', 
                            'unet', 'resnet_transformer', 
                            'se', 'se_transformer'],
                   help='Type of encoder to use')

parser.add_argument('--se_reduction', type=int, default=4,
                    help='Reduction ratio for SE block')
parser.add_argument('--multi_scale_kernels', type=str, default='3,5,7',
                    help='Comma separated kernel sizes for multi-scale convolution')

# Transformer encoder specific arguments
parser.add_argument('--num_heads', type=int, default=8,
                    help='Number of attention heads for transformer')
parser.add_argument('--ff_dim', type=int, default=256,
                    help='Feed-forward dimension in transformer')
parser.add_argument('--num_transformer_blocks', type=int, default=3,
                    help='Number of transformer blocks')
parser.add_argument('--transformer_dropout', type=float, default=0.1,
                    help='Dropout rate for transformer')

# dataset and model
parser.add_argument('--model', type=str, default='CAGE', choices=['BaselineCNN', 'DeepConvLSTM', 'LSTMConvNet', 'EarlyFusion', 'CAGE'])
parser.add_argument('--dataset', type=str, default='UCI_HAR', 
                    choices=['UCI_HAR', 'WISDM', 'Opportunity', 'USC_HAD', 'PAMAP2', 'mHealth', 'MobiAct', 
                            'MobiFall', 'SisFall', 'UMAFall'])
parser.add_argument('--no_clean', action='store_false', default=False)
parser.add_argument('--no_null', action='store_false', default=True)
parser.add_argument('--train_portion', type=float, default=1.0, help='use portion of trainset')
parser.add_argument('--model_path', type=str, default='save', help='path to save model')
parser.add_argument('--load_model', type=str, default='', help='load the pretrained model')
parser.add_argument('--lambda_cls', type=float, default=0.0, help='loss weight for classification loss')
parser.add_argument('--lambda_ssl', type=float, default=1.0, help='loss weight for reconstruction loss')
parser.add_argument('--proj_dim', type=int, default=64)

# contrastive learning arguments
parser.add_argument('--loss_type', type=str, default='default',
                    choices=['default', 'nt_xent', 'triplet'],
                    help='Type of contrastive loss to use')
parser.add_argument('--temperature', type=float, default=0.07,
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