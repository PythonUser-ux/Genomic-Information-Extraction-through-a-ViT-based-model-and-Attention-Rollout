# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import numpy as np
import json
import sys

from datetime import timedelta, datetime

import torch
import time

#from monai.metrics import ConfusionMatrixMetric
from train import train

from models.modeling import VisionTransformer, ParallelVisionTransformer, CNNClassifier, CONFIGS
from utils.data_utils import get_loss_weights
from utils.utils import count_parameters, set_seed
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

def setup(args):
    # Prepare model
    
    
    if args.dataset in ['NGSLUNG_attMap']:
        in_channels = 2
        config = None
    elif args.dataset in ['NGSLUNG_patchedImage']:
        in_channels = 1
        config = None
    else:
        in_channels = 1
        config = CONFIGS[args.model_type]
    
    if '4classes' in args.label_key:
        num_classes = 4
    elif 'all_gene' in args.label_key:
        num_classes = 10
    else:
        num_classes = 2
    
    if not 'ViT' in args.model_type:
        model = CNNClassifier(args.img_size, in_channels=in_channels, num_classes=num_classes, loss_weights = args.loss_weights, model_type = args.model_type, pretrained = args.pretrained, use_clinical_data = args.use_clinical_data, fusion_strategy = args.fusion_strategy, combination_type = args.combination_type, clin_feats = args.clinical_features)
    elif 'parallel' in args.model_type:
        model = ParallelVisionTransformer(config, args.img_size, in_channels=in_channels, zero_head=True, num_classes=num_classes, loss_weights = args.loss_weights, embeddings_type = args.embeddings_type, depth_kernel_size = args.depth_kernel_size, out_depth = args.out_depth, loss_type = args.loss_type)
    else:
        model = VisionTransformer(config, args.img_size, in_channels=in_channels, zero_head=True, num_classes=num_classes, loss_weights = args.loss_weights, embeddings_type = args.embeddings_type, depth_kernel_size = args.depth_kernel_size, out_depth = args.out_depth, loss_type = args.loss_type)
    if args.pretrained and 'ViT' in args.model_type:
        model.load_from(np.load(args.pretrained_dir))
    model.float().to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return config, args, model

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument('--root_dir', type=str, default=os.path.join('data','ngslung_images'))
    parser.add_argument('--split_path', type=str, default=os.path.join('data','5BalancedCrossValFold_allGene_114sample.json'))
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")


    '''Model Params'''
    parser.add_argument("--dataset", choices=["NGSLUNG", 'NGSLUNG_crop', 'NGSLUNG_crop_ValidVoting', 'NGSLUNG_crop_ValidVoting_ScaledRange', 'NGSLUNG_crop_ScaledRange_AllVolume', 'NGSLUNG_crop_ScaledRange_3D', 'NGSLUNG_attMap', 'NGSLUNG_patchedImage'], default="NGSLUNG",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", 
                                                 "ViT-MRI", 'R50-ViT-MRI', 'ViT-half', 'ViT-half2', 'ViT-1Lay', 'ViT-h12l2', 'ViT-h8l2', 'ViT-h8l2hid384', 'ViT-h4l2hid384', 'ViT-parallel_h12l2', 'ViT-parallel', 'ResNet18', 'ResNet50', 'ResNet101','AlexNet','DenseNet121', 'Vgg16','MobileNet_v2', 'EfficientNet_b6', 'EfficientNet_b5', 'CoAtNet_0', 'CoAtNet_1', 'CoAtNet_2', 'CoAtNet_3', 'CoAtNet_4'], 
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained", action='store_true' ,
                        help="If use a pretrained model.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--vit_pretrained_dir", type=str, default = None,
                        help="Where to search for pretrained ViT models. Used only for ViT-Resnet model")
    parser.add_argument("--train_only_classifier", action='store_true',
                        help="Freeze all network and train only the classifier")
    parser.add_argument('--window', type = str, choices=['parenchyma', 'mediastinum'], default = 'parenchyma',
                        help = 'Which window to use, if windowing is applied')
    
    parser.add_argument("--execute_test", action = 'store_true', help = 'If execute test phase or not. If no, the test split is used with training set during the training phase.')
    
    '''Input Params'''
    parser.add_argument("--img_size", default=192, type=int,
                        help="Resolution size")
    parser.add_argument("--num_patches", default=9, type=int,
                        help="Number of patches to create a patched input")
    parser.add_argument("--k_divisible", type = int, default = 1,
                        help = "Minimum value of the crop dimension")
    parser.add_argument("--padding", type = str, default = 'constant', choices = ['constant', 'edge', 'reflect'],
                        help = "Minimum value of the crop dimension")
    parser.add_argument("--eval_stride", type = int, default = 2, help = "Stride to use in ListPatchedImages during validation phase")
    #params to use clinacl data with image extracted features
    parser.add_argument("--use_clinical_data", action = 'store_true', help = 'If using also clinical data with image features to classify images.')
    parser.add_argument("--fusion_strategy", type =  str, default = 'learned_features',choices = ['features', 'learned_features'], help = "What type of clinical features concatenate when clinical data are used.")
    parser.add_argument("--combination_type", type =  str, default = 'concat',choices = ['concat', 'sum', 'mul'], help = 'How to fuse image and clinical features')
    parser.add_argument("--clinical_features", type = int, default = 10, help = "Number of used clinical features")
    
    
    '''Training params'''
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--cuda_id", type=int, default=0,
                        help="Index of GPU")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size", default=16, type=int,
                        help="Total batch size for test.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument('--label_key', type=str, default='label')
    parser.add_argument("--num_fold", choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], type=int, required=True,
                        help="Which Cross Validation folder use.")
    parser.add_argument("--num_inner_loop", type=int, default= None,
                        help="Number of fold of inner loop, for Nested train.")
    parser.add_argument("--accuracy", choices=['simple','balanced', 'both'], default='both', 
                        help="The type of accuracy computed")
    
    parser.add_argument('--early_stopping', action='store_true', help='If training using early stopping strategy')
    parser.add_argument('--es_patience', type= int, default = 5, help='When early stopping is used, how long to wait after last time validation loss improved.')
    parser.add_argument('--es_delta', type = float, default = 0.01, help = 'Minimum change in the monitored quantity to qualify as an improvement.')
    parser.add_argument('--stop_on', type = str, choices = ['accuracy', 'loss', 'auc'], default = 'loss', 
                        help = 'If use val_accuracy or val_loss for early stopping')
    parser.add_argument('--eval_auc', action='store_true', help='If evalueate auc during validation')
    
    '''Loss param'''
    parser.add_argument('--loss_type', type = str, choices = ['CrossEntropy', 'MSE' ,'MultiLabelSoftMarginLoss', 'BCELoss', 'BCEWithLogits'], default = 'CrossEntropy', 
                        help = 'Which type of loss to use. ')
    parser.add_argument('--weighted_loss', action='store_true', help='If use a weighted loss')
    
    '''Optimizer param'''
    parser.add_argument("--optimizer", choices=["SGD", "Adam", 'RMSprop', 'AdamW'], default='SGD', type=str,
                        help="The optimizer.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    
    
    '''Scheduler'''
    parser.add_argument('--use_scheduler', action='store_true',
                        help = "if or not use a scheduler for learning rate decay. The decay type is determined from the decay_type parameter")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--cycles_scheduler", default = .5, type=float,
                        help = "If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.")
    
    
    '''Arguments for 3D embeddings'''
    parser.add_argument('--embeddings_type', type = str, choices = ['2D', '3D'], default = '2D', help = 'if using a 2D or a 3D Conv to convert patches into embeddings')
    parser.add_argument('--depth_kernel_size', type = int, default = 5, help = 'depth kernel dimension if embeddings_type is 3D')
    parser.add_argument('--out_depth', type = int, default = 3, help = 'depth after adaptive pooling if embeddings_type is 3D')
    parser.add_argument('--max_depth', type = int, default = 67, help = 'Max number of segmemnted slices.')
    
    '''Optimization Params (from ViT original)'''
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    
    
    args = parser.parse_args()
    
    args.loss_weights = None
    if args.weighted_loss:
        args.loss_weights = get_loss_weights(args.split_path, args.label_key)
    
    #verify if the selected split_path is correct w.r.t the selected label
    if not 'multi_label' in args.label_key:
        assert args.label_key in args.split_path, 'ERROR! Label key for classificatin and split path don\'t match'
    else:
        assert 'multi_label' in args.split_path, 'ERROR! Label key for classificatin and split path don\'t match'
    
    #if the 3D embeddings is selected, verify that the correct dataset is selected (NGSLUNG_crop_ScaledRange_3D)
    if args.embeddings_type == '3D':
        assert args.dataset in ['NGSLUNG_crop_ScaledRange_3D'], 'ERROR! Embeddings type and dataset don\'t match'
    
    if args.loss_type in ['MSE', 'Hamming', 'MultiLabelSoftMarginLoss']:
        assert 'multi_label' in args.label_key, f'ERROR! You can use {args.loss_type} only for multi label classification.'
    
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    
    timestamp_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    args.name = timestamp_str + args.name + '_lr'+str(args.learning_rate) +'_wd'+str(args.weight_decay)+'_fold' + str(args.num_fold)
    
    if args.eval_auc:
        args.output_dir = os.path.join(args.output_dir, args.label_key, 'crop', args.model_type, 'withAUC', args.name)
    else: 
        args.output_dir = os.path.join(args.output_dir, args.label_key, 'crop', args.model_type, args.name)
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    
    set_seed(args)
    if args.dataset in ['NGSLUNG_attMap', 'NGSLUNG_patchedImage']:
        KEYS = ('patched_img','att_maps', args.label_key)
    else:
        KEYS = ('image','mask', args.label_key)
    
    dict_args = vars(args).copy()
    dict_args['device'] = str(dict_args['device'])
    dict_args['loss_weights'] = dict_args['loss_weights'].tolist() if args.loss_weights is not None else None
    # Saving training info
    
    if not 'Nested' in args.dataset:
        # Model & Tokenizer Setup
        config, args, model = setup(args)
        
        info = {
            'model_name': model.__class__.__name__,
            'KEYS': KEYS,
            'model_config': config.to_dict() if config is not None else None, 
            'model_args': dict_args
            }
        
        with open(os.path.join(args.output_dir, args.name+'.json'), 'w') as fp:
            json.dump(info, fp)
    
        l = str(sys.argv)
        with open(os.path.join(args.output_dir, 'cmd.json'), 'w') as fp:
            json.dump(l, fp)

        # Training
        train(args, logger, model, KEYS, args.num_inner_loop, info)
    else:
        out_dir = args.output_dir
        for i in range(args.num_inner_loop):
            args.output_dir = os.path.join(out_dir, f'inner_loop_{i}')
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
            config, args, model = setup(args)
            info = {
                'model_name': model.__class__.__name__,
                'KEYS': KEYS,
                'model_config': config.to_dict(), 
                'model_args': dict_args
                }
            with open(os.path.join(args.output_dir, args.name+'.json'), 'w') as fp:
                json.dump(info, fp)
    
            l = str(sys.argv)
            with open(os.path.join(args.output_dir, 'cmd.json'), 'w') as fp:
                json.dump(l, fp)

            # Training
            train(args, logger, model, KEYS, i, info)
    
        

if __name__ == "__main__":
    main()
