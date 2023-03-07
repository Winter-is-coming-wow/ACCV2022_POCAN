import os
import sys
import random
import logging
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime


def check_arguments(args):
    assert args.result_path is not None, 'result_path must be entered.'
    return args


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           type=str,       default='pretext',
                        choices=['pretext', 'lincls'])
    parser.add_argument("--dataset",        type=str,       default='cifar10')
    parser.add_argument("--freeze",         default=True)
    parser.add_argument("--backbone",       type=str,       default='resnet18')
    parser.add_argument("--batch_size",     type=int,       default=256)
    parser.add_argument("--classes",        type=int,       default=10)
    parser.add_argument("--img_size",       type=int,       default=32)
    parser.add_argument("--channel",        type=int,       default=3)
    parser.add_argument("--proj_hidden_dim",       type=int,       default=2048)
    parser.add_argument("--proj_output_dim",    type=int,       default=128)
    parser.add_argument("--prototype_dim",   type=int,      default=10)
    parser.add_argument("--num_crop",                        default=[2, 0])
    parser.add_argument("--size_crop",                        default=[32, 24])  #[32,24] [48,32]
    parser.add_argument("--min_scale",                        default=[0.5, 0.14])
    parser.add_argument("--max_scale",                        default=[1., 0.5])

    parser.add_argument("--weight_decay",   type=float,     default=1e-6)
    parser.add_argument("--use_bias",       default=True)
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=500)

    parser.add_argument("--lr_mode",        type=str,       default='cosine',  
                        choices=['constant', 'cosine'])
    parser.add_argument("--initial_lr",      type=float,    default=0.1)
    parser.add_argument("--traindata_num",   type=int,      default=50000)#28709
    parser.add_argument('--data_path',      type=str,       default=None)
    parser.add_argument('--result_path',    type=str,       default='output/trained_models')
    parser.add_argument('--train_info',     type=str,       default='500epochs')
    parser.add_argument('--snapshot',       type=str,       default='output/trained_models/cifar10_resnet50_100_reuse_100epochs/backbone')
    parser.add_argument('--snapshot_proj',  type=str,       default='output/trained_models/cifar10_resnet50_100_256/projection_prototype')
    parser.add_argument('--reuse',          type=bool,      default=False )
    parser.add_argument("--gpus",           type=str,       default='0')
    parser.add_argument('--temperature',      type=float,      default=0.1)
    parser.add_argument('--epsilon',       default=0.03)
    parser.add_argument('--selected_emotion',            default= {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6})

    return check_arguments(parser.parse_args())


def set_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def get_logger(name,file_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    file_handler=logging.FileHandler(os.path.join(file_path,'{}.log'.format(name)),encoding='utf-8')
    screen_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    logger.addHandler(file_handler)
    return logger


def get_session(args):
    assert int(tf.__version__.split('.')[0]) >= 2.0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.gpus != '-1':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def create_stamp():
    weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    temp = datetime.now()
    return "{:02d}{:02d}{:02d}_{}_{:02d}_{:02d}_{:02d}".format(
        temp.year % 100,
        temp.month,
        temp.day,
        weekday[temp.weekday()],
        temp.hour,
        temp.minute,
        temp.second,
    )
