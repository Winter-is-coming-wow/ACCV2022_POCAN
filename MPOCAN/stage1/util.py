# -- coding: utf-8 --
import tensorflow as tf
import os
import sys
import random
import logging
import numpy as np
from datetime import datetime
from tensorflow.keras.layers import Dense
from sklearn import metrics
LARGE_NUM = 1e9

class SupervisedHead(tf.keras.Model):

    def __init__(self, num_classes, name='head_supervised', **kwargs):
        super(SupervisedHead, self).__init__(name=name, **kwargs)
        self.linear_layer = tf.keras.Sequential([
            # Dense(512,activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs, training=None):
        inputs = self.linear_layer(inputs)
        return inputs


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

def decayed_lr(epoch,args):
    p=epoch/args.epochs
    coe=(1+10*p)**args.b
    return args.initial_lr*coe



