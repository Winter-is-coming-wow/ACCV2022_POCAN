# -- coding: utf-8 --
import tensorflow as tf


class Config():
    def __init__(self):
        self.train_mode = 'pretrain'
        self.dataset = 'RAF'
        self.source = self.dataset
        self.target = 'fer2013'
        self.backbone = 'resnet50'
        self.img_size = 112
        self.channel = 3
        self.use_blur = False

        self.batch_size =128
        self.num_classes = 7
        self.epochs = 300
        self.resnet_depth = 50
        self.width_multiplier = 1

        if self.resnet_depth >= 50:
            self.dense_units_list = [2048, 256]
        else:
            self.dense_units_list = [512, 128]

        self.num_proj_layers = len(self.dense_units_list)

        self.optimizer = 'sgdw'
        self.momentum = 0.9
        self.weight_decay = 0.00001
        self.learning_rate_scaling = 'sqrt'  # ['linear', 'sqrt']
        self.temperature = 0.1
        self.learning_rate = 0.001
        self.hidden_norm = True
        self.finetune_epochs = 50
        self.lineareval_while_pretraining = False
        self.result_path = 'output'
        self.gpus = '4'
        self.selected_emotion = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

        self.train_info = 'bz{}_{}_addnorm_256F1'.format(self.batch_size,self.optimizer)
        self.encoder_snapshot = None
        self.proj_snapshot = None
        self.classfy_snapshot = None
        self.reuse = False

        self.uniform_index2label = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised',
                                    6: 'Neutral'}
        self.uniform_label2index = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprised': 5,
                                    'Neutral': 6}

        self.fer2013_uniform = tf.Variable([0, 1, 2, 3, 4, 5, 6])
        self.fer2013plus_uniform = tf.Variable([6, 3, 5, 4, 0, 1, 2])
        self.RAF_uniform = tf.Variable([5, 2, 1, 3, 4, 0, 6])
        self.ExpW_uniform = tf.Variable([0, 1, 2, 3, 4, 5, 6])
        self.AffectNet_uniform = tf.Variable([6, 3, 4, 5, 2, 1, 0])
        self.sfew_uniform = tf.Variable([0, 1, 2, 3, 4, 5, 6])
        self.JAFFE_uniform = tf.Variable([5, 2, 1, 3, 4, 0, 6])
        self.ckplus_uniform = tf.Variable([6, 0, 1, 2, 3, 4, 5])
        self.oulu_uniform = tf.Variable([0, 1, 2, 3, 4, 5, 6])
