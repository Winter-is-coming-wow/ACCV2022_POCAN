# -- coding: utf-8 --
import tensorflow as tf


class Config():
    def __init__(self):
        # fer2013 fer2013plus RAF JAFFE sfew ckplus  oulu ExpW
        self.source = 'RAF'
        self.target = 'ckplus'
        self.threshold = 0.95
        self.interval = 1
        self.temperature = 0.7
        self.n=3
        self.beta=2

        self.backbone = 'resnet50'
        self.resnet_depth = 50
        self.width_multiplier = 1
        self.dense_units_list = [2048, 128]
        self.dim = self.dense_units_list[-1]
        self.num_proj_layers = len(self.dense_units_list)

        self.num_classes = 7
        self.img_size = 112
        self.channel = 3
        self.use_blur = False
        self.batch_size = 64
        self.source_bz =9
        self.initial_lr = 0.0001
        self.epochs = 20

        self.lincls_initial_lr = 0.01
        self.lincls_epoch = 20

        self.optimizer = 'sgdw'
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.learning_rate_scaling = 'sqrt'  # ['linear', 'sqrt']

        # self.max_t = 1.0
        # self.min_t = 0.1

        self.hidden_norm = True

        self.lineareval_while_pretraining = False
        self.output_path = r'ablation\{}_{}'.format(self.source, self.target)
        self.gpus = '1'

        if self.source == 'fer2013':
            self.encoder_snapshot = None  # r'G:\demo\cross_dataset\Sup_contrastive\output\fer2013_resnet18_150_7emotion_bz256\backbone'
        elif self.source == 'RAF':
            self.encoder_snapshot = r''
            self.proj_snapshot = r''
            self.classifier_snapshot = r''
            self.proj_target_snapshot = None

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

        self.train_info = '{}_{}_{}_nopst_{}'.format(self.backbone, self.epochs,self.optimizer,self.beta)

