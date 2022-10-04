# -- coding: utf-8 --
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation
import tensorflow as tf
from Resnet import resnet
from Vgg import VGG16
from MyResnet import RESNET50
import numpy as np


class UnitNormLayer(tf.keras.layers.Layer):
    '''Normalize vectors (euclidean norm) in batch to unit hypersphere.
    '''

    def __init__(self):
        super(UnitNormLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.nn.l2_normalize(input_tensor, axis=1)
        return norm


class ProjectionHead(tf.keras.Model):
    def __init__(self, num_proj_layers, dense_units_list=None):
        super(ProjectionHead, self).__init__()
        self.num_proj_layers = num_proj_layers
        if dense_units_list is None:
            dense_units_list = [512, 128]
        self.dense_layers = []
        for j in range(num_proj_layers):
            if j != num_proj_layers - 1:
                # for the middle layers, use bias and relu for the output.
                self.dense_layers.append(
                    tf.keras.Sequential([Dense(dense_units_list[j], use_bias=True),
                                         BatchNormalization(),
                                         Activation('relu')])
                )

            else:
                # for the final layer, neither bias nor relu is used.
                self.dense_layers.append(
                    tf.keras.Sequential([Dense(dense_units_list[j],use_bias=False),UnitNormLayer()])
                )


    def call(self, inputs,training=None):

        for j in range(self.num_proj_layers):
            inputs = self.dense_layers[j](inputs)

        return inputs


class SupervisedHead(tf.keras.Model):

    def __init__(self, num_classes, name='head_supervised', **kwargs):
        super(SupervisedHead, self).__init__(name=name, **kwargs)
        self.linear_layer = tf.keras.Sequential([
            # Dense(512,activation='relu'),
            Dense(num_classes)
        ])

    def call(self, inputs, training=None):
        inputs = self.linear_layer(inputs)
        return inputs


def encoder_net(args):
    input_shape = (args.img_size, args.img_size, args.channel)
    inputs = tf.keras.Input(input_shape)

    if args.backbone == 'resnet50_weighted':

        print('resnet50_weighted')
        base_model = RESNET50(input_shape=input_shape)
    elif args.backbone == 'resnet50':
        print('resnet50')
        base_model = resnet(
            resnet_depth=args.resnet_depth,
            width_multiplier=args.width_multiplier,
        )
    elif args.backbone == 'resnet34':
        print('resnet34')
        base_model = resnet(
            resnet_depth=args.resnet_depth,
            width_multiplier=args.width_multiplier,
        )
    elif args.backbone == 'resnet18':
        print('resnet18')
        base_model = resnet(
            resnet_depth=args.resnet_depth,
            width_multiplier=args.width_multiplier,
        )
    else:
        print('vgg')
        # base_model = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=input_shape,
        #                                          pooling='avg')
        base_model=VGG16()

    embeddings = base_model(inputs)
    norm_layer=UnitNormLayer()
    output=norm_layer(embeddings)
    encoder_network = tf.keras.Model(inputs, output)
    return encoder_network


class POCAN(tf.keras.Model):
    def __init__(self,args,logger,**kwargs):
        super(MoCo,self).__init__(**kwargs)
        self.args=args

        if args.encoder_snapshot:
            self.encoder = tf.keras.models.load_model(args.encoder_snapshot)
            self.encoder.summary()
            logger.info("load encoder at {}".format(args.encoder_snapshot))
        else:
            self.encoder = encoder_net(args)

        if args.proj_snapshot:
            self.proj = tf.keras.models.load_model(args.proj_snapshot)
            self.proj.summary()
            logger.info("load proj at {}".format(args.proj_snapshot))
        else:
            self.proj = ProjectionHead(args.num_proj_layers)

        if args.classifier_snapshot:
            self.source_classify_head=tf.keras.models.load_model(args.classifier_snapshot)
            self.target_classify_head=tf.keras.models.load_model(args.classifier_snapshot)
            logger.info("load classify head at {}".format(args.classifier_snapshot))
        else:
            self.source_classify_head=SupervisedHead(args.num_classes)
            self.target_classify_head=SupervisedHead(args.num_classes)

