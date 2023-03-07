# -- coding: utf-8 --
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation
import tensorflow as tf
from MyResnet import RESNET50
import os


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

    def __init__(self, num_classes,  **kwargs):
        super(SupervisedHead, self).__init__(**kwargs)
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

    print('resnet50_weighted')
    base_model = RESNET50(input_shape=input_shape)

    embeddings = base_model(inputs)
    norm_layer=UnitNormLayer()
    output=norm_layer(embeddings)
    encoder_network = tf.keras.Model(inputs, output)
    return encoder_network


class MoCo(tf.keras.Model):
    def __init__(self,args,logger,**kwargs):
        super(MoCo,self).__init__(**kwargs)
        self.args=args

        if args.encoder_snapshot:
            self.encoder = tf.keras.models.load_model(args.encoder_snapshot)
            self.encoder.summary()
            logger.info("load encoder at {}".format(args.encoder_snapshot))
        else:
            self.encoder = encoder_net(args)

        self.proj = []
        if args.proj_snapshot:
            for j in range(len(self.args.sources)):
                snap = os.path.join(args.proj_snapshot,"proj{}".format(args.sources[j]))
                proj = tf.keras.models.load_model(snap)
                self.proj.append(proj)
                logger.info("load {} proj at {}".format(args.sources[j],snap))
        else:
            for source in self.args.sources:
                proj = ProjectionHead(self.args.num_proj_layers, self.args.dense_units_list)
                self.proj.append(proj)

        self.source_classifier=[]
        self.target_classifier=[]
        if args.classifier_snapshot:
            for j in range(len(self.args.sources)):
                snap = os.path.join(args.classifier_snapshot,"classifier_{}".format(args.sources[j]))
                # classifier = tf.keras.models.load_model(snap)
                self.source_classifier.append(tf.keras.models.load_model(snap))
                logger.info("load {} classifier at {}".format(args.sources[j], snap))
                self.target_classifier.append(tf.keras.models.load_model(snap))
        else:
            for source in self.args.sources:
                #classifier = SupervisedHead(args.num_classes)
                self.source_classifier.append(SupervisedHead(args.num_classes))
                self.target_classifier.append(SupervisedHead(args.num_classes))
