# -- coding: utf-8 --
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation
import tensorflow as tf
from losses import contrastive_loss, ContrastiveLoss
from Resnet import resnet
from MyResnet import RESNET50
from Vgg import VGG16


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
                    tf.keras.Sequential([Dense(dense_units_list[j], use_bias=False), UnitNormLayer()])
                )

    def call(self, inputs):

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

    base_model = RESNET50(input_shape=input_shape)
    base_model.trainable = True

    norm_layer=UnitNormLayer()
    embeddings = base_model(inputs)
    output=norm_layer(embeddings)

    encoder_network = tf.keras.Model(inputs, output)
    return encoder_network


def add_weight_decay(model, weight_decay):
    """Compute weight decay from flags."""
    # Weight decay are taking care of by optimizer for these cases.
    # Except for supervised head, which will be added here.
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_variables
        if 'bias' not in v.name
    ]
    if l2_losses:
        return weight_decay * tf.add_n(l2_losses)
    else:
        return 0


class CustomModel(tf.keras.Model):
    def __init__(self, args):
        super(CustomModel, self).__init__()
        self.args = args
        if self.args.reuse:
            self.encoder = tf.keras.models.load_model(self.args.encoder_snapshot)
            print("loas encoder weights from {}".format(self.args.encoder_snapshot))
            self.encoder.summary()
            self.proj = tf.keras.models.load_model(self.args.proj_snapshot)
            print("loas projection weights from {}".format(self.args.proj_snapshot))
        else:
            self.encoder = encoder_net(self.args)
            self.encoder.trainable=True
            self.encoder.summary()
            self.proj = ProjectionHead(self.args.num_proj_layers, self.args.dense_units_list)
            self.proj.trainable=True

        self.compute_loss = contrastive_loss
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    @tf.function
    def train_step(self, data):
        x, y = data

        num_transforms = 2

        # Split channels, and optionally apply extra batched augmentation.
        features_list = tf.split(
            x, num_or_size_splits=num_transforms, axis=-1)
        x = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)

        y = tf.concat([y, y], axis=-1)

        with tf.GradientTape() as tape:
            t = self.encoder(x, training=True)
            y_pred = self.proj(t, training=True)
            loss = self.compute_loss(y, y_pred)

            # loss+=self.encoder.losses+self.proj.losses
        trainable_var = self.encoder.trainable_variables + self.proj.trainable_variables
        gradients = tape.gradient(loss, trainable_var)
        self.optimizer.apply_gradients(zip(gradients, trainable_var))

        self.loss_tracker.update_state(loss)


        return {"loss": self.loss_tracker.result()}
        # return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker]
