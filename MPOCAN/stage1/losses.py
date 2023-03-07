"""Implements contrastive losses over features from multiple views of data."""


import tensorflow as tf
from gl import Config
from tensorflow_addons.optimizers import SGDW,AdamW
config_ob=Config()


# Derived wholesale from (unexported) LossFunctionWrapper in keras Losses.py


class ContrastiveLoss(tf.keras.losses.Loss):

    def __init__(self,
                 name='contrastive_loss',
                 temperature=0.5, base_temperature=0.07
                 ):
        super(ContrastiveLoss, self).__init__()

    def call(self, y_true, y_pred):
        #return contrastive_loss( y_true,y_pred, temperature=0.5, base_temperature=0.07)
        return contrastive_loss(y_true,y_pred)

def contrastive_loss( y,z, temperature=0.1, base_temperature=0.07):
    '''
    Supervised normalized temperature-scaled cross entropy loss.
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    Implementation modified from:
        - https://github.com/google-research/simclr/blob/master/objective.py
        - https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    batch_size = tf.shape(z)[0]
    y = tf.expand_dims(y, -1)

    #z=tf.clip_by_value(z,1e-8,1e8)
    # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
    #     has the same class as sample i. Can be asymmetric.
    mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)

    anchor_dot_contrast = tf.divide(
        tf.matmul(z, tf.transpose(z)),
        temperature
    )

    # # for numerical stability
    logits_max = tf.reduce_max(tf.stop_gradient(anchor_dot_contrast), axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max
    # # tile mask
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask
    # compute log_prob
    logits=tf.cast(logits,tf.float32)
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - \
        tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

    # compute mean of log-likelihood over positive
    # this may introduce NaNs due to zero division,
    # when a class only has one example in the batch
    mask_sum = tf.reduce_sum(mask, axis=1)

    mean_log_prob_pos = tf.math.divide_no_nan(tf.reduce_sum(mask * log_prob, axis=1)[mask_sum > 0],mask_sum[mask_sum > 0])
    # loss
    loss = - mean_log_prob_pos
    # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
    loss = tf.reduce_mean(loss)
    return loss

def build_optimizer(learning_rate,weight_decay=None):
    """Returns the optimizer."""
    if config_ob.optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate, config_ob.momentum)
    elif config_ob.optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate)
    elif config_ob.optimizer == 'sgdw':
        return SGDW(weight_decay=weight_decay, learning_rate=learning_rate, momentum=config_ob.momentum)
    elif config_ob.optimizer == 'adamw':
        return AdamW(weight_decay=weight_decay, learning_rate=learning_rate)



    # elif config_ob.optimizer == 'lars':
    #     return LARSOptimizer(
    #         learning_rate,
    #         momentum=config_ob.momentum,
    #         weight_decay=config_ob.weight_decay,
    #         exclude_from_weight_decay=[
    #             'batch_normalization', 'bias', 'head_supervised'
    #         ])
    else:
        raise ValueError('Unknown optimizer {}'.format(config_ob.optimizer))


def add_weight_decay(model, adjust_per_optimizer=True):
    """Compute weight decay from flags."""
    if adjust_per_optimizer and 'lars' in config_ob.optimizer:
        # Weight decay are taking care of by optimizer for these cases.
        # Except for supervised head, which will be added here.
        l2_losses = [
            tf.nn.l2_loss(v)
            for v in model.trainable_variables
            if 'head_supervised' in v.name and 'bias' not in v.name
        ]
        if l2_losses:
            return config_ob.weight_decay * tf.add_n(l2_losses)
        else:
            return 0

    # TODO(srbs): Think of a way to avoid name-based filtering here.
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_weights
        if 'batch_normalization' not in v.name
    ]
    loss = config_ob.weight_decay * tf.add_n(l2_losses)
    return loss

