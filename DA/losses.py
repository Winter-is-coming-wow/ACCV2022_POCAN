import tensorflow as tf
import numpy as np
from sklearn import metrics
from collections import Counter

LARGE_NUM = 1e9


def contrastive_loss(y, z, temperature=0.1, base_temperature=0.07):
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

    # z=tf.clip_by_value(z,1e-8,1e8)
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
    logits = tf.cast(logits, tf.float32)
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - \
               tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

    # compute mean of log-likelihood over positive
    # this may introduce NaNs due to zero division,
    # when a class only has one example in the batch
    mask_sum = tf.reduce_sum(mask, axis=1)

    mean_log_prob_pos = tf.math.divide_no_nan(tf.reduce_sum(mask * log_prob, axis=1)[mask_sum > 0],
                                              mask_sum[mask_sum > 0])
    # loss
    loss = - mean_log_prob_pos
    # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
    loss = tf.reduce_mean(loss)
    return loss


def SSL_contrastive_loss(hidden, temperature=0.1):
    """Compute loss for model.
    Args:
      hidden: hidden vector (`Tensor`) of shape (bsz, dim).
      hidden_norm: whether or not to use normalization on the hidden vector.
      temperature: a `floating` number for temperature scaling.
      strategy: context information for tpu.
    Returns:
      A loss scalar.
      The logits for contrastive prediction task.
      The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    # Gather hidden1/hidden2 across replicas and create local labels.

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)
    # return loss, logits_ab, labels

    return loss


def EMLoss(prob_target):
    prob_target=tf.nn.softmax(prob_target)
    loss_sum = tf.reduce_sum(tf.multiply( prob_target,tf.math.log(prob_target)), axis=1)
    return -tf.reduce_mean(loss_sum)


def EMLoss_with_logit(prob_source, prob_target):
    prob_source = tf.nn.softmax(prob_source)
    prob_target = tf.nn.softmax(prob_target)
    prob_sum = prob_source + prob_target
    loss_sum = tf.reduce_sum(tf.multiply(prob_sum, tf.math.log(prob_sum)), axis=1)
    return -tf.reduce_mean(loss_sum)


def cross_entrop(y, y_pred):
    y = tf.stop_gradient(tf.nn.softmax(y))
    y_pred = tf.nn.softmax(y_pred)
    loss_sum = tf.reduce_sum(tf.multiply(y, tf.math.log(y_pred)), axis=1)
    return -tf.reduce_mean(loss_sum)


def ce_loss(y, logits, class_weights=None):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.squeeze(y), logits)
    if class_weights is not None:
        e = class_weights[np.asarray(y, dtype=np.int)]
        loss = loss * e
    return tf.reduce_mean(loss)

def prototype_contrastive_loss(y, z, proto, density=None, temperature=0.5):
    y = tf.expand_dims(y, -1)
    proto_index = np.array([0, 1, 2, 3, 4, 5, 6])

    mask = tf.cast(tf.equal(y, proto_index), tf.float32)

    logits_metrix = tf.matmul(z, proto, transpose_b=True)
    if density is None:
        anchor_dot_contrast = tf.divide(
            logits_metrix,
            temperature
        )
    else:
        temp_proto = density[tf.squeeze(y)]
        anchor_dot_contrast = tf.divide(
            logits_metrix,
            temp_proto
        )

    logits_max = tf.reduce_max(tf.stop_gradient(anchor_dot_contrast), axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max
    exp_logits = tf.exp(logits)

    log_prob = logits - \
               tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))
    # log_prob=tf.cast(log_prob,tf.float32)

    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1)

    loss = -mean_log_prob_pos
    loss = tf.reduce_mean(loss)
    return loss


def cdc_loss(ys, yt, zs, zt, temperature=0.1):
    """
    implement according to <Cross-domain Contrastive Learning for Unsupervised Domain Adaptation>
    """
    ys = tf.expand_dims(ys, -1)
    yt = tf.expand_dims(yt, -1)

    mask_s = tf.cast(tf.equal(ys, tf.transpose(yt)), tf.float32)
    anchor_dot_s = tf.divide(tf.matmul(zs, tf.transpose(zt)), temperature)
    logits_max_s = tf.reduce_max(tf.stop_gradient(anchor_dot_s), axis=1, keepdims=True)
    logits_s = anchor_dot_s - logits_max_s
    logits_s = tf.cast(logits_s, tf.float32)
    exp_logits_s = tf.exp(logits_s)
    log_prob_s = logits_s - tf.math.log(tf.reduce_sum(exp_logits_s, axis=1, keepdims=True))
    mask_s_sum = tf.reduce_sum(mask_s, axis=1)
    mean_log_prob_s = tf.math.divide_no_nan(tf.reduce_sum(log_prob_s * mask_s, axis=1)[mask_s_sum > 0],
                                            mask_s_sum[mask_s_sum > 0])

    mask_t = tf.transpose(mask_s)
    anchor_dot_t = tf.transpose(anchor_dot_s)
    logits_max_t = tf.reduce_max(tf.stop_gradient(anchor_dot_t), axis=1, keepdims=True)
    logits_t = anchor_dot_t - logits_max_t
    logits_t = tf.cast(logits_t, tf.float32)
    exp_logits_t = tf.exp(logits_t)
    log_prob = logits_t - tf.math.log(tf.reduce_sum(exp_logits_t, axis=1, keepdims=True))
    mask_t_sum = tf.reduce_sum(mask_t, axis=1)
    mean_log_prob_t = tf.math.divide_no_nan(tf.reduce_sum(log_prob * mask_t, axis=1)[mask_t_sum > 0],
                                            mask_t_sum[mask_t_sum > 0])

    total_loss = -(tf.reduce_mean(mean_log_prob_s) + tf.reduce_mean(mean_log_prob_t)) / 2

    return total_loss


def ce_proto(y, z, proto, class_weights=None):
    logits = tf.matmul(z, proto, transpose_b=True) / 0.1
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.squeeze(y), logits)
    if class_weights is not None:
        e = class_weights[np.asarray(y, dtype=np.int)]
        loss = loss * e

    return tf.reduce_mean(loss)