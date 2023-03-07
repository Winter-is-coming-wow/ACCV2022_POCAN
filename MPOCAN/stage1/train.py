# -- coding: utf-8 --
import tensorflow as tf
from functools import reduce
import os, math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from gl import Config
from sklearn import svm
from common import get_logger, get_session
from Dataloader import set_dataset, DataLoader, RAFdataset2, Fer2013dataset, Fer2013Plus, ExpW,CKplus, oulu_CASIA, \
     JAFFEdataset, SFEW
from model import CustomModel
from util import get_logger, get_session, create_stamp,set_seed
from losses import contrastive_loss,build_optimizer

def transer(dataset, label):
    args = Config()
    if dataset == 'fer2013':
        label = args.fer2013_uniform[label]
    elif dataset == 'RAF':
        label = args.RAF_uniform[label]
    elif dataset == 'fer2013plus':
        label = args.fer2013plus_uniform[label]
    elif dataset == 'AffectNet':
        label = args.AffectNet_uniform[label]
    elif dataset == 'ExpW':
        label = args.ExpW_uniform[label]
    elif dataset == 'ckplus':
        label = args.ckplus_uniform[label]
    elif dataset == 'JAFFE':
        label = args.JAFFE_uniform[label]
    elif dataset == 'oulu':
        label = args.oulu_uniform[label]
    elif dataset == 'sfew':
        label = args.sfew_uniform[label]
    else:
        raise ("no such dataset,can not transform label")
    return label

def train_step(data,model,optimizer,args,index):
    x, y = data

    num_transforms = 2

    # Split channels, and optionally apply extra batched augmentation.
    features_list = tf.split(
        x, num_or_size_splits=num_transforms, axis=-1)
    x = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)

    y = tf.concat([y, y], axis=-1)


    with tf.GradientTape() as tape:
        t = model.encoder(x, training=True)
        y_pred = model.projs[index](t, training=True)
        loss = contrastive_loss(y, y_pred)
    trainable_var = model.encoder.trainable_variables + model.projs[index].trainable_variables
    gradients = tape.gradient(loss, trainable_var)
    optimizer.apply_gradients(zip(gradients, trainable_var))

    return loss

def train(sources,args,logger, output_path):
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))
    start_time = create_stamp()
    logger.info(start_time)
    logger.info("-----------------------------------------------------")

    net = CustomModel(args)
    net.encoder.trainable = True
    net.encoder.summary()
    for proj in net.projs:
        proj.trainable = True
    train_steps = 0

    source_datasets = []

    for index in range(len(args.datasets)):
        train_steps = max(train_steps, math.ceil(len(list(sources[index].as_numpy_iterator())) / args.batch_size))
        source_dataset = DataLoader(args, 'pretext', 'train', sources[index], args.batch_size).dataloader_sup()
        source_datasets.append(iter(source_dataset))
    logger.info("train_steps:{}".format(train_steps))

    # optimizer = tf.keras.optimizers.SGD(0.0001, momentum=0.9)
    # opt = tf.keras.optimizers.SGD(0.001, momentum=0.9)
    # optimizer = SGDW(weight_decay=args.weight_decay, learning_rate=0.0001, momentum=0.9)
    # opt = SGDW(weight_decay=args.weight_decay, learning_rate=0.001, momentum=0.9)
    decay_steps = train_steps * args.epochs
    logger.info("decay steps {}".format(decay_steps))
    # lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=args.learning_rate, decay_steps=decay_steps,
    #                                                   alpha=0.0001)
    # optimizer = build_optimizer(lr_decayed_fn, args.weight_decay)
    for epoch in range(args.epochs):
        logger.info("{}/{}".format(epoch, args.epochs))
        for step in range(train_steps):
            logger.info("{}\{}".format(step,train_steps))
            alpha=0.0001
            cosine_decay = 0.5 * (1 + tf.math.cos((train_steps*epoch+step) * np.pi / decay_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            lr = args.learning_rate * decayed
            #lr=args.learning_rate * ((1-0.00001) * ((1 + tf.math.cos((train_steps*epoch+step) * np.pi / decay_steps)) / 2)+0.00001)
            logger.info("lr:{}".format(lr))
            optimizer = build_optimizer(lr, args.weight_decay)
            for index in range(len(args.datasets)):
                source_batch = source_datasets[index].get_next()
                results = train_step(source_batch, net, optimizer, args, index)
                logger.info("loss:{}".format(results))

    # logger.info("save model ")
    net.encoder.save(os.path.join(output_path, "backbone"))
    for j in range(len(args.datasets)):
        net.projs[j].save(os.path.join(output_path, "proj{}".format(args.datasets[j])))
    # for j in range(len(args.datasets)):
    #     net.classifiers[j].save(os.path.join(output_path, "classifiers{}_{}".format(j, args.epochs + 1)))
    end_time = create_stamp()
    logger.info(end_time)


def main():
    args = Config()
    get_session(args)
    set_seed()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    output_path = os.path.join(args.output_path, "{}".format(args.train_info))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    logger = get_logger("MyLogger", output_path)

    sources = []

    for ds_name in args.datasets:
        if ds_name == 'fer2013':
            dataset = Fer2013dataset()
            train_ds, val_ds = dataset.load_data(stage=1)
            # test_ds = dataset.load_data(stage=2)
            # total_ds = test_ds.concatenate(train_ds.concatenate(val_ds))
            sources.append(train_ds)

        elif ds_name == 'RAF':
            dataset = RAFdataset2()
            train_ds, val_ds = dataset.load_data(stage=1, channel=3)
            # test_ds = dataset.load_data(stage=2, channel=3)
            # total_ds = test_ds.concatenate(train_ds.concatenate(val_ds))
            sources.append(train_ds)

        elif ds_name == 'JAFFE':
            dataset = JAFFEdataset()
            train_ds, test_ds = dataset.load_data()
            train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
            # test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
            # total_ds = test_ds.concatenate(train_ds)
            sources.append(train_ds)

        elif ds_name == 'ckplus':
            dataset = CKplus()
            train_ds, test_ds = dataset.load_data()
            train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
            # test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
            # total_ds = test_ds.concatenate(train_ds)
            sources.append(train_ds)

        elif ds_name == 'oulu':
            dataset = oulu_CASIA()
            train_ds, test_ds = dataset.load_data()
            train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
            # test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
            # total_ds = test_ds.concatenate(train_ds)
            sources.append(train_ds)

        elif ds_name == 'sfew':
            dataset = SFEW()
            train_ds, test_ds = dataset.load_data(channels=3)
            # total_ds = test_ds.concatenate(train_ds)
            sources.append(train_ds)
    #
    # if args.target == 'fer2013':
    #     dataset = Fer2013dataset()
    #     target_train_ds, target_val_ds = dataset.load_data(stage=1)
    #     target_test_ds = dataset.load_data(stage=2)
    #
    # elif args.target == 'RAF':
    #     dataset = RAFdataset2()
    #     target_train_ds, target_val_ds = dataset.load_data(stage=1, channel=1)
    #     target_test_ds = dataset.load_data(stage=2, channel=1)
    #
    # elif args.target == 'sfew':
    #     dataset = SFEW()
    #     target_train_ds, target_test_ds = dataset.load_data(channels=3)
    #
    # elif args.target == 'ckplus':
    #     dataset = CKplus()
    #     train_ds, test_ds = dataset.load_data()
    #     target_train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    #     target_test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    #
    # elif args.target == 'oulu':
    #     dataset = oulu_CASIA()
    #     train_ds, test_ds = dataset.load_data()
    #     target_train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    #     target_test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    #
    # elif args.target == 'JAFFE':
    #     dataset = JAFFEdataset()
    #     train_ds, test_ds = dataset.load_data()
    #     target_train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    #     target_test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    #
    # else:
    #     raise Exception("no such target dataset")

    #target = {'target_train_ds': target_train_ds, 'target_test_ds': target_test_ds}

    train(sources,  args, logger, output_path)


if __name__ == '__main__':
    main()

