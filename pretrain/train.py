# -- coding: utf-8 --
import tensorflow as tf

import os, time, pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from gl import Config
from sklearn import svm
from common import get_logger, get_session
from Dataloader import set_dataset, DataLoader, RAFdataset2, Fer2013dataset, Fer2013Plus, ExpW
from model import CustomModel
import losses


class SaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 300 == 0:
            self.model.layers[0].save(os.path.join('output/cache/{}'.format(epoch + 1), 'backbone'))
            # net.layers[0].save_weights(os.path.join(output_dir, 'backbone_weight'))
            self.model.layers[1].save(os.path.join('output/cache/{}'.format(epoch + 1), 'projection'))


def main():
    args = Config()
    output_dir = os.path.join(args.result_path,
                              '{}_{}_{}_{}'.format(args.dataset, args.backbone, args.epochs, args.train_info))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    get_session(args)
    logger = get_logger("MyLogger", output_dir)
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))
    logger.info("-----------------------------------------------------")
    train_num=None
    if args.dataset == 'fer2013':
        dataset = Fer2013dataset()
        train_ds = dataset.load_data(stage=0)
        train_dataset = DataLoader(args, 'pretext', 'train', train_ds, args.batch_size).dataloader_sup()

    elif args.dataset == 'fer2013plus':
        dataset = Fer2013Plus('Majority')
        train_ds = dataset.load_data(stage=0)
        train_dataset = DataLoader(args, 'pretext', 'train', train_ds, args.batch_size).dataloader_sup()

    elif args.dataset == 'RAF':
        dataset = RAFdataset()
        train_ds = dataset.load_data(stage=0, channel=3)
        train_num=len(list(train_ds.as_numpy_iterator()))
        train_dataset = DataLoader(args, 'pretext', 'train', train_ds, args.batch_size).dataloader_sup()

    elif args.dataset == 'ExpW':
        dataset = ExpW()
        train_ds = dataset.load_data(type=1)
        train_dataset = DataLoader(args, 'pretext', 'train', train_ds, args.batch_size).dataloader_sup()

    net = CustomModel(args)
    # Build LR schedule and optimizer.
    # learning_rate = Losses.WarmUpAndCosineDecay(config_ob.learning_rate,
    #                                                config_ob.num_train_examples)
    logger.info("train num {}".format(train_num))
    decay_steps = tf.math.ceil(train_num / args.batch_size) * args.epochs
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=args.learning_rate, decay_steps=decay_steps,
                                                      alpha=0.0001)
    optimizer = losses.build_optimizer(lr_decayed_fn,args.weight_decay)


    net.compile(optimizer)

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'pretext.log'))
    History = net.fit(train_dataset, epochs=args.epochs, callbacks=[csv_logger, SaveModel()])

    net.layers[0].save(os.path.join(output_dir, 'backbone'))
    # net.layers[0].save_weights(os.path.join(output_dir, 'backbone_weight'))
    net.layers[1].save(os.path.join(output_dir, 'projection'))

    loss = History.history["loss"]
    plt.plot(list(range(len(loss))), loss)
    plt.savefig(os.path.join(output_dir, 'pretext_loss.png'))
    plt.show()


if __name__ == '__main__':
    main()
