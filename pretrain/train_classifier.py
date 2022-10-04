# -- coding: utf-8 --
import tensorflow as tf
import os
from util import get_logger, get_session, create_stamp


from model import CCD
from Dataloader import set_dataset, DataLoader, RAFdataset, Fer2013dataset, Fer2013Plus, ExpW,RAFdataset2
from gl import Config
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from util import get_index, get_T, decayed_lr, get_num_cluster



def get_classifier(net, args):
    """
    return a classify model
    """
    inputs = tf.keras.Input((args.img_size, args.img_size, args.channel))
    net.encoder.trainable = False
    net.proj.trainable=False
    net.source_classify_head.trainable=True
    x = net.encoder(inputs, training=False)
    x = net.proj(x, training=False)
    x = net.source_classify_head(x, training=True)
    return tf.keras.Model(inputs, x)

def scheduler(epoch, lr):
    if epoch < 20:
        return 0.0001
    elif epoch<40:
        return 0.00001
    else:
        return 0.000001

def train_classifier():
    """
    use source train dataset train a classifier based on pretrained backbone
    """
    args = Config()
    args.backbone='resnet50'
    args.encoder_model = r'C:\Users\DELL\PycharmProjects\superwang\final\pretrain\output\RAF_resnet50_300_bz128_sgdw_001\backbone'
    args.proj_snapshot = r'C:\Users\DELL\PycharmProjects\superwang\final\pretrain\output\RAF_resnet50_300_bz128_sgdw_001\projection'
    args.classifier_snapshot=None

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    output_path = os.path.join(args.output_path, "source_classifier_with_proj_sgdw")
    encoder_output_path = os.path.join(args.output_path, "source_encoder_with_proj_sgdw")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    get_session(args)
    logger = get_logger("MyLogger_classifier", output_path)
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))
    start_time = create_stamp()
    logger.info(start_time)
    logger.info("------------------------------------------------------------------")


    dataset = Fer2013dataset()

    test_ds = dataset.load_data(stage=2)

    #val_dataset = DataLoader(args, 'lincls', 'val', val_ds, args.batch_size).dataloader_sup()
    fer_test_dataset = DataLoader(args, 'lincls', 'val', test_ds, args.batch_size).dataloader()

    if args.source == 'RAF':
        dataset = RAFdataset2()
        train_ds, val_ds = dataset.load_data(stage=1, channel=3)
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'val', val_ds, args.batch_size).dataloader_sup()
        test_dataset = val_dataset

    elif args.source == 'fer2013plus':
        dataset = Fer2013Plus()
        train_ds, val_ds = dataset.load_data(stage=1)
        test_ds = dataset.load_data(stage=2)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'val', val_ds, args.batch_size).dataloader_sup()
        test_dataset = DataLoader(args, 'lincls', 'val', test_ds, args.batch_size).dataloader_sup()

    class_sum = [0] * 7

    for j,i in train_ds:
        class_sum[i] += 1
    logger.info('{}'.format(class_sum))
    class_sum /= np.sum(class_sum)
    density = np.asarray([pow(1 - i, 1.5) for i in class_sum])
    class_weights={}
    for i in range(7):
        class_weights[i]=density[i]

    logger.info('{}'.format(density))

    net = CCD(args, logger)
    model = get_classifier(net, args)
    model.summary()
    decay_steps = tf.math.ceil(dataset.train_image_num / args.batch_size) * args.lincls_epoch
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.01,
                                                      decay_steps=decay_steps,
                                                      alpha=0.0001)
    #lr_decayed_fn=tf.keras.callbacks.LearningRateScheduler(scheduler)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn,momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(output_path, 'lincls.log'))
    model.fit(train_dataset, validation_data=fer_test_dataset, epochs=20, callbacks=[csv_logger],class_weight=class_weights)
    loss, acc = model.evaluate(test_dataset)
    loss, acc = model.evaluate(fer_test_dataset)
    logger.info("loss:{},acc:{}".format(loss, acc))
    # only save classify-head
    model.layers[-1].save(output_path)
    model.layers[-2].save(encoder_output_path)

    end_time = create_stamp()
    logger.info(end_time)


if __name__ == '__main__':
    train_classifier()
