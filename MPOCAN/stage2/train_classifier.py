# -- coding: utf-8 --
import tensorflow as tf
from tensorflow_addons.optimizers import SGDW, AdamW
import os
from util import get_logger, get_session, create_stamp
from functools import reduce
from model import MoCo
from Dataloader import set_dataset, DataLoader, RAFdataset2, Fer2013dataset, Fer2013Plus, ExpW,CKplus, oulu_CASIA, \
     JAFFEdataset, SFEW
from gl import Config
import numpy as np




def get_classifier(net, args,index):
    """
    return a classify model
    """
    inputs = tf.keras.Input((args.img_size, args.img_size, args.channel))
    net.encoder.trainable = False
    net.proj[index].trainable=False
    net.source_classifier[index].trainable=True
    x = net.encoder(inputs, training=False)
    x = net.proj[index](x, training=False)
    x = net.source_classifier[index](x, training=True)
    return tf.keras.Model(inputs, x)

def scheduler(epoch):
    if epoch < 5:
        return 0.001
    elif epoch<10:
        return 0.0001
    else:
        return 0.00001

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

def train_classifier():
    """
    use source train dataset train a classifier based on pretrained backbone
    """
    args = Config()
    args.backbone='resnet50'
    args.encoder_snapshot = r'C:\Users\DELL\PycharmProjects\superwang\final\DA\multi-POCAN\two\stage1\output\JAFFE_sgdw_cJo\backbone'
    args.proj_snapshot = r'C:\Users\DELL\PycharmProjects\superwang\final\DA\multi-POCAN\two\stage1\output\JAFFE_sgdw_cJo'
    args.classifier_snapshot=None

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    # output_path = os.path.join(args.output_path, "classifier")
    #encoder_output_path = os.path.join(args.output_path, "source_encoder_with_proj_adam_norm")
    get_session(args)
    logger = get_logger("MyLogger_classifier", args.output_path)
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))
    start_time = create_stamp()
    logger.info(start_time)
    logger.info("------------------------------------------------------------------")

    if args.target == 'fer2013':
        dataset = Fer2013dataset()
        test_ds = dataset.load_data(stage=2)
    elif args.target == 'RAF':
        dataset=RAFdataset2()
        test_ds=dataset.load_data(stage=2,channel=3)
    elif args.target == 'ckplus':
        dataset = CKplus()
        _, test_ds = dataset.load_data()
        test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    elif args.target == 'JAFFE':
        dataset = JAFFEdataset()
        train_ds, test_ds = dataset.load_data()
        test_ds=tf.data.Dataset.from_tensor_slices(test_ds)
    elif args.target == 'oulu':
        dataset = oulu_CASIA()
        train_ds, test_ds = dataset.load_data()
        test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    elif args.target == 'sfew':
        dataset = SFEW()
        train_ds, test_ds = dataset.load_data(channels=3)
    else:
        test_ds=None

    fer_test_dataset = DataLoader(args, 'lincls', 'val', test_ds, args.batch_size).dataloader()

    sources = []
    for ds_name in args.sources:
        if ds_name == 'fer2013':
            dataset = Fer2013dataset()
            train_ds, val_ds = dataset.load_data(stage=1)

            train_ds = train_ds.map(lambda x, y: (x, transer(ds_name, y)))
            sources.append(train_ds)

        elif ds_name == 'RAF':
            dataset = RAFdataset2()
            train_ds, val_ds = dataset.load_data(stage=1, channel=3)
            # test_ds = dataset.load_data(stage=2, channel=3)
            # total_ds = test_ds.concatenate(train_ds.concatenate(val_ds))
            train_ds = train_ds.map(lambda x, y: (x, transer(ds_name, y)))
            sources.append(train_ds)

        elif ds_name == 'JAFFE':
            dataset = JAFFEdataset()
            train_ds, test_ds = dataset.load_data()
            train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
            train_ds = train_ds.map(lambda x, y: (x, transer(ds_name, y)))
            # test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
            # total_ds = test_ds.concatenate(train_ds)
            sources.append(train_ds)

        elif ds_name == 'ckplus':
            dataset = CKplus()
            train_ds, test_ds = dataset.load_data()
            train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
            # test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
            # total_ds = test_ds.concatenate(train_ds)
            train_ds = train_ds.map(lambda x, y: (x, transer(ds_name, y)))
            sources.append(train_ds)

        elif ds_name == 'oulu':
            dataset = oulu_CASIA()
            train_ds, test_ds = dataset.load_data()
            train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
            # test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
            # total_ds = test_ds.concatenate(train_ds)
            train_ds = train_ds.map(lambda x, y: (x, transer(ds_name, y)))
            sources.append(train_ds)

        elif ds_name == 'sfew':
            dataset = SFEW()
            train_ds, test_ds = dataset.load_data(channels=3)
            # total_ds = test_ds.concatenate(train_ds)
            train_ds = train_ds.map(lambda x, y: (x, transer(ds_name, y)))
            sources.append(train_ds)

    net = MoCo(args, logger)
    for index in range(len(args.sources)):
        logger.info(args.sources[index])
        output_path = os.path.join(args.output_path, "classifier_{}".format(args.sources[index]))
        logger.info("output for {} : {}".format(args.sources[index],output_path))
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        train_dataset = DataLoader(args, 'lincls', 'val', sources[index], args.batch_size).dataloader_sup()
        class_sum = [0] * 7

        for j,i in sources[index]:
            class_sum[i] += 1
        logger.info('{}'.format(class_sum))
        total_num = np.sum(class_sum)
        class_sum /= np.sum(class_sum)
        density = np.asarray([pow(1 - i, 2) for i in class_sum])
        class_weights={}
        for i in range(7):
            class_weights[i]=density[i]
        logger.info('{}'.format(density))

        model = get_classifier(net, args,index)
        model.summary()
        decay_steps = tf.math.ceil(np.sum(total_num)/ args.batch_size) * args.lincls_epoch
        logger.info("decay_steps: {}".format(decay_steps))
        lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.01,
                                                          decay_steps=decay_steps,
                                                          alpha=0.0001)
        #lr_decayed_fn=tf.keras.callbacks.LearningRateScheduler(scheduler)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn,momentum=0.9)
        optimizer = SGDW(weight_decay=args.weight_decay, learning_rate=lr_decayed_fn, momentum=0.9)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(output_path, 'lincls.log'))
        model.fit(train_dataset, validation_data=fer_test_dataset, epochs=args.lincls_epoch, callbacks=[csv_logger])
        loss, acc = model.evaluate(fer_test_dataset)
        logger.info("loss:{},acc:{}".format(loss, acc))
        # only save classify-head
        model.layers[-1].save(output_path)
        #model.layers[-2].save(encoder_output_path)

        end_time = create_stamp()
        logger.info(end_time)


if __name__ == '__main__':
    train_classifier()
