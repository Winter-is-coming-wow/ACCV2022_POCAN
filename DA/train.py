# -- coding: utf-8 --
from cmath import log
import tensorflow as tf
from tensorflow_addons.optimizers import SGDW, AdamW
import os, math
import gc
from util import get_logger, get_session, create_stamp, set_seed
from losses import contrastive_loss,EMLoss,SSL_contrastive_loss,  ce_proto, EMLoss_with_logit, ce_loss
from model import POCAN
from Dataloader import set_dataset, DataLoader, RAFdataset, Fer2013dataset, Fer2013Plus, ExpW, CKplus, oulu_CASIA, JAFFEdataset, SFEW
from gl import Config
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from util import get_index, get_T, decayed_lr, get_num_cluster, set_seed
from functools import reduce
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from collections import Counter


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


def calculate_prototype(embeddings, y):
    label_0_index = tf.where(y == 0)
    anchors_0 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_0_index), axis=0),
                                      tf.cast(tf.size(label_0_index), tf.float32))
    label_1_index = tf.where(y == 1)
    anchors_1 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_1_index), axis=0),
                                      tf.cast(tf.size(label_1_index), tf.float32))
    label_2_index = tf.where(y == 2)
    anchors_2 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_2_index), axis=0),
                                      tf.cast(tf.size(label_2_index), tf.float32))
    label_3_index = tf.where(y == 3)
    anchors_3 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_3_index), axis=0),
                                      tf.cast(tf.size(label_3_index), tf.float32))
    label_4_index = tf.where(y == 4)
    anchors_4 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_4_index), axis=0),
                                      tf.cast(tf.size(label_4_index), tf.float32))
    label_5_index = tf.where(y == 5)
    anchors_5 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_5_index), axis=0),
                                      tf.cast(tf.size(label_5_index), tf.float32))
    label_6_index = tf.where(y == 6)
    anchors_6 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_6_index), axis=0),
                                      tf.cast(tf.size(label_6_index), tf.float32))
    anchors = np.array([anchors_0, anchors_1, anchors_2, anchors_3, anchors_4, anchors_5, anchors_6])
    for i in range(len(anchors)):
        anchors[i] = tf.math.divide_no_nan(anchors[i], np.sqrt(np.sum(anchors[i] * anchors[i])))
    anchors = np.squeeze(anchors)

    return anchors


def train_step(data, model, optimizer, opt, args, lam, source_class_weight, target_class_weight):
    source_batch, target_batch, target_batch_untrust = data
    target_trust_bz = target_batch[0].shape[0]
    # target_untrust_bz = target_batch_untrust[0].shape[0]

    source_x = tf.split(source_batch[0], num_or_size_splits=2, axis=-1)
    source_x = tf.cast(tf.concat(source_x, axis=0), tf.float32)
    source_y = tf.cast(tf.concat([source_batch[1], source_batch[1]], axis=-1), tf.int64)

    target_x = tf.split(tf.concat([target_batch[0], target_batch_untrust[0]], axis=0), num_or_size_splits=2, axis=-1)
    target_x = tf.cast(tf.concat(target_x, axis=0), tf.float32)
    target_trust_y = tf.cast(tf.concat([target_batch[1], target_batch[1]], axis=-1), tf.int64)

    with tf.GradientTape(persistent=True) as tape:
        source_embedding = model.encoder(source_x, training=True)
        source_projection = model.proj(source_embedding, training=True)

        target_embedding = model.encoder(target_x, training=True)
        target_projection = model.proj(target_embedding, training=True)

        source_logits_s = model.source_classify_head(source_projection, training=False)
        source_logits_t = model.target_classify_head(source_projection, training=False)

        projection1, projection2 = tf.split(target_projection, 2, 0)
        target_projection_trust = tf.concat([projection1[0:target_trust_bz], projection2[0:target_trust_bz]], axis=0)
        target_projection_untrust = tf.concat([projection1[target_trust_bz:], projection2[target_trust_bz:]], axis=0)

        target_logits_t = model.target_classify_head(target_projection_trust, training=False)
        target_logits_s = model.source_classify_head(target_projection_trust, training=False)

        target_untrust_logit_t = model.target_classify_head(target_projection_untrust, training=False)
        target_untrust_logit_s = model.source_classify_head(target_projection_untrust, training=False)

        # compute losses
        loss_source = contrastive_loss(source_y, source_projection)
        loss_target = SSL_contrastive_loss(target_projection, temperature=args.temperature)

        loss_indomain_source = ce_loss(source_y, source_logits_s, source_class_weight)
        loss_indomain_target = ce_loss(target_trust_y, target_logits_t, target_class_weight) * lam
        loss_in_domain = loss_indomain_source + loss_indomain_target

        loss_s2t = ce_loss(source_y, source_logits_t, source_class_weight)
        loss_t2s = ce_loss(target_trust_y, target_logits_s, target_class_weight)
        loss_cross_domain = loss_s2t + loss_t2s

        Pst_source = tf.nn.softmax(tf.concat([source_logits_s, source_logits_t], axis=1))
        Pst_target = tf.nn.softmax(tf.concat([tf.concat([target_logits_s, target_untrust_logit_s], axis=0),
                                              tf.concat([target_logits_t, target_untrust_logit_t], axis=0)], axis=1))
        loss_pst_s = -tf.reduce_mean(tf.math.log(tf.reduce_sum(Pst_source[:, 7:], axis=1)))
        loss_pst_t = -tf.reduce_mean(tf.math.log(tf.reduce_sum(Pst_target[:, :7], axis=1)))
        loss_pst = loss_pst_s + loss_pst_t

        loss_encoder_proj = loss_target + loss_source + loss_in_domain + loss_cross_domain# + loss_pst

    trainable_var = model.encoder.trainable_variables + model.proj.trainable_variables
    gradients = tape.gradient(loss_encoder_proj, trainable_var)
    optimizer.apply_gradients(zip(gradients, trainable_var))

    with tf.GradientTape(persistent=True) as tape:
        source_embedding = model.encoder(source_x, training=False)
        source_projection = model.proj(source_embedding, training=False)

        target_embedding = model.encoder(target_x, training=False)
        target_projection = model.proj(target_embedding, training=False)

        source_logits_s = model.source_classify_head(source_projection, training=True)
        source_logits_t = model.target_classify_head(source_projection, training=True)

        projection1, projection2 = tf.split(target_projection, 2, 0)
        target_projection_trust = tf.concat([projection1[0:target_trust_bz], projection2[0:target_trust_bz]], axis=0)
        target_projection_untrust = tf.concat([projection1[target_trust_bz:], projection2[target_trust_bz:]], axis=0)

        target_logits_t = model.target_classify_head(target_projection_trust, training=True)
        target_logits_s = model.source_classify_head(target_projection_trust, training=True)

        target_untrust_logit_t = model.target_classify_head(target_projection_untrust, training=True)
        target_untrust_logit_s = model.source_classify_head(target_projection_untrust, training=True)

        loss_indomain_source = ce_loss(source_y, source_logits_s, source_class_weight)
        loss_indomain_target = ce_loss(target_trust_y, target_logits_t, target_class_weight) * lam

        # loss_s2t = ce_loss(source_y, source_logits_t, source_class_weight)
        # loss_t2s = ce_loss(target_trust_y, target_logits_s, target_class_weight)*lam
        # loss_cross_domain = loss_s2t + loss_t2s

        Pst_source = tf.nn.softmax(tf.concat([source_logits_s, source_logits_t], axis=1))
        Pst_target = tf.nn.softmax(tf.concat([tf.concat([target_logits_s, target_untrust_logit_s], axis=0),
                                              tf.concat([target_logits_t, target_untrust_logit_t], axis=0)], axis=1))
        loss_pst_s = -tf.reduce_mean(tf.math.log(tf.reduce_sum(Pst_source[:, :7], axis=1)))
        loss_pst_t = -tf.reduce_mean(tf.math.log(tf.reduce_sum(Pst_target[:, 7:], axis=1)))
        loss_pst1 = loss_pst_s + loss_pst_t

        loss_source_head = loss_indomain_source + loss_pst1

        loss_target_head = loss_indomain_target + loss_pst1

    trainable_var = model.source_classify_head.trainable_variables
    gradients = tape.gradient(loss_source_head, trainable_var)
    opt.apply_gradients(zip(gradients, trainable_var))

    trainable_var = model.target_classify_head.trainable_variables
    gradients = tape.gradient(loss_target_head, trainable_var)
    opt.apply_gradients(zip(gradients, trainable_var))

    # model.queue = tf.concat([tf.transpose(target_projection), model.queue], axis=-1)
    # model.queue = model.queue[:, :args.queue_size]

    results = {'loss_s2t': loss_s2t,
               'loss_t2s': loss_t2s,
               'loss_source': loss_source,
               'loss_indomain_source': loss_indomain_source,
               'loss_indomain_target': loss_indomain_target,
               'loss_em': 0,
               'loss_target': loss_target,
               'loss_pst': loss_pst,
               'loss_pst1': loss_pst1
               }
    return results


def train(source, target, args, logger, output_path):
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))
    start_time = create_stamp()
    logger.info(start_time)
    logger.info("------------------------------------------------------------------")
    net = POCAN(args, logger)
    net.encoder.trainable = True
    net.proj.trainable = True
    net.source_classify_head.trainable = True
    net.target_classify_head.trainable = True

    target_train_dataset = None
    target_train_dataset_untrust = None
    train_steps = None
    best_acc = {'epoch': 0, 'acc': 0}
    initial_acc = None

    def embedding_net(source=True):
        inputs = tf.keras.Input((args.img_size, args.img_size, args.channel))
        x = net.encoder(inputs, training=False)
        x = net.proj(x, training=False)
        if source:
            logits = net.source_classify_head(x, training=False)
        else:
            logits = net.target_classify_head(x, training=False)
        m = tf.keras.Model(inputs, logits)
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        return m

    source_ds = source['source_train_ds']
    class_sum = [0] * 7
    Label_Source = []
    for image, label in source_ds:
        label = transer(args.source, label)
        Label_Source.append(label)
        class_sum[label] += 1
    Label_Source = np.array(Label_Source)
    source_dataset = DataLoader(args, 'pretext', 'train', source_ds, args.batch_size).dataloader_sup()

    source_num = len(Label_Source)
    logger.info('source class weights{}'.format(class_sum))
    class_sum /= np.sum(class_sum)
    source_class_weight = np.asarray([pow(1 - i, 2) for i in class_sum])

    # target train label
    Label_Target = []
    for image, label in target['target_train_ds']:
        label = transer(args.target, label)
        Label_Target.append(label)
    total_num = len(Label_Target)
    Label_Target = np.array(Label_Target)

    # target test dataset
    Label_Target_test = []
    for image, label in target['target_test_ds']:
        label = transer(args.target, label)
        Label_Target_test.append(label)
    Label_Target_test = np.array(Label_Target_test)

    target_test_dataset = DataLoader(args, 'lincls', 'val', target['target_test_ds'], args.batch_size).dataloader()
    del target['target_test_ds']
    # train_steps=None
    source_train_dataset_evaluate = DataLoader(args, 'lincls', 'val', source_ds,
                                               args.batch_size).dataloader_sup()
    target_train_dataset_evaluate = DataLoader(args, 'lincls', 'val', target['target_train_ds'],
                                               args.batch_size).dataloader()


    for epoch in range(args.epochs):
        logger.info("{}/{}".format(epoch, args.epochs))
        lr = 0.00001
        optimizer = SGDW(weight_decay=args.weight_decay, learning_rate=lr, momentum=0.9)
        #lr = decayed_lr(epoch, args)
        logger.info("lr : {}".format(lr))
        opt = SGDW(weight_decay=args.weight_decay, learning_rate=lr, momentum=0.9)
        if epoch % args.interval == 0 or epoch == args.epochs - 1:

            classifier = embedding_net()
            _, acc = classifier.evaluate(source_train_dataset_evaluate)
            logger.info("evaluate source train data at epoch {},   acc : {}!".format(epoch, acc))
            # evaluate target with source classify head
            logits_with_source = classifier.predict(target_train_dataset_evaluate)
            logits_with_source = softmax(logits_with_source)
            target_pseudo_label_with_source = np.argmax(logits_with_source, axis=-1)
            acc = np.sum(target_pseudo_label_with_source == Label_Target) / Label_Target.shape[0]
            logger.info("evaluate target train data at epoch with source proto {},   acc : {}!".format(epoch, acc))

            # evaluate
            classifier = embedding_net(False)
            # evaluate target with target classify head
            target_logits = classifier.predict(target_train_dataset_evaluate)
            target_logits = softmax(target_logits)
            target_pseudo_label = np.argmax(target_logits, axis=-1)
            target_confidence = [target_logits[i][target_pseudo_label[i]] for i in range(len(target_pseudo_label))]
            target_confidence = np.asarray(target_confidence)

            confidence = [[], [], [], [], [], [], []]
            for i in range(len(target_pseudo_label)):
                confidence[target_pseudo_label[i]].append(target_confidence[i])
            avgc = np.asarray([np.mean(np.asarray(i)) for i in confidence])
            logger.info("{}".format(avgc))
            maxc = np.asarray([0 if len(i) == 0 else np.nanmax(np.asarray(i)) for i in confidence])
            logger.info("{}".format(maxc))
            confidence = avgc + (maxc - avgc) * (1 - np.power(avgc, args.beta))

            # confidence = np.asarray([np.exp((np.mean(np.asarray(i)) - 1)*0.8) for i in confidence])

            # confidence = np.asarray([np.mean(np.asarray(i)) for i in confidence])
            # confidence = (np.nanmax(confidence)) - (np.nanmax(confidence) - confidence) / (np.nanmax(confidence) - np.nanmin(confidence)) * 0.1
            logger.info('args.threshold:{}'.format(confidence))
            t = np.asarray(confidence)[target_pseudo_label]
            trust_index = np.squeeze(tf.where(target_confidence >= t))
            # Feature_Target_trust = Feature_Target[trust_index]
            target_pseudo_label_trust = target_pseudo_label[trust_index]
            c = Counter(target_pseudo_label_trust)
            logger.info('target_pseudo_label_trust {}'.format(dict(c)))

            # target class weight
            class_sum = [0] * 7
            for i in target_pseudo_label:
                class_sum[i] += 1
            logger.info('class num over pseudo label{}'.format(class_sum))
            class_sum /= np.sum(class_sum)
            target_class_weight = np.asarray([pow(1 - i, 1) for i in class_sum])
            logger.info('{}'.format(target_class_weight))

            # evaluate source with target classify head
            logits = classifier.predict(source_train_dataset_evaluate)
            logits = softmax(logits)
            source_pseudo_label_2 = np.argmax(logits, axis=-1)
            acc = np.sum(source_pseudo_label_2 == Label_Source) / Label_Source.shape[0]
            logger.info("evaluate source train data at epoch with target proto {},   acc : {}!".format(epoch, acc))

            # evaluate trust target train dataset

            Label_Target_trust = Label_Target[trust_index]
            c = Counter(Label_Target_trust)
            logger.info('lable target{}'.format(dict(c)))
            # target_confidence=[logits[i][pseudo_label[i]] for i in range (len(pseudo_label))]
            acc = np.sum(target_pseudo_label_trust == Label_Target_trust) / Label_Target_trust.shape[0]
            logger.info("evaluate trust target train data at epoch {},   acc : {}!".format(epoch, acc))
            acc = np.sum(target_pseudo_label == Label_Target) / Label_Target.shape[0]
            logger.info("evaluate target train data at epoch {},   acc : {}!".format(epoch, acc))
            logger.info('{}'.format(classification_report(Label_Target, target_pseudo_label)))
            cfm = confusion_matrix(target_pseudo_label, Label_Target, labels=[0, 1, 2, 3, 4, 5, 6],
                                   normalize='true')
            logger.info('confuse matrix :{}'.format(cfm))

            logits_target_test = softmax(classifier.predict(target_test_dataset))
            acc_target_test = np.sum(np.argmax(logits_target_test, axis=-1) == Label_Target_test) / \
                              Label_Target_test.shape[0]
            if acc_target_test > best_acc['acc']:
                best_acc['acc'] = acc_target_test
                best_acc['epoch'] = epoch
            if epoch == 0:
                initial_acc = acc_target_test

            logger.info("evaluate target test data at epoch {},   acc : {}!".format(epoch, acc_target_test))

            target_pseudo_label = tf.data.Dataset.from_tensor_slices(target_pseudo_label)
            target_confidence = tf.data.Dataset.from_tensor_slices(target_confidence)
            target_train_ds = tf.data.Dataset.zip((target['target_train_ds'], target_pseudo_label, target_confidence))

            # 挑选置信度高于阈值的数据
            confidence = tf.convert_to_tensor(confidence)
            target_train_ds_trust = target_train_ds.filter(
                lambda x, y, z: z >= tf.cast(tf.gather_nd(confidence, tf.reshape(y, (1,))), tf.float32))
            target_train_ds_untrust = target_train_ds.filter(
                lambda x, y, z: z < tf.cast(tf.gather_nd(confidence, tf.reshape(y, (1,))), tf.float32))

            target_train_trust_len = len(target_pseudo_label_trust)
            target_train_untrust_len = len(Label_Target) - target_train_trust_len
            trust_bz = int(
                target_train_trust_len / (target_train_untrust_len + target_train_trust_len) * args.batch_size)
            untrust_bz = args.batch_size - trust_bz

            logger.info('source_num:{}'.format(source_num))
            logger.info('target_train_trust_len:{}'.format(target_train_trust_len))
            logger.info('target_train_untrust_len:{}'.format(target_train_untrust_len))
            train_steps = int(max(source_num / args.batch_size,
                                  max(target_train_trust_len / trust_bz, target_train_untrust_len / untrust_bz)))
            target_train_dataset = DataLoader(args, 'pretext', 'train', target_train_ds_trust,
                                              trust_bz).dataloader()
            target_train_dataset_untrust = DataLoader(args, 'pretext', 'train', target_train_ds_untrust,
                                                      untrust_bz).dataloader()

        step = 0
        # lam = 1 / (1 + tf.math.exp(-1 * 10 * epoch / args.epochs))
        lam = 2 / (1 + tf.math.exp(-1 * 10 * epoch / args.epochs)) - 1
        beta = max(1 - epoch / args.epochs, 0.1)
        logger.info('beta: {}'.format(beta))
        logger.info("train_steps: {}".format(train_steps))
        logger.info("lam: {}".format(lam))
        for source_batch, target_batch, target_batch_untrust in zip(source_dataset, target_train_dataset,
                                                                    target_train_dataset_untrust):
            step += 1
            if step > train_steps:
                break
            # source_batch = reduce(lambda x, y: (tf.concat([x[0], y[0]], 0), tf.concat([x[1], y[1]], 0)), source_batch)
            # images,labels,confidence=batch_shuffle(images,labels,confidence)
            results = train_step((source_batch, target_batch, target_batch_untrust), net,
                                 optimizer, opt, args, lam, source_class_weight, target_class_weight)
            logger.info(
                "{}/{} loss_s2t:{}  loss_t2s {}  loss_source:{}  loss_em:{}  loss_target:{}  loss_indomain_source':{} "
                " loss_indomain_target':{}  loss_pst:{} loss_pst1:{}"
                    .format(step, train_steps,
                            results['loss_s2t'],
                            results['loss_t2s'],
                            results['loss_source'],
                            results['loss_em'],
                            results['loss_target'],
                            results['loss_indomain_source'],
                            results['loss_indomain_target'],
                            results['loss_pst'],
                            results['loss_pst1']
                            ))
    logger.info('initial acc:{}'.format(initial_acc))
    logger.info('best acc:{}'.format(best_acc))
    logger.info("save model ")
    net.encoder.save(os.path.join(output_path, "backbone"))
    net.proj.save(os.path.join(output_path, "proj"))
    net.source_classify_head.save(os.path.join(output_path, "cls_s"))
    net.target_classify_head.save(os.path.join(output_path, "cls_t"))
    del net, source, target
    end_time = create_stamp()

    logger.info(end_time)
    logger.info("------------------------------------------------------------------")
    return best_acc, initial_acc


def main():
    args = Config()
    get_session(args)
    set_seed()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    output_path = os.path.join(args.output_path, "{}_{}_{}".format(args.source, args.target, args.train_info))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    logger = get_logger("MyLogger", output_path)

    source_train_ds = None
    # source_val_ds = None
    source_test_ds = None

    target_train_ds = None
    # target_val_ds = None
    target_test_ds = None

    if args.source == 'fer2013':
        dataset = Fer2013dataset()
        source_train_ds, source_val_ds = dataset.load_data(stage=1)
        source_test_ds = dataset.load_data(stage=2)

    elif args.source == 'fer2013plus':
        dataset = Fer2013Plus('Majority')
        source_train_ds, source_val_ds = dataset.load_data(stage=1)
        source_test_ds = dataset.load_data(stage=2)

    elif args.source == 'RAF':

        dataset = RAFdataset()
        source_train_ds, source_test_ds = dataset.load_data(stage=1, channel=3)
    elif args.source == 'ExpW':
        dataset = ExpW()
        train_ds = dataset.load_data(type=1)
        # source_train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup

    else:
        raise Exception("no such source dataset")

    source = {'source_train_ds': source_train_ds, 'source_test_ds': source_test_ds}

    if args.target == 'fer2013':
        dataset = Fer2013dataset()
        target_train_ds = dataset.load_data(stage=0)
        target_test_ds = dataset.load_data(stage=2)
    elif args.target == 'fer2013plus':
        dataset = Fer2013Plus('Majority')
        target_train_ds, target_val_ds = dataset.load_data(stage=1)
        target_test_ds = dataset.load_data(stage=2)
    elif args.target == 'RAF':
        dataset = RAFdataset()
        target_train_ds, target_val_ds = dataset.load_data(stage=1, channel=1)
        target_test_ds = dataset.load_data(stage=2, channel=1)

    elif args.target == 'ExpW':
        dataset = ExpW()
        target_train_ds, target_test_ds = dataset.load_data(mode=0)
        # target_test_ds = dataset.load_data(mode=1)
        # source_train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup
    elif args.target == 'sfew':
        dataset = SFEW()
        target_train_ds, target_test_ds = dataset.load_data(channels=3)

    elif args.target == 'ckplus':
        dataset = CKplus()
        train_ds, test_ds = dataset.load_data()
        target_train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
        target_test_ds = tf.data.Dataset.from_tensor_slices(test_ds)

    elif args.target == 'oulu':
        dataset = oulu_CASIA()
        train_ds, test_ds = dataset.load_data()
        target_train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
        target_test_ds = tf.data.Dataset.from_tensor_slices(test_ds)

    elif args.target == 'JAFFE':
        dataset = JAFFEdataset()
        train_ds, test_ds = dataset.load_data()
        target_train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
        target_test_ds = tf.data.Dataset.from_tensor_slices(test_ds)

    elif args.target == "AffectNet":
        dataset = AffectNet()
        target_train_ds, target_test_ds = dataset.load_data(mode=1)


    else:
        raise Exception("no such target dataset")

    # if args.source != 'RAF':
    #     source = {'source_train_ds': source_train_ds, 'source_val_ds': source_val_ds, 'source_test_ds': source_test_ds}
    target = {'target_train_ds': target_train_ds, 'target_test_ds': target_test_ds}

    train(source, target, args, logger, output_path)


def softmax(x):
    # x_row_max = x.max(axis=-1)
    # x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    # x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


if __name__ == '__main__':
    main()
