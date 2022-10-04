from re import T
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from Dataloader import Fer2013dataset,Fer2013Plus,RAFdataset,ExpW,CKplus,DataLoader,oulu_CASIA
import cv2,os
from model import CCD
from gl import Config
from util import get_session,get_logger,create_stamp
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
from clustering import run_kmeans


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
    else:
        raise ("no such dataset,can not transform label")
    return label


def Visualization(net, datasetName,args,with_proto=False):
    '''Feature Visualization in Source/Target Domain.'''
    model=tf.keras.Sequential([net.encoder,net.proj])
    model.summary()
    
    Feature, Label = [], []

    if datasetName == 'fer2013':
        dataset = Fer2013dataset()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader()
        val_dataset = DataLoader(args, 'lincls', 'val', val_ds, args.batch_size).dataloader()
        pred = model.predict(train_dataset)
        print(pred.shape)
        for feature in pred:
            Feature.append(feature)
        for image, label in train_ds:
            Label.append(label)
    elif datasetName == 'RAF':
        dataset = RAFdataset()
        tf_list=dataset.load_data(channel=3)
        train_ds=reduce(lambda x,y:x.concatenate(y),tf_list)
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        print(pred.shape)
        for feature in pred:
            Feature.append(feature)
        for image, label in train_ds:
            Label.append(label)

    elif datasetName == 'fer2013plus':
        dataset = Fer2013Plus()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'val', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        
        for feature in pred:
            Feature.append(feature)
        for image, label in train_ds:
            Label.append(label)
    
    elif datasetName == 'fer2013part':
        dataset = Fer2013dataset()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_ds=part_target(net,args,train_ds)
        train_dataset=DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader_vis()
        pred = model.predict(train_dataset)
        
        for feature in pred:
            Feature.append(feature)
        for value, pseual_label,confidence in train_ds:
            Label.append(value[1])
    
    elif datasetName=='ckplus':
        dataset=CKplus()
        images,labels=dataset.load_data()
        train_ds=tf.data.Dataset.from_tensor_slices((images,labels))
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature.append(feature)
        for image, label in train_ds:
            Label.append(transer('ckplus',label))

    elif datasetName=='oulu':
        dataset=oulu_CASIA()
        images,labels=dataset.load_data()
        train_ds=tf.data.Dataset.from_tensor_slices((images,labels))
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature.append(feature)
        for image, label in train_ds:
            Label.append(transer('oulu',label))
    
    if(with_proto):
        proto_feature=calculate_prototype(np.array(Feature),np.array(Label))
        for i in proto_feature:
            i=np.squeeze(i)
            #print(i)
            Feature.append(i)
    
        Label_proto=[10,11,12,13,14,15,16]
        Label.extend(Label_proto)
    

    Label=np.array(Label)

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=50)
    embedding = tsne.fit_transform(Feature)
    # Draw Visualization of Feature
    if with_proto:
        colors = {0: 'tomato', 1: 'cornflowerblue', 2: 'cyan', 3: 'lime', 4: 'orange', 5: 'purple', 6: 'lightgrey',
        10: 'purple', 11: 'yellow', 12: 'black', 13: 'black', 14: 'white', 15: 'white', 16: 'red'}
                #10: 'black', 11: 'black', 12: 'black', 13: 'black',14: 'black', 15: 'black', 16: 'black'}
        labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral',
        10: 'Angry_proto',11: 'Disgust_proto', 12: 'Fear_proto', 13: 'Happy_proto', 14: 'Sad_proto',
            15: 'Surprised_proto', 16: 'Neutral_proto'}
    else:
        colors = {0: 'tomato', 1: 'cornflowerblue', 2: 'lightblue', 3: 'green', 4: 'orange', 5: 'purple', 6: 'lightgrey'}
        labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    
    for i in range(7):
        data_x, data_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        scatter = plt.scatter(data_x, data_y,  edgecolors=colors[i], s=5, label=labels[i], marker='^', alpha=0.6)
    if with_proto:
        for i in range(10,17):
            data_x, data_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
            scatter = plt.scatter(data_x, data_y,  edgecolors=colors[i], s=30, label=labels[i], marker='*')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(7)],
               loc='upper left', prop={'size': 8}, bbox_to_anchor=(1.05, 0.85), borderaxespad=0)
    plt.savefig(fname='{}_{}_{}.png'.format(args.source,datasetName,args.train_info), bbox_inches='tight')


def VisualizationForTwoDomain(net, source_dataset, target_dataset,args,with_proto=False):
    '''Feature Visualization in Source and Target Domain.'''
    proto_s=None
    model=tf.keras.Sequential([net.encoder,net.proj])
    Feature_Source, Label_Source = [], []

    if source_dataset == 'fer2013':
        dataset = Fer2013dataset()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Source.append(feature)
        for image, label in train_ds:
            Label_Source.append(label)
    elif source_dataset == 'RAF':
        dataset = RAFdataset()
        tf_list=dataset.load_data(channel=3)
        train_ds=reduce(lambda x,y:x.concatenate(y),tf_list)
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        print(pred.shape)
        for feature in pred:
            Feature_Source.append(feature)
        for image, label in train_ds:
            Label_Source.append(transer('RAF',label))
    elif source_dataset == 'fer2013plus':
        dataset = Fer2013Plus()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Source.append(feature)
        for image, label in train_ds:
            Label_Source.append(label)
    if with_proto:
        proto_s=calculate_prototype(np.array(Feature_Source),np.array(Label_Source))
        for i in proto_s:
            Feature_Source.append(np.squeeze(i))
        Label_Source.extend([14,15,16,17,18,19,20])


    Feature_Target, Label_Target = [], []

    # Get Feature and Label in Target Domain
    if target_dataset == 'fer2013':
        dataset = Fer2013dataset()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader()
        val_dataset = DataLoader(args, 'lincls', 'val', val_ds, args.batch_size).dataloader()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Target.append(feature)
        for image, label in train_ds:
            Label_Target.append(transer('fer2013',label))
    elif target_dataset == 'RAF':
        dataset = RAFdataset()
        train_ds, val_ds = dataset.load_data(stage=1, channel=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Target.append(feature)
        for image, label in train_ds:
            Label_Target.append(label)
    elif target_dataset == 'fer2013plus':
        dataset = Fer2013Plus()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Target.append(feature)
        for image, label in train_ds:
            Label_Target.append(label)
    elif target_dataset == 'fer2013part':
        dataset = Fer2013dataset()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_ds=part_target(net,args,train_ds)
        train_dataset=DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_vis()
        pred = model.predict(train_dataset)
        
        for feature in pred:
            Feature_Target.append(feature)
        for value, pseual_label,confidence in train_ds:
            Label_Target.append(value[1])
    elif target_dataset=='ckplus':
        dataset=CKplus()
        images,labels=dataset.load_data()
        train_ds=tf.data.Dataset.from_tensor_slices((images,labels))
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Target.append(feature)
        for image, label in train_ds:
            Label_Target.append(transer('ckplus',label))
    elif target_dataset=='oulu':
        dataset=oulu_CASIA()
        images,labels=dataset.load_data()
        train_ds=tf.data.Dataset.from_tensor_slices((images,labels))
        train_dataset = DataLoader(args, 'lincls', 'val', train_ds, args.batch_size).dataloader()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Target.append(feature)
        for image, label in train_ds:
            Label_Target.append(transer('oulu',label))

    if with_proto:
        #proto_t=calculate_prototype(np.array(Feature_Target),np.array(Label_Target))
        proto_t=run_kmeans(np.asarray(Feature_Target),np.asarray(proto_s))
        logits = cosine_similarity(np.asarray(Feature_Target), proto_t)
        Label_Target = np.argmax(logits, axis=-1).tolist()
        
        for i in proto_t:
            Feature_Target.append(np.squeeze(i))
        Label_Target.extend([14,15,16,17,18,19,20])

    Label_Source=np.array(Label_Source)
    Label_Target=np.array(Label_Target)
    Label_Target += 7

    Feature = Feature_Source+Feature_Target
    Label = np.append(Label_Source, Label_Target)
    print(np.array(Feature).shape)
    print(Label.shape)
    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=20)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    # colors = {0: 'red', 1: 'blue', 2: 'lightblue', 3: 'green', 4: 'orange', 5: 'purple', 6: 'darkslategray', \
    #           7: 'red', 8: 'blue', 9: 'lightblue', 10: 'green', 11: 'orange', 12: 'purple', 13: 'darkslategray'}
    if with_proto:
        labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral', \
            7: 'Angry', 8: 'Disgust', 9: 'Fear', 10: 'Happy', 11: 'Sad', 12: 'Surprised', 13: 'Neutral', \
              14: 'Angry_source', 15: 'Disgust_source', 16: 'Fear_source', 17: 'Happy_source', 18: 'Sad_source',\
              19: 'Surprised_source', 20: 'Neutral_source',\
              21: 'Angry_target', 22: 'Disgust_target', 23: 'Fear_target', 24: 'Happy_target', 25: 'Sad_target',\
              26: 'Surprised_target', 27: 'Neutral_target',}
        # colors = {0: 'red', 1: 'blue', 2: 'lightblue', 3: 'green', 4: 'orange', 5: 'purple', 6: 'darkslategray',\
        #       7: 'silver', 8: 'blue', 9: 'lightblue', 10: 'green', 11: 'orange', 12: 'purple', 13: 'darkslategray',\
        #       14: 'black', 15: 'black', 16: 'black', 17: 'black', 18: 'black',19: 'black', 20: 'black',\
        #       21: 'red', 22: 'red', 23: 'red', 24: 'red', 25: 'red',26: 'red', 27: 'red'}
        colors = {0: 'deepskyblue', 1: 'deepskyblue', 2: 'deepskyblue', 3: 'deepskyblue', 4: 'deepskyblue', 5: 'deepskyblue', 6: 'deepskyblue', \
              7: 'red', 8: 'blue', 9: 'lightblue', 10: 'green', 11: 'orange', 12: 'purple', 13: 'darkslategray',
              14: 'black', 15: 'red', 16: 'green', 17: 'orange', 18: 'purple',19: 'silver', 20: 'blue',\
              21: 'black', 22: 'red', 23: 'green', 24: 'orange', 25: 'purple',26: 'silver', 27: 'blue'}
    else:
        labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral', \
                7: 'Angry', 8: 'Disgust', 9: 'Fear', 10: 'Happy', 11: 'Sad',12: 'Surprised', 13: 'Neutral'}
        colors = {0: 'deepskyblue', 1: 'deepskyblue', 2: 'deepskyblue', 3: 'deepskyblue', 4: 'deepskyblue', 5: 'deepskyblue', 6: 'deepskyblue', \
                7: 'silver', 8: 'silver', 9: 'silver', 10: 'silver', 11: 'silver',12: 'silver', 13: 'silver'}


    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(28):

        data_x, data_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]

        if i < 7:
            scatter = plt.scatter(data_x, data_y, c=colors[i], s=3, label=labels[i], marker='^',alpha=0.6)
        elif i<14:
            scatter = plt.scatter(data_x, data_y, c=colors[i], s=3, label=labels[i], marker='o',alpha=0.2)
        elif i<21:
            scatter = plt.scatter(data_x, data_y, c=colors[i], s=30, label=labels[i], marker='*',alpha=0.9)
        else:
            scatter = plt.scatter(data_x, data_y, c=colors[i], s=30, label=labels[i], marker='^',alpha=0.9)


        if i == 0:
            source = scatter
        elif i == 7:
            target = scatter
        elif i==14:
            source_proto=scatter
        elif i==21:
            target_proto=scatter


    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(7)],
                    loc='upper left', prop={'size': 8}, bbox_to_anchor=(1.05, 0.85), borderaxespad=0)
    plt.legend([source, target,source_proto,target_proto], ['Source Domain', 'Target Domain','source_proto','target_proto'], loc='upper left', prop={'size': 7},
               bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
    plt.gca().add_artist(l1)

    plt.savefig(fname='{}_{}_{}_{}.png'.format(args.source,source_dataset,target_dataset,args.train_info), bbox_inches='tight')


def VisualizationForPreDomain(net, source_dataset, target_dataset,args,with_proto=False):
    '''Feature Visualization in Source and Target Domain.'''
    model=tf.keras.Sequential([net.encoder,net.proj])
    Feature_Source, Label_Source = [], []

    if source_dataset == 'fer2013':
        dataset = Fer2013dataset()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Source.append(feature)
        for image, label in train_ds:
            Label_Source.append(label)
    elif source_dataset == 'RAF':
        dataset = RAFdataset()
        train_ds, val_ds = dataset.load_data(stage=1, channel=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Source.append(feature)
        for image, label in train_ds:
            Label_Source.append(label)
    elif source_dataset == 'fer2013plus':
        dataset = Fer2013Plus()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Source.append(feature)
        for image, label in train_ds:
            Label_Source.append(label)
    if with_proto:
        proto_s=calculate_prototype(np.array(Feature_Source),np.array(Label_Source))
        for i in proto_s:
            Feature_Source.append(np.squeeze(i))
        Label_Source.extend([10,11,12,13,14,15,16])


    Feature_Target, Label_Target = [], []

    # Get Feature and Label in Target Domain
    if target_dataset == 'fer2013':
        dataset = Fer2013dataset()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Target.append(feature)
        for image, label in train_ds:
            Label_Target.append(label)
    elif target_dataset == 'RAF':
        dataset = RAFdataset()
        train_ds, val_ds = dataset.load_data(stage=1, channel=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Target.append(feature)
        for image, label in train_ds:
            Label_Target.append(label)
    elif target_dataset == 'fer2013plus':
        dataset = Fer2013Plus()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_dataset = DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_sup()
        val_dataset = DataLoader(args, 'lincls', 'train', val_ds, args.batch_size).dataloader_sup()
        pred = model.predict(train_dataset)
        for feature in pred:
            Feature_Target.append(feature)
        for image, label in train_ds:
            Label_Target.append(label)
    elif target_dataset == 'fer2013part':
        dataset = Fer2013dataset()
        train_ds, val_ds = dataset.load_data(stage=1)
        train_ds=part_target(net,args,train_ds)
        train_dataset=DataLoader(args, 'lincls', 'train', train_ds, args.batch_size).dataloader_vis()
        pred = model.predict(train_dataset)
        
        for feature in pred:
            Feature_Target.append(feature)
        for value, pseual_label,confidence in train_ds:
            Label_Target.append(value[1])


    # if with_proto:
    #     proto_t=calculate_prototype(np.array(Feature_Target),np.array(Label_Target))
    #     for i in proto_t:
    #         Feature_Target.append(np.squeeze(i))
    #     Label_Target.extend([10,11,12,13,14,15,16])
    
    proto_s=np.array(proto_s)
    logits=cosine_similarity(np.array(Feature_Target),np.squeeze(proto_s))
    Label_Target=np.argmax(logits,axis=-1)

    Feature=Feature_Target
    Label=Label_Target

    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=50)
    embedding = tsne.fit_transform(Feature)
    # Draw Visualization of Feature
    if with_proto:
        colors = {0: 'tomato', 1: 'cornflowerblue', 2: 'cyan', 3: 'lime', 4: 'orange', 5: 'purple', 6: 'lightgrey',
        10: 'purple', 11: 'yellow', 12: 'black', 13: 'black', 14: 'white', 15: 'white', 16: 'red'}
                #10: 'black', 11: 'black', 12: 'black', 13: 'black',14: 'black', 15: 'black', 16: 'black'}
        labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral',
        10: 'Angry_proto',11: 'Disgust_proto', 12: 'Fear_proto', 13: 'Happy_proto', 14: 'Sad_proto',
            15: 'Surprised_proto', 16: 'Neutral_proto'}
    else:
        colors = {0: 'tomato', 1: 'cornflowerblue', 2: 'lightblue', 3: 'green', 4: 'orange', 5: 'purple', 6: 'lightgrey'}
        labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprised', 6: 'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    
    for i in range(7):
        data_x, data_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        scatter = plt.scatter(data_x, data_y,  edgecolors=colors[i], s=5, label=labels[i], marker='^', alpha=0.6)
    if with_proto:
        for i in range(10,17):
            data_x, data_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
            scatter = plt.scatter(data_x, data_y,  edgecolors=colors[i], s=30, label=labels[i], marker='*')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(7)],
               loc='upper left', prop={'size': 8}, bbox_to_anchor=(1.05, 0.85), borderaxespad=0)
    plt.savefig(fname='{}_{}_pre.png'.format(source_dataset,target_dataset,), bbox_inches='tight')


def calculate_prototype(embeddings,y):
    label_0_index = tf.where(y == 0)
    anchors_0 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings ,label_0_index),axis=0),
                            tf.cast(tf.size(label_0_index),tf.float32)).numpy()
    label_1_index = tf.where(y == 1)
    anchors_1 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_1_index),axis=0),
                            tf.cast(tf.size(label_1_index), tf.float32)).numpy()
    label_2_index = tf.where(y == 2)
    anchors_2 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_2_index),axis=0),
                            tf.cast(tf.size(label_2_index), tf.float32)).numpy()
    label_3_index = tf.where(y == 3)
    anchors_3 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_3_index),axis=0),
                            tf.cast(tf.size(label_3_index), tf.float32)).numpy()
    label_4_index = tf.where(y == 4)
    anchors_4 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_4_index),axis=0),
                            tf.cast(tf.size(label_4_index), tf.float32)).numpy()
    label_5_index = tf.where(y == 5)
    anchors_5 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_5_index),axis=0),
                            tf.cast(tf.size(label_5_index), tf.float32)).numpy()
    label_6_index = tf.where(y == 6)
    anchors_6 = tf.math.divide_no_nan(tf.reduce_sum(tf.gather(embeddings, label_6_index),axis=0),
                            tf.cast(tf.size(label_6_index), tf.float32)).numpy()
    anchors=[anchors_0,anchors_1,anchors_2,anchors_3,anchors_4,anchors_5,anchors_6]

    for i in range(len(anchors)):
        anchors[i]=anchors[i]/np.sqrt(np.sum(anchors[i]*anchors[i]))
    
    return anchors


def part_target(net,args,ds):
    def set_classifier():
        inputs=tf.keras.Input((args.img_size,args.img_size,args.channel))
        x=net.encoder(inputs,training=False)
        x=net.classify_head(x,training=False)
        m=tf.keras.Model(inputs,x)
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        return m
    

    target_train_dataset_evaluate=DataLoader(args,'lincls','train',ds,args.batch_size).dataloader()
    classifier=set_classifier()
    logits=classifier.predict(target_train_dataset_evaluate)
    pseudo_label=np.argmax(logits,axis=-1)
    target_confidence=[logits[i][pseudo_label[i]] for i in range (len(pseudo_label))]

    pseudo_label=tf.data.Dataset.from_tensor_slices(pseudo_label)
    target_confidence=tf.data.Dataset.from_tensor_slices(target_confidence)
    target_train_ds=tf.data.Dataset.zip((ds,pseudo_label,target_confidence))
    #挑选置信度高于阈值的数据
    ds=target_train_ds.filter(lambda x,y,z:z>=args.threshold)
    print(len(list(ds.as_numpy_iterator())))
    return ds


def main():
    args = Config()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    output_path = os.path.join(args.output_path, "source_classifier")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    get_session(args)
    logger = get_logger("MyLogger_visualization", output_path)
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))
    start_time = create_stamp()
    logger.info(start_time)
    logger.info("------------------------------------------------------------------")
    args.encoder_snapshot=r'/home/njustguest/wangchao/cross_domains/version261/output/RAF_fer2013/RAF_fer2013_proto_6_1/backbone_100'
    args.proj_snapshot = r'/home/njustguest/wangchao/cross_domains/version261/output/RAF_fer2013/RAF_fer2013_proto_6_1/proj_100'
    args.classifier_snapshot=None#r'/home/njustguest/wangchao/cross_domains/output/{}_{}_encoder_add_norm/source_classifier'.format(self.source,self.target)
    model=CCD(args,logger)
    #net=tf.keras.Sequential([model.encoder,model.proj])
    #Visualization(model,'fer2013',args,True)
    VisualizationForTwoDomain(model,'RAF','oulu',args,True)
    #VisualizationForPreDomain(model,'RAF','fer2013',args,True)



if __name__ == '__main__':
    main()