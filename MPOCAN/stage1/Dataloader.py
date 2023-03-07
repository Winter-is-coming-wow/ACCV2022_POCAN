# -- coding: utf-8 --
from re import I
from tensorflow.python.client import device_lib
from tensorflow.python.ops.image_ops_impl import ssim
device_lib.list_local_devices()
import cv2 as cv
import os, sys
import tensorflow as tf
import numpy as np
import random
import functools
import matplotlib.pyplot as plt
from Augment import Augment
import csv
from collections import Counter
from itertools import islice
from PIL import Image
# from util.face_detection import detect
from sklearn.model_selection import StratifiedKFold
from mtcnn import MTCNN

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

detector = MTCNN()


class DataLoader:
    def __init__(self, args, task, mode, datalist, batch_size, num_workers=1, shuffle=True):
        self.args = args
        self.task = task
        self.mode = mode
        self.datalist = datalist
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.augset = Augment(self.args, self.mode)
        self.dataloader = self._dataloader
        self.dataloader_sup = self._dataloader_sup


    def augmentation(self, img, shape):
        if self.task == 'pretext':
            img_list = []
            for _ in range(2):  # query, key
                aug_img = tf.identity(img)
                aug_img = self.augset._augment_pretext(aug_img, shape)
                img_list.append(aug_img)
            images = tf.concat(img_list, -1)
            return images
        return self.augset._augment_lincls(img, shape)

    def augmentation_sup(self, img, shape):
        if self.task == 'pretext':
            img_list = []
            for _ in range(2):  # query, key
                aug_img = tf.identity(img)
                aug_img = self.augset._augment_pretext(aug_img, shape)
                img_list.append(aug_img)
            images = tf.concat(img_list, -1)
            return images
        else:
            return self.augset._augment_lincls(img, shape)

    def translabel(self,dataset,label):
            if dataset == 'fer2013':
                label = self.args.fer2013_uniform[label]
            elif dataset == 'RAF':
                label = self.args.RAF_uniform[label]
            elif dataset == 'fer2013plus':
                label = self.args.fer2013plus_uniform[label]
            elif dataset == 'AffectNet':
                label = self.args.AffectNet_uniform[label]
            elif dataset == 'ExpW':
                label = self.args.ExpW_uniform[label]
            elif dataset == 'ckplus':
                label = self.args.ckplus_uniform[label]
            elif dataset == 'JAFFE':
                label = self.args.JAFFE_uniform[label]
            elif dataset == 'oulu':
                label = self.args.oulu_uniform[label]
            else:
                raise("no such dataset,can not transform label")
            return label

    def dataset_parser(self, value, cluster_label,confidence=None): #for target dataset
        shape = (self.args.img_size, self.args.img_size, self.args.channel)
        if self.task!='pretext':
            
            img=value
            inputs = self.augmentation(img, shape)
            label=cluster_label
            label=self.translabel(self.args.target,label)
            return (inputs, label)
        else:
            img = value[0]
            label=cluster_label
            inputs = self.augmentation(img, shape)
            #label=tf.cast(label,tf.int32)
            return (inputs, label)

    def dataset_parser_sup(self, value, cluster_label=None): #for source dataset

        shape = (self.args.img_size, self.args.img_size, self.args.channel)
        
        if self.task!='pretext':
            img = value
            label=cluster_label
            inputs = self.augmentation_sup(img, shape)
            return (inputs, label)
        else:
            img = value
            label=cluster_label
            inputs = self.augmentation_sup(img, shape)
            return (inputs,label)

    def _dataloader(self):  # for target dataset
        dataset = self.datalist
        dataset = dataset.map(self.dataset_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.task == 'pretext':
            dataset=dataset.repeat()
            dataset=dataset.shuffle(512)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def _dataloader_sup(self):  # for source dataset
        dataset = self.datalist
        dataset = dataset.map(self.dataset_parser_sup, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.task == 'pretext':
            dataset=dataset.repeat()
            dataset=dataset.shuffle(512)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


def set_dataset(dataset, selected_emotion=None, stage=None, channel=None):
    if dataset == 'fer2013':
        ds = Fer2013dataset(selected_emotion=selected_emotion)
        return ds.load_data(stage)
    elif dataset == 'fer2013plus':
        ds = Fer2013Plus('Majority')
        return ds.load_data(stage)
    elif dataset == 'RAF':
        ds = RAFdataset()
        return ds.load_data(stage)
    elif dataset == 'JAFFE':
        ds = JAFFEdataset()
        return ds.load_data()
    elif dataset == 'ckplus':
        ds = CKplus()
        return ds.load_data()
    elif dataset == 'oulu':
        ds = oulu_CASIA()
        return ds.load_data()
    elif dataset == 'sfew':
        ds = SFEW()
        return ds.load_data()
    elif dataset == 'ExpW':
        ds = ExpW()
        return ds.load_data(channel)
    elif dataset == 'AffectNet':
        ds = AffectNet()
        return ds.load_data(stage, channel)
    else:
        raise Exception('NO SUCH DATASET!')


class RAFdataset2:
    def __init__(self):
        self.index2label = {1: 'Surprise',
                            2: 'Fear',
                            3: 'Disgust',
                            4: 'Happiness',
                            5: 'Sadness',
                            6: 'Anger',
                            7: 'Neutral'}

        self.image_path = r'G:/deeplearning/FER datasets/RAF/basic/Image/aligned/aligned'
        # self.train_tfrecord_path = r'G:\demo\cross_dataset\datasets\RAF_train_tfrecord'
        self.train_tfrecord_path = r'G:/superwang/RAF/RAF_train_tfrecord_gray'
        # self.test_tfrecord_path = r'G:\demo\cross_dataset\datasets\RAF_test_tfrecord'
        self.test_tfrecord_path = r'G:/superwang/RAF/RAF_test_tfrecord_gray'


        self.train_image_list = []
        self.test_image_list = []
        self.train_image_num = 12271
        self.test_image_num = 3068

        self.img_size = 112
        self.channel = 1

    def load_data(self, stage, channel=3):
        """
        stage :if equal 0 then return train dataset,elif equal 1 return test dataset
        """
        if not os.path.exists(self.train_tfrecord_path):
            self.stastics()
            with tf.io.TFRecordWriter(self.train_tfrecord_path) as writer:
                np.random.shuffle(self.train_image_list)
                for item in self.train_image_list:
                    name, label = item
                    image = open(os.path.join(self.image_path, name[:-4] + '_aligned.jpg'), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=channel)
            return feature_dict['image'], feature_dict['label']

        if stage == 0:  # pretext
            raw_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
            train_dataset = raw_dataset.map(_parse_example)
            return train_dataset

        if not os.path.exists(self.test_tfrecord_path):
            np.random.shuffle(self.test_image_list)
            with tf.io.TFRecordWriter(self.test_tfrecord_path) as writer:
                for item in self.test_image_list:
                    name, label = item
                    image = open(os.path.join(self.image_path, name[:-4] + '_aligned.jpg'), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        if stage == 1:  # lincls
            raw_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
            train_dataset = raw_dataset.map(_parse_example)
            raw_dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
            test_dataset = raw_dataset.map(_parse_example)
            return train_dataset, test_dataset
        else:
            raw_dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
            test_dataset = raw_dataset.map(_parse_example)
            return test_dataset

    def load_part(self, proportion):
        if int(proportion) == 1:
            return self.load_data(1)
        self.stastics()
        np.random.shuffle(self.train_image_list)

        subset = random.sample(self.train_image_list, int(len(self.train_image_list) * proportion))
        images = []
        labels = []
        for item in subset:
            name, label = item
            image_path = os.path.join(self.image_path, name[:-4] + '_aligned.jpg')
            image = cv.imread(image_path, 0)
            image = np.expand_dims(image, -1)
            images.append(image)
            labels.append(label)
        print("len of subset ", len(labels))
        train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=3)
            return feature_dict['image'], feature_dict['label']

        raw_dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
        val_dataset = raw_dataset.map(_parse_example)
        return train_dataset, val_dataset

    def stastics(self):
        label_path = r'G:\deeplearning\FER datasets\RAF\basic\EmoLabel\list_patition_label.txt'
        class_num = [0] * 7
        train_class_num = [0] * 7
        test_class_num = [0] * 7
        with open(label_path, 'r') as f:
            for line in f:
                if 'train' in line:
                    name, label = line.split()
                    label = int(label) - 1
                    self.train_image_list.append([name, label])
                    class_num[label] += 1
                    train_class_num[label] += 1
                elif 'test' in line:
                    name, label = line.split()
                    label = int(label) - 1
                    self.test_image_list.append([name, label])
                    class_num[label] += 1
                    test_class_num[label] += 1



class RAFdataset:
    def __init__(self):
        self.index2label = {1: 'Surprise',
                            2: 'Fear',
                            3: 'Disgust',
                            4: 'Happiness',
                            5: 'Sadness',
                            6: 'Anger',
                            7: 'Neutral'}     

        self.image_path = r'/home/njustguest/wangchao/datasets/RAF/aligned'
        self.tfrecord_path = r'/home/njustguest/wangchao/datasets/RAF/raf'

        self.train_image_list = []
        self.test_image_list = []
        self.train_image_num = 12271
        self.test_image_num = 3068

        self.img_size = 100
        self.channel = 1

    def to_tfrecord(self, images_list,filename):
        """
        stage :if equal 0 then return train dataset,elif equal 1 return test dataset
        """
        if not os.path.exists(filename):
            with tf.io.TFRecordWriter(os.path.join(self.tfrecord_path,filename)) as writer:
                np.random.shuffle(images_list)
                for item in images_list:
                    name, label = item
                    image = open(os.path.join(self.image_path, name[:-4] + '_aligned.jpg'), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

    def load_data(self,channel=3):

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=channel)
            return feature_dict['image'], feature_dict['label']

        tf_list=[]
        for tf_file in os.listdir(self.tfrecord_path):
            file_name=os.path.join(self.tfrecord_path,tf_file)
            raw_dataset = tf.data.TFRecordDataset(file_name)
            tf_list.append(raw_dataset.map(_parse_example))
        return tf_list



    def stastics(self):
        label_path = r'/home/njustguest/wangchao/datasets/RAF/list_patition_label.txt'
        class_num = [0] * 7
        class_images=[[] for i in range(7)]
        with open(label_path, 'r') as f:
            for line in f:
                name, label = line.split()
                label = int(label) - 1
                class_num[label] += 1
                class_images[label].append((name,label))
        print(class_num)

        for i in range(7):
            self.to_tfrecord(class_images[i],'{}_tfrecord'.format(i))


class JAFFEdataset:
    def __init__(self):
        self.label2index = {'surprise': 1, 'fear': 2, 'disgust': 3, 'happiness': 4, 'sadness': 5, 'anger': 6,
                            'neutral': 7}

        self.root_dir = r'G:\superwang\JAFFE\jaffe'

        self.write_dir = r'G:\superwang\JAFFE\jaffe_processed'

        self.images_num = 213

        self.img_size = 256

    def load_data(self):
        train_images, val_images = [], []
        train_labels, val_labels = [], []
        class_num = [0] * 7
        for dir in os.listdir(self.write_dir):
            for file in os.listdir(os.path.join(self.write_dir, dir)):
                image = cv.imread(os.path.join(os.path.join(self.write_dir, dir), file))
                label = self.label2index[dir] - 1
                image = cv.resize(image, (112, 112))
                # image = image / 255.
                train_images.append(image)
                train_labels.append(label)
                val_images.append(image)
                val_labels.append(label)
                class_num[label] += 1
        index = list(range(len(train_labels)))
        np.random.shuffle(index)
        train_images = np.asarray(train_images)[index]
        train_labels = np.asarray(train_labels)[index]
        index = list(range(len(val_labels)))
        np.random.shuffle(index)
        val_images = np.asarray(val_images)[index]
        val_labels = np.asarray(val_labels)[index]
        print(class_num)
        return (train_images, train_labels), (val_images, val_labels)
    def write_data(self):
        for i in self.label2index.keys():
            if not os.path.exists(os.path.join(self.write_dir, i)):
                os.mkdir(os.path.join(self.write_dir, i))
        for emotion in os.listdir(self.root_dir):
            for image_name in os.listdir(os.path.join(self.root_dir, emotion)):

                if '.jpg' not in image_name:
                    print(image_name)
                    continue
                image_path = os.path.join(self.root_dir, emotion, image_name)
                image = cv.imread(image_path)
                face_image = self.align(image)
                if face_image is None:
                    print("mtcnn detect no face in {}, dlib will be used".format(image_path))
                    face_image = self.image_cut(image)
                    # continue
                try:
                    ret = cv.imwrite(os.path.join(self.write_dir, emotion, "{}".format(image_name)),
                                     face_image)
                    if not ret:
                        print("{} write failed".format(image_path))
                except:
                    print("{} write failed".format(image_path))

        print("Done!")

    # 裁剪人脸部分
    def align(self, image, name=None):
        result = detector.detect_faces(image)
        if len(result) == 0:
            return None  # , {'left_eye': '', 'right_eye': '', 'nose': '', 'mouth_left': '', 'mouth_right': ''}
        bounding_box = result[0]['box']
        for i in range(len(bounding_box)):
            if bounding_box[i] < 0:
                bounding_box[i] = 0
        landmarks = result[0]['keypoints']
        crop = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
               bounding_box[0]: bounding_box[0] + bounding_box[2]]
        landmarks['left_eye'] = '{} {}'.format(landmarks['left_eye'][0], landmarks['left_eye'][1])
        landmarks['right_eye'] = '{} {}'.format(landmarks['right_eye'][0], landmarks['right_eye'][1])
        landmarks['nose'] = '{} {}'.format(landmarks['nose'][0], landmarks['nose'][1])
        landmarks['mouth_left'] = '{} {}'.format(landmarks['mouth_left'][0], landmarks['mouth_left'][1])
        landmarks['mouth_right'] = '{} {}'.format(landmarks['mouth_right'][0], landmarks['mouth_right'][1])
        return crop  # , landmarks

    def image_cut(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 人脸检测器
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        # cv2检测人脸中心区域
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5, 5)
        )

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                if w < 70 and h < 70:
                    continue
                # 裁剪人脸核心部分
                crop = image[y:y + h, x: x + w, :]
                # 缩小为256*256
                return crop

        # crop = detect(gray)

        return image


class Fer2013dataset:
    def __init__(self, selected_emotion=None):
        # self.train_images_path = r'G:\demo\python\practice\fer\data\train'
        # self.val_images_path = r'G:\demo\python\practice\fer\data\val'
        # self.test_images_path = r'G:\demo\python\practice\fer\data\test'

        self.train_images_path = r'/home/njustguest/wangchao/datasets/fer2013/aligned/train'
        self.val_images_path = r'/home/njustguest/wangchao/datasets/fer2013/aligned/val'
        self.test_images_path = r'/home/njustguest/wangchao/datasets/fer2013/aligned/test'

        # self.train_tfrecord_path = r'G:\demo\cross_dataset\datasets\fer2013_train_tfrecord'
        # self.val_tfrecord_path = r'G:\demo\cross_dataset\datasets\fer2013_val_tfrecord'
        # self.test_tfrecord_path = r'G:\demo\cross_dataset\datasets\fer2013_test_tfrecord'

        self.train_tfrecord_path = r'G:/superwang/fer2013/fer2013_train_aligned_tfrecord'
        self.val_tfrecord_path = r'G:/superwang/fer2013/fer2013_val_aligned_tfrecord'
        self.test_tfrecord_path = r'G:/superwang/fer2013/fer2013_test_aligned_tfrecord'

        self.index2label = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprised', 6: 'neutral'}
        self.label2index = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprised': 5, 'neutral': 6}

        if selected_emotion is not None:
            self.selected_emotion = selected_emotion
        else:
            self.selected_emotion = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        self.train_image_num = 28709
        self.val_image_num = 3589
        self.test_image_num = 3589

        self.img_size = 48
        self.channel = 1

    def load_data(self, stage):

        if not os.path.exists(self.train_tfrecord_path):
            images_path = []
            for p in self.selected_emotion.keys():
                all_images_path = os.listdir(os.path.join(self.train_images_path, str(p)))
                all_images_labels = zip([p] * len(all_images_path), all_images_path)
                images_path.extend(all_images_labels)
            np.random.shuffle(images_path)
            print('train:', len(images_path))
            with tf.io.TFRecordWriter(self.train_tfrecord_path) as writer:
                for item in images_path:
                    label, filename = item[0], item[1]
                    image = open(os.path.join(self.train_images_path, str(label), filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=3)
            return feature_dict['image'], feature_dict['label']

        if stage == 0:  # pretext
            raw_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
            train_dataset = raw_dataset.map(_parse_example)
            return train_dataset


        if not os.path.exists(self.val_tfrecord_path):
            images_path = []
            for p in self.selected_emotion.keys():
                all_images_path = os.listdir(os.path.join(self.val_images_path, str(p)))
                all_images_labels = zip([p] * len(all_images_path), all_images_path)
                images_path.extend(all_images_labels)
            print('val:', len(images_path))
            with tf.io.TFRecordWriter(self.val_tfrecord_path) as writer:
                for item in images_path:
                    label, filename = item[0], item[1]
                    image = open(os.path.join(self.val_images_path, str(label), filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        if stage == 1:  # lincls
            raw_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
            train_dataset = raw_dataset.map(_parse_example)
            raw_dataset = tf.data.TFRecordDataset(self.val_tfrecord_path)
            val_dataset = raw_dataset.map(_parse_example)
            return train_dataset, val_dataset


        if not os.path.exists(self.test_tfrecord_path):
            images_path = []
            for p in self.selected_emotion.keys():
                all_images_path = os.listdir(os.path.join(self.test_images_path, str(p)))
                all_images_labels = zip([p] * len(all_images_path), all_images_path)
                images_path.extend(all_images_labels)
            print('test:', len(images_path))
            with tf.io.TFRecordWriter(self.test_tfrecord_path) as writer:
                for item in images_path:
                    label, filename = item[0], item[1]
                    image = open(os.path.join(self.test_images_path, str(label), filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        raw_dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
        test_dataset = raw_dataset.map(_parse_example)

        return test_dataset

    def load_part(self, proportion):
        if int(proportion) == 1:
            return self.load_data(1)
        images_path = []
        for p in self.selected_emotion.keys():
            all_images_path = os.listdir(os.path.join(self.train_images_path, str(p)))
            all_images_labels = zip([p] * len(all_images_path), all_images_path)
            images_path.extend(all_images_labels)
        np.random.shuffle(images_path)

        subset = random.sample(images_path, int(len(images_path) * proportion))
        images = []
        labels = []
        for item in subset:
            label, filename = item[0], item[1]
            image_path = os.path.join(self.train_images_path, str(label), filename)
            image = cv.imread(image_path, 0)
            image = np.expand_dims(image, -1)
            images.append(image)
            labels.append(label)
        train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=1)
            return feature_dict['image'], feature_dict['label']

        raw_dataset = tf.data.TFRecordDataset(self.val_tfrecord_path)
        val_dataset = raw_dataset.map(_parse_example)
        return train_dataset, val_dataset


class Fer2013Plus:
    def __init__(self, mode=None):
        self.train_images_path = r'G:\deeplearning\FER datasets\fer2013plus\Majority\train'
        self.val_images_path = r'G:\deeplearning\FER datasets\fer2013plus\Majority\val'
        self.test_images_path = r'G:\deeplearning\FER datasets\fer2013plus\Majority\test'

        self.train_tfrecord_path = r'G:\demo\cross_dataset\datasets\fer2013plus_train_tfrecord'
        self.val_tfrecord_path = r'G:\demo\cross_dataset\datasets\fer2013plus_val_tfrecord'
        self.test_tfrecord_path = r'G:\demo\cross_dataset\datasets\fer2013plus_test_tfrecord'

        # self.train_tfrecord_path = r'/home/njustguest/wangchao/datasets/fer2013/train_tfrecord'
        # self.val_tfrecord_path = r'/home/njustguest/wangchao/datasets/fer2013/val_tfrecord'
        # self.test_tfrecord_path = r'/home/njustguest/wangchao/datasets/fer2013/test_tfrecord'

        self.index2label = {0: 'neutral', 1: 'happy', 2: 'surprised', 3: 'sad', 4: 'angry', 5: 'disgust', 6: 'fear',
                            7: 'contempt'}

        self.train_image_num = 24938
        self.val_image_num = 3186
        self.test_image_num = 3137

        self.img_size = 48
        self.channel = 1

        self.mode = mode  # ['Majority','Crossentropy','Probability','Multi_target']

    def load_data(self, stage):
        images_path = []
        for p in range(7):
            all_images_path = os.listdir(os.path.join(self.train_images_path, str(p)))
            all_images_labels = zip([p] * len(all_images_path), all_images_path)
            images_path.extend(all_images_labels)
        np.random.shuffle(images_path)
        print('train:', len(images_path))
        if not os.path.exists(self.train_tfrecord_path):
            with tf.io.TFRecordWriter(self.train_tfrecord_path) as writer:
                for item in images_path:
                    label, filename = item[0], item[1]
                    image = open(os.path.join(self.train_images_path, str(label), filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_png(feature_dict['image'], channels=1)
            return feature_dict['image'], feature_dict['label']

        if stage == 0:  # pretext
            raw_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
            train_dataset = raw_dataset.map(_parse_example)
            return train_dataset

        images_path = []
        for p in range(7):
            all_images_path = os.listdir(os.path.join(self.val_images_path, str(p)))
            all_images_labels = zip([p] * len(all_images_path), all_images_path)
            images_path.extend(all_images_labels)
        print('val:', len(images_path))
        if not os.path.exists(self.val_tfrecord_path):
            with tf.io.TFRecordWriter(self.val_tfrecord_path) as writer:
                for item in images_path:
                    label, filename = item[0], item[1]
                    image = open(os.path.join(self.val_images_path, str(label), filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        if stage == 1:  # lincls
            raw_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
            train_dataset = raw_dataset.map(_parse_example)
            raw_dataset = tf.data.TFRecordDataset(self.val_tfrecord_path)
            val_dataset = raw_dataset.map(_parse_example)
            return train_dataset, val_dataset

        images_path = []
        for p in range(7):
            all_images_path = os.listdir(os.path.join(self.test_images_path, str(p)))
            all_images_labels = zip([p] * len(all_images_path), all_images_path)
            images_path.extend(all_images_labels)
        print('test:', len(images_path))
        if not os.path.exists(self.test_tfrecord_path):
            with tf.io.TFRecordWriter(self.test_tfrecord_path) as writer:
                for item in images_path:
                    label, filename = item[0], item[1]
                    image = open(os.path.join(self.test_images_path, str(label), filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        raw_dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
        test_dataset = raw_dataset.map(_parse_example)

        return test_dataset

    def load_part(self, proportion):
        if int(proportion) == 1:
            return self.load_data(1)
        images_path = []
        for p in range(7):
            all_images_path = os.listdir(os.path.join(self.train_images_path, str(p)))
            all_images_labels = zip([p] * len(all_images_path), all_images_path)
            images_path.extend(all_images_labels)
        np.random.shuffle(images_path)

        subset = random.sample(images_path, int(len(images_path) * proportion))
        images = []
        labels = []
        for item in subset:
            label, filename = item[0], item[1]
            image_path = os.path.join(self.train_images_path, str(label), filename)
            image = cv.imread(image_path, 0)
            image = np.expand_dims(image, -1)
            images.append(image)
            labels.append(label)
        train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=1)
            return feature_dict['image'], feature_dict['label']

        raw_dataset = tf.data.TFRecordDataset(self.val_tfrecord_path)
        val_dataset = raw_dataset.map(_parse_example)
        return train_dataset, val_dataset

    def str_to_image(self, image_blob):
        ''' Convert a string blob to an image object. '''
        image_string = image_blob.split(' ')
        image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
        return Image.fromarray(image_data)

    def write_images(self):
        ferplus_path = r'G:\deeplearning\FER datasets\fer2013plus\FERPlus-master\FERPlus-master\fer2013new.csv'
        fer_path = r'G:\deeplearning\FER datasets\FER2013\fer2013\fer2013\fer2013.csv'

        print("Start generating ferplus images.")

        for path in [self.train_images_path, self.val_images_path, self.test_images_path]:
            for i in range(10):
                if not os.path.exists(os.path.join(path, str(i))):
                    os.mkdir(os.path.join(path, str(i)))

        ferplus_entries = []
        with open(ferplus_path, 'r') as csvfile:
            ferplus_rows = csv.reader(csvfile, delimiter=',')
            for row in islice(ferplus_rows, 1, None):
                ferplus_entries.append(row)

        index = 0
        with open(fer_path, 'r') as csvfile:
            fer_rows = csv.reader(csvfile, delimiter=',')
            for row in islice(fer_rows, 1, None):
                ferplus_row = ferplus_entries[index]
                file_name = ferplus_row[1].strip()
                if len(file_name) > 0:
                    image = self.str_to_image(row[1])
                    label = self.get_label([float(j) for j in ferplus_row[2:]], self.mode)
                    if row[2] == 'Training':
                        image_path = os.path.join(self.train_images_path, str(label), file_name)
                    elif row[2] == 'PublicTest':
                        image_path = os.path.join(self.val_images_path, str(label), file_name)
                    elif row[2] == 'PrivateTest':
                        image_path = os.path.join(self.test_images_path, str(label), file_name)
                    image.save(image_path, compress_level=0)
                else:
                    file_name = str(index) + '.png'
                    image = self.str_to_image(row[1])
                    label = 9  # No face
                    if row[2] == 'Training':
                        image_path = os.path.join(self.train_images_path, str(label), file_name)
                    elif row[2] == 'PublicTest':
                        image_path = os.path.join(self.val_images_path, str(label), file_name)
                    elif row[2] == 'PrivateTest':
                        image_path = os.path.join(self.test_images_path, str(label), file_name)
                    image.save(image_path, compress_level=0)
                index += 1

        print("Done...")

    def get_label(self, emotion_raw, mode):
        '''
        Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:

        Majority: return the emotion that has the majority vote, or unknown if the count is too little.
        Probability or Crossentropty: convert the count into probability distribution.abs
        Multi-target: treat all emotion with 30% or more votes as equal.
        '''
        size = len(emotion_raw)
        # remove emotions with a single vote (outlier removal)
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)

        if mode == 'Majority':
            # find the peak value of the emo_raw list
            maxval = max(emotion_raw)
            if maxval > 0.5 * sum_list:
                return np.argmax(emotion_raw)
            else:
                return 8  # force setting as unknown
        # elif (mode == 'probability') or (mode == 'crossentropy'):
        #     sum_part = 0
        #     count = 0
        #     valid_emotion = True
        #     while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
        #         maxval = max(emotion_raw)
        #         for i in range(size):
        #             if emotion_raw[i] == maxval:
        #                 emotion[i] = maxval
        #                 emotion_raw[i] = 0
        #                 sum_part += emotion[i]
        #                 count += 1
        #                 if i >= 8:  # unknown or non-face share same number of max votes
        #                     valid_emotion = False
        #                     if sum(emotion) > maxval:  # there have been other emotions ahead of unknown or non-face
        #                         emotion[i] = 0
        #                         count -= 1
        #                     break
        #     if sum(
        #             emotion) <= 0.5 * sum_list or count > 3:  # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
        #         emotion = emotion_unknown  # force setting as unknown
        # elif mode == 'multi_target':
        #     threshold = 0.3
        #     for i in range(size):
        #         if emotion_raw[i] >= threshold * sum_list:
        #             emotion[i] = emotion_raw[i]
        #     if sum(emotion) <= 0.5 * sum_list:  # less than 50% of the votes are integrated, we discard this example
        #         emotion = emotion_unknown  # set as unknown
        #
        # return [float(i) / sum(emotion) for i in emotion]


class CKplus:
    def __init__(self):
        self.root_dir = r'/home/njustguest/wangchao/datasets/ckplus/processed'
        # self.write_dir = r'/home/njustguest/wangchao/datasets/ckplus/aligned'
        self.write_dir = r'G:\superwang\ck+\processed'
        self.label_path = r'G:\superwang\ck+\label_ck.txt'
        self.train_dir = r'G:\superwang\ck+\train'
        self.test_dir = r'G:\superwang\ck+\test'
        self.index2label = {0: 'neutral', 1: 'angry', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sad', 6: 'surprised'}
        self.clsss_num = {'0': 309, '6': 249, '4': 207, '2': 177, '1': 135, '5': 84, '3': 75}
        # self.clsss_num = {'0': 309, '6': 249, '4': 207, '2': 177, '1': 135, '5': 84, '3': 75}

        self.train_image_num = 1236

        self.img_size = [256, 256]
        self.channel = 1

    def load_data(self):
        train_images,test_images = [],[]
        train_labels,test_labels = [],[]
        class_num = [0] * 7
        for dir in os.listdir(self.train_dir):
            for file in os.listdir(os.path.join(self.train_dir, dir)):
                image = cv.imread(os.path.join(self.train_dir, dir, file))
                image = cv.resize(image, (112, 112))
                # image=image.astype(np.float32)
                label = int(dir)
                train_images.append(image)
                train_labels.append(label)
                class_num[label] += 1
        print(class_num)
        index = list(range(len(train_labels)))
        np.random.shuffle(index)
        train=( np.asarray(train_images)[index], np.asarray(train_labels)[index])

        class_num = [0] * 7
        for dir in os.listdir(self.test_dir):
            for file in os.listdir(os.path.join(self.test_dir, dir)):
                image = cv.imread(os.path.join(self.test_dir, dir, file))
                image = cv.resize(image, (112, 112))
                # image=image.astype(np.float32)
                label = int(dir)
                test_images.append(image)
                test_labels.append(label)
                class_num[label] += 1
        print(class_num)
        index = list(range(len(test_labels)))
        np.random.shuffle(index)
        test = (np.asarray(test_images)[index], np.asarray(test_labels)[index])

        return train,test

    def write_data(self):
        for i in range(7):
            if not os.path.exists(os.path.join(self.write_dir, str(i))):
                os.mkdir(os.path.join(self.write_dir, str(i)))
        choice = 0
        with open(self.label_path, 'r') as f:
            for line in f:
                if choice == 1 or choice == 2:
                    choice += 1
                    continue
                elif choice == 3:
                    choice = 0
                else:
                    choice += 1
                split = line.split()
                label = split[1]
                name = split[0]
                image_path = os.path.join(self.root_dir, name[6:10], name[11:14], name[6:])
                image = cv.imread(image_path, 0)
                face_image = self.image_cut(image)
                cv.imwrite(os.path.join(self.write_dir, label, name[6:]), face_image)

    # 裁剪人脸部分
    def image_cut(self, gray):
        # 人脸检测器
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        # cv2检测人脸中心区域
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5, 5)
        )

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                if w < 100 and h < 100:
                    continue
                # 裁剪人脸核心部分
                crop = gray[y:y + h, x: x + w]
                # 缩小为256*256
                crop = cv.resize(crop, (256, 256))
                return crop
        return None

    def split_train_test(self):

        for i in range(7):
            if not os.path.exists(os.path.join(self.train_dir,str(i))):
                os.mkdir(os.path.join(self.train_dir,str(i)))
            if not os.path.exists(os.path.join(self.test_dir,str(i))):
                os.mkdir(os.path.join(self.test_dir,str(i)))
        for dir in os.listdir(self.write_dir):
            emotion_list=os.listdir(os.path.join(self.write_dir, dir))
            emotion_len=len(emotion_list)
            emotion_list=np.asarray(emotion_list)
            np.random.shuffle(emotion_list)
            for i in range(emotion_len):
                image=cv.imread(os.path.join(self.write_dir, dir, emotion_list[i]))
                if i<=emotion_len*0.9:
                    cv.imwrite(os.path.join(self.train_dir, dir, emotion_list[i]),image)
                elif i>=emotion_len*0.9:
                    cv.imwrite(os.path.join(self.test_dir, dir, emotion_list[i]),image)


class SFEW:
    def __init__(self, mode=None):
        self.raw_train = r'G:\deeplearning\FER datasets\SFEW\processed\train'
        self.raw_val = r'G:\deeplearning\FER datasets\SFEW\processed\val'

        self.train_images_path = r'G:\superwang\SFEW2.0\train_aligned'
        self.val_images_path = r'G:\superwang\SFEW2.0\val_aligned'

        self.train_annotation = r'G:\deeplearning\FER datasets\SFEW\processed\train\Annotations\Bboxs'
        self.val_annotation = r'G:\deeplearning\FER datasets\SFEW\processed\val\Annotations\Bboxs'

        self.train_tfrecord_path = r'G:\superwang\SFEW2.0\sfew_train_tfrecord'
        self.val_tfrecord_path = r'G:\superwang\SFEW2.0\sfew_val_tfrecord'

        # self.train_tfrecord_path = r'/home/njustguest/wangchao/datasets/fer2013/train_tfrecord'
        # self.val_tfrecord_path = r'/home/njustguest/wangchao/datasets/fer2013/val_tfrecord'
        # self.test_tfrecord_path = r'/home/njustguest/wangchao/datasets/fer2013/test_tfrecord'

        self.index2label = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        self.label2index = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}

        self.train_image_num = 958
        self.val_image_num = 436

        self.img_size = 112
        self.channel = 3

        self.mode = mode  # ['Majority','Crossentropy','Probability','Multi_target']

    def load_data(self, channels=3):

        if not os.path.exists(self.train_tfrecord_path):
            images_path = []
            for p in os.listdir(self.train_images_path):
                all_images_names = os.listdir(os.path.join(self.train_images_path, p))
                all_images_path = [os.path.join(p, name) for name in all_images_names]
                all_images_labels = zip([self.label2index[p]] * len(all_images_path), all_images_path)
                images_path.extend(all_images_labels)
            np.random.shuffle(images_path)
            print('train:', len(images_path))
            with tf.io.TFRecordWriter(self.train_tfrecord_path) as writer:
                for item in images_path:
                    label, filename = item[0], item[1]
                    print(filename)
                    image = open(os.path.join(self.train_images_path, filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_png(feature_dict['image'], channels=channels)
            return feature_dict['image'], feature_dict['label']


        if not os.path.exists(self.val_tfrecord_path):
            images_path = []
            for p in os.listdir(self.val_images_path):
                all_images_names = os.listdir(os.path.join(self.val_images_path, p))
                all_images_path = [os.path.join(p, name) for name in all_images_names]
                all_images_labels = zip([self.label2index[p]] * len(all_images_path), all_images_path)
                images_path.extend(all_images_labels)
            print('val:', len(images_path))
            with tf.io.TFRecordWriter(self.val_tfrecord_path) as writer:
                for item in images_path:
                    label, filename = item[0], item[1]
                    image = open(os.path.join(self.val_images_path, filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        raw_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
        train_dataset = raw_dataset.map(_parse_example)
        raw_dataset = tf.data.TFRecordDataset(self.val_tfrecord_path)
        val_dataset = raw_dataset.map(_parse_example)

        total_dataset=train_dataset.concatenate(val_dataset)
        return total_dataset,total_dataset

    def write_data(self):
        def write(raw_path, boxes_path, write_path):
            for i in self.label2index.keys():
                if not os.path.exists(os.path.join(write_path, i)):
                    os.mkdir(os.path.join(write_path, i))

            for emotion in os.listdir(boxes_path):
                for txt_name in os.listdir(os.path.join(boxes_path, emotion)):
                    txt_path = os.path.join(boxes_path, emotion, txt_name)
                    x1, y1, x2, y2 = [int(i) if i > 0 else 0 for i in np.loadtxt(txt_path)]

                    if os.path.exists(os.path.join(raw_path, emotion, txt_name[:-3] + 'png')):
                        image = cv.imread(os.path.join(raw_path, emotion, txt_name[:-3] + 'png'))
                    elif os.path.exists(os.path.join(raw_path, emotion, txt_name[:-3] + 'jpg')):
                        image = cv.imread(os.path.join(raw_path, emotion, txt_name[:-3] + 'jpg'))
                    else:
                        print(os.path.join(raw_path, emotion, txt_name[:-3] + '*') + ' no exist')
                        continue

                    if image is None:
                        print(txt_name)
                    face = image[y1:y2, x1:x2, :]
                    face = cv.resize(face, (100, 100))
                    face = cv.equalizeHist(face)
                    # image=cv.rectangle(image,(x1,y1),(x2,y2),(255,0,0), 2)
                    # cv.imshow('a', face)
                    # cv.waitKey(0)

                    ret = cv.imwrite(os.path.join(write_path, emotion, txt_name[:-3] + 'png'), face)
                    if not ret:
                        print("{} write failed".format(os.path.join(write_path, emotion, txt_name[:-3] + 'png')))
            print("Done!")

        write(self.raw_train, self.train_annotation, self.train_images_path)
        write(self.raw_val, self.val_annotation, self.val_images_path)

    def write_align_data(self):
        write_dir = r'G:\superwang\SFEW2.0\aligned\train'
        train_dir = r'G:\superwang\SFEW2.0\data\train'
        val_dir = r'G:\superwang\SFEW2.0\data\val'
        for emotion in self.label2index.keys():
            if not os.path.exists(os.path.join(write_dir, emotion)):
                os.mkdir(os.path.join(write_dir, emotion))
                # align train data
        # for emotion in self.label2index.keys():
        #     for image_name in os.listdir(os.path.join(train_dir,emotion,emotion)):
        #         if not ('.jpg' in image_name or '.png' in image_name):
        #             print(image_name)
        #             continue
        #
        #         image_path = os.path.join(train_dir, emotion,emotion, image_name)
        #         image = cv.imread(image_path)
        #         face_image = self.align(image)
        #         if face_image is None:
        #             print("mtcnn detect no face in {}, dlib will be used".format(image_path))
        #             face_image = self.image_cut(image,image_path)
        #             # continue
        #         try:
        #             ret = cv.imwrite(os.path.join(write_dir, emotion, "{}".format(image_name)),
        #                              face_image)
        #             if not ret:
        #                 print("{} write failed".format(image_path))
        #         except:
        #             print("{} write failed".format(image_path))
        # print("Done!")

        write_dir = r'G:\superwang\SFEW2.0\aligned\val'
        for emotion in self.label2index.keys():
            for image_name in os.listdir(os.path.join(val_dir, emotion, emotion)):
                if not ('.jpg' in image_name or '.png' in image_name):
                    print(image_name)
                    continue

                image_path = os.path.join(val_dir, emotion, emotion, image_name)
                image = cv.imread(image_path)
                face_image = self.align(image)
                if face_image is None:
                    print("mtcnn detect no face in {}, dlib will be used".format(image_path))
                    face_image = self.image_cut(image, image_path)
                    # continue
                try:
                    ret = cv.imwrite(os.path.join(write_dir, emotion, "{}".format(image_name)),
                                     face_image)
                    if not ret:
                        print("{} write failed".format(image_path))
                except:
                    print("{} write failed".format(image_path))
        print("Done!")

    def align(self, image, name=None):
        result = detector.detect_faces(image)
        if len(result) == 0:
            return None  # , {'left_eye': '', 'right_eye': '', 'nose': '', 'mouth_left': '', 'mouth_right': ''}
        bounding_box = result[0]['box']
        for i in range(len(bounding_box)):
            if bounding_box[i] < 0:
                bounding_box[i] = 0
        landmarks = result[0]['keypoints']
        crop = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
               bounding_box[0]: bounding_box[0] + bounding_box[2]]
        landmarks['left_eye'] = '{} {}'.format(landmarks['left_eye'][0], landmarks['left_eye'][1])
        landmarks['right_eye'] = '{} {}'.format(landmarks['right_eye'][0], landmarks['right_eye'][1])
        landmarks['nose'] = '{} {}'.format(landmarks['nose'][0], landmarks['nose'][1])
        landmarks['mouth_left'] = '{} {}'.format(landmarks['mouth_left'][0], landmarks['mouth_left'][1])
        landmarks['mouth_right'] = '{} {}'.format(landmarks['mouth_right'][0], landmarks['mouth_right'][1])
        return crop  # , landmarks

    def image_cut(self, image, name=None):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 人脸检测器
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        # cv2检测人脸中心区域
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5, 5)
        )

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                if w < 70 and h < 70:
                    continue
                # 裁剪人脸核心部分
                crop = image[y:y + h, x: x + w, :]
                # 缩小为256*256
                return crop

        # crop = detect(gray)
        print('\tdlib detect no face in {} , origin image will be return'.format(name))

        return image


class oulu_CASIA:
    def __init__(self):
        self.root_dir = r'G:\superwang\oulu\Strong'
        self.write_dir = r'G:\superwang\oulu\Strong_aligned'
        self.train_dir = r'G:\superwang\oulu\train_rgb'
        self.test_dir = r'G:\superwang\oulu\test_rgb'
        self.index2label = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sadness', 5: 'Surprise',
                            6: 'Neutral'}
        self.label2index = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Sadness': 4, 'Surprise': 5,
                            'Neutral': 6}
        self.clsss_num = [237, 236, 239, 239, 239, 236, 480]
        # self.clsss_num = {'0': 309, '6': 249, '4': 207, '2': 177, '1': 135, '5': 84, '3': 75}

        self.train_image_num = 1920

        self.img_size = [256, 256]
        self.channel = 3

    def load_data(self):
        # train_images, test_images = [], []
        # train_labels, test_labels = [], []
        # class_num = [0] * 7
        # for dir in os.listdir(self.train_dir):
        #     for file in os.listdir(os.path.join(self.train_dir, dir)):
        #         image = cv.imread(os.path.join(self.train_dir, dir, file))
        #         image = cv.resize(image, (112, 112))
        #         # image=image.astype(np.float32)
        #         label = int(self.label2index[dir])
        #         train_images.append(image)
        #         train_labels.append(label)
        #         class_num[label] += 1
        # print(class_num)
        # index = list(range(len(train_labels)))
        # np.random.shuffle(index)
        # train = (np.asarray(train_images)[index], np.asarray(train_labels)[index])
        #
        # class_num = [0] * 7
        # for dir in os.listdir(self.test_dir):
        #     for file in os.listdir(os.path.join(self.test_dir, dir)):
        #         image = cv.imread(os.path.join(self.test_dir, dir, file))
        #         image = cv.resize(image, (112, 112))
        #         # image=image.astype(np.float32)
        #         label = int(self.label2index[dir])
        #         test_images.append(image)
        #         test_labels.append(label)
        #         class_num[label] += 1
        # print(class_num)
        # index = list(range(len(test_labels)))
        # np.random.shuffle(index)
        # test = (np.asarray(test_images)[index], np.asarray(test_labels)[index])
        #
        # return train, test
        images,labels = [], []
        class_num = [0] * 7
        for dir in os.listdir(self.train_dir):
            for file in os.listdir(os.path.join(self.train_dir, dir)):
                image = cv.imread(os.path.join(self.train_dir, dir, file))
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = cv.resize(image, (112, 112))
                # image=image.astype(np.float32)
                label = int(self.label2index[dir])
                images.append(image)
                labels.append(label)
                class_num[label] += 1
        class_num = [0] * 7
        for dir in os.listdir(self.test_dir):
            for file in os.listdir(os.path.join(self.test_dir, dir)):
                image = cv.imread(os.path.join(self.test_dir, dir, file))
                image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
                image = cv.resize(image, (112, 112))
                # image=image.astype(np.float32)
                label = int(self.label2index[dir])
                images.append(image)
                labels.append(label)
                class_num[label] += 1
        print(class_num)
        index = list(range(len(labels)))
        np.random.shuffle(index)
        dataset = (np.asarray(images)[index], np.asarray(labels)[index])

        return dataset,dataset


    def write_data(self):
        for i in self.label2index.keys():
            if not os.path.exists(os.path.join(self.write_dir, i)):
                os.mkdir(os.path.join(self.write_dir, i))
        for subject in os.listdir(self.root_dir):
            for emotion in os.listdir(os.path.join(self.root_dir, subject)):
                names = sorted(os.listdir(os.path.join(self.root_dir, subject, emotion)))
                choosed = [names[0], names[-3], names[-2], names[-1]]
                for i, image_name in enumerate(choosed):
                    if '.jpeg' not in image_name:
                        print(image_name)
                        continue
                    image_path = os.path.join(self.root_dir, subject, emotion, image_name)
                    image = cv.imread(image_path)
                    face_image = self.align(image)
                    if face_image is None:
                        print("mtcnn detect no face in {}, dlib will be used".format(image_path))
                        face_image = self.image_cut(image)
                        # continue
                    if i == 0:
                        try:
                            ret = cv.imwrite(
                                os.path.join(self.write_dir, 'Neutral',
                                             "{}_{}_{}".format(subject, emotion, image_name)), face_image)
                        except:
                            # face_image=self.image_cut(image)
                            # ret = cv.imwrite(
                            #     os.path.join(self.write_dir, 'Neutral', "{}_{}_{}".format(subject, emotion, image_name)),face_image)
                            # ret=cv.imwrite(os.path.join(self.write_dir,image_name),face_image)
                            print("{} write failed".format(image_path))
                        if not ret:
                            print("{} write failed".format(image_path))
                    else:
                        try:
                            ret = cv.imwrite(os.path.join(self.write_dir, emotion, "{}_{}".format(subject, image_name)),
                                             face_image)
                        except:
                            # face_image=self.image_cut(image)
                            # ret = cv.imwrite(os.path.join(self.write_dir, emotion, "{}_{}".format(subject, image_name)),
                            #                 face_image)
                            # ret=cv.imwrite(os.path.join(self.write_dir,image_name),face_image)
                            print("{} write failed".format(image_path))
                        if not ret:
                            print("{} write failed".format(image_path))
        print("Done!")

    # 裁剪人脸部分
    def align(self, image, name=None):
        result = detector.detect_faces(image)
        if len(result) == 0:
            return None  # , {'left_eye': '', 'right_eye': '', 'nose': '', 'mouth_left': '', 'mouth_right': ''}
        bounding_box = result[0]['box']
        for i in range(len(bounding_box)):
            if bounding_box[i] < 0:
                bounding_box[i] = 0
        landmarks = result[0]['keypoints']
        crop = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
               bounding_box[0]: bounding_box[0] + bounding_box[2]]
        landmarks['left_eye'] = '{} {}'.format(landmarks['left_eye'][0], landmarks['left_eye'][1])
        landmarks['right_eye'] = '{} {}'.format(landmarks['right_eye'][0], landmarks['right_eye'][1])
        landmarks['nose'] = '{} {}'.format(landmarks['nose'][0], landmarks['nose'][1])
        landmarks['mouth_left'] = '{} {}'.format(landmarks['mouth_left'][0], landmarks['mouth_left'][1])
        landmarks['mouth_right'] = '{} {}'.format(landmarks['mouth_right'][0], landmarks['mouth_right'][1])
        return crop  # , landmarks

    def image_cut(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 人脸检测器
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        # cv2检测人脸中心区域
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5, 5)
        )

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                if w < 70 and h < 70:
                    continue
                # 裁剪人脸核心部分
                crop = image[y:y + h, x: x + w, :]
                # 缩小为256*256
                return crop

        # crop = detect(gray)

        return image


class ExpW:
    def __init__(self):
        self.raw_data = r'G:\superwang\ExpW\data\image\origin\origin'
        self.write_path = r'G:\superwang\ExpW\data\image\origin\processed'
        self.label_path = r'G:\superwang\ExpW\data\label\label.lst'
        self.write_label_path= r'G:\superwang\ExpW\data\label\label_processed.txt'

        self.train_tfrecord_path = r'G:\superwang\ExpW\ExpW_train_tfrecord'
        self.val_tfrecord_path = r'G:\superwang\ExpW\ExpW_val_tfrecord'
        self.test_tfrecord_path = r'G:\superwang\ExpW\ExpW_test_tfrecord'

        self.index2label = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
        self.label2index = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

        self.train_image_num = 91793  # [3671, 3995, 1088, 30537, 10559, 7060, 34883]
        self.train_num=28848
        self.val_num=28848
        self.test_num=34097

    def load_data(self, channel=3, mode=0):
        images_path = []
        Label=[]
        with open(self.write_label_path) as f:
            for line in f:
                line = line.split()
                label = int(line[-1])
                image_name = line[0]
                Label.append(label)
                images_path.append((label, image_name))
        np.random.shuffle(images_path)
        train_val_files,test_files=train_test_split(images_path,train_size=28848*2,test_size=34097,random_state=0,stratify=Label)
        Label=[]
        for l,i in train_val_files:
            Label.append(l)
        train_files,val_files=train_test_split(train_val_files,train_size=28848,test_size=28848,random_state=0,stratify=Label)


        if not os.path.exists(self.train_tfrecord_path):
            with tf.io.TFRecordWriter(self.train_tfrecord_path) as writer:
                for item in train_files:
                    label, filename = int(item[0]), item[1]
                    if not os.path.exists(os.path.join(self.write_path, filename)):
                        continue
                    image = open(os.path.join(self.write_path, filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
        if not os.path.exists(self.val_tfrecord_path):
            with tf.io.TFRecordWriter(self.val_tfrecord_path) as writer:
                for item in val_files:
                    label, filename = int(item[0]), item[1]
                    if not os.path.exists(os.path.join(self.write_path, filename)):
                        continue
                    image = open(os.path.join(self.write_path, filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
        if not os.path.exists(self.test_tfrecord_path):
            with tf.io.TFRecordWriter(self.test_tfrecord_path) as writer:
                for item in test_files:
                    label, filename = int(item[0]), item[1]
                    if not os.path.exists(os.path.join(self.write_path, filename)):
                        continue
                    image = open(os.path.join(self.write_path, filename), 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_png(feature_dict['image'], channels=channel)
            feature_dict['image'] = tf.reverse(feature_dict['image'], axis=[-1])
            return feature_dict['image'], feature_dict['label']

        if mode == 0:  # 返回3通道图片
            raw_dataset = tf.data.TFRecordDataset(self.train_tfrecord_path)
            train_dataset = raw_dataset.map(_parse_example)

            raw_dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
            test_dataset = raw_dataset.map(_parse_example)
            return train_dataset,test_dataset

        if mode==1:
            raw_dataset = tf.data.TFRecordDataset(self.test_tfrecord_path)
            test_dataset = raw_dataset.map(_parse_example)
            return test_dataset

    def write_data(self):
        label_txt = r'G:\superwang\ExpW\data\label\label_processed.txt'
        image_num = 0
        classes_num = [0] * 7
        c = 0
        with open(self.label_path) as f:
            for line in f:
                if image_num % 10000 == 0:
                    print(image_num)
                line = line.split()
                image_name = line[0]
                face_id = line[1]
                [top, left, right, bottom] = [int(i) for i in line[2:-2]]
                classes_num[int(line[-1])] += 1
                image_num += 1
                # if os.path.exists(os.path.join(self.write_path, image_name)):
                #     continue

                image = cv.imread(os.path.join(self.raw_data, image_name))

                if image_name[-4] == '.' and image_name[-3:] == 'jpg':
                    image_name = image_name[:-4] + '_' + face_id + image_name[-4:]
                else:
                    print(image_name)
                with open(label_txt, 'a+', newline='') as f:
                    f.write('{} {} {}\n'.format(image_name, line[-2], line[-1]))

                if image is None:
                    print('{} image is not exist'.format(self.raw_data, image_name))
                    continue
                face = image[top:bottom, left:right, :]

                if image is None:
                    print("未找到 {}".format(os.path.join(self.raw_data, image_name)))
                else:
                    face_image = self.align(face)
                    if face_image is None:
                        c += 1
                        # print("mtcnn detect no face in {}, dlib will be used".format(
                        # os.path.join(self.raw_data, image_name)))
                        face_image = self.image_cut(face)
                    face_image = cv.resize(face_image, (100, 100))
                    cv.imwrite(os.path.join(self.write_path, image_name), face_image)
        print(image_num, classes_num)
        print(c)
        print(len(list(os.listdir(self.write_path))))
        # 91793 [3671, 3995, 1088, 30537, 10559, 7060, 34883]
        # 25987
        print("Done!")

        # 裁剪人脸部分

    def align(self, image, name=None):
        result = detector.detect_faces(image)
        if len(result) == 0:
            return None  # , {'left_eye': '', 'right_eye': '', 'nose': '', 'mouth_left': '', 'mouth_right': ''}
        bounding_box = result[0]['box']
        for i in range(len(bounding_box)):
            if bounding_box[i] < 0:
                bounding_box[i] = 0
        # landmarks = result[0]['keypoints']
        crop = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
               bounding_box[0]: bounding_box[0] + bounding_box[2]]
        # landmarks['left_eye'] = '{} {}'.format(landmarks['left_eye'][0], landmarks['left_eye'][1])
        # landmarks['right_eye'] = '{} {}'.format(landmarks['right_eye'][0], landmarks['right_eye'][1])
        # landmarks['nose'] = '{} {}'.format(landmarks['nose'][0], landmarks['nose'][1])
        # landmarks['mouth_left'] = '{} {}'.format(landmarks['mouth_left'][0], landmarks['mouth_left'][1])
        # landmarks['mouth_right'] = '{} {}'.format(landmarks['mouth_right'][0], landmarks['mouth_right'][1])
        return crop  # , landmarks

    def image_cut(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 人脸检测器
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        # cv2检测人脸中心区域
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5, 5)
        )

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                if w < 70 and h < 70:
                    continue
                # 裁剪人脸核心部分
                crop = image[y:y + h, x: x + w, :]
                # 缩小为256*256
                return crop

        # crop = detect(gray)

        return image


class AffectNet:
    def __init__(self):
        self.train_csv = r'G:\deeplearning\FER datasets\AffectNet\AffectNet\Manually_Annotated_file_lists\training.csv'
        self.val_csv = r'G:\deeplearning\FER datasets\AffectNet\AffectNet\Manually_Annotated_file_lists\validation.csv'

        self.image_source = r'G:\deeplearning\FER datasets\AffectNet\AffectNet\Manually_Annotated\Manually_Annotated\Manually_Annotated_Images'

        self.write_path_train = r'G:\deeplearning\FER datasets\AffectNet\train'
        self.write_path_val = r'G:\deeplearning\FER datasets\AffectNet\val'

        self.tfrecord_dir = r'G:\superwang\AffectNet'

        self.index2label = {0: "neutral", 1: "happy", 2: "sad", 3: "surprise", 4: "fear", 5: "disgust", 6: "angry"}
        self.label2index = {'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3, 'fear': 4, 'disgust': 5, 'angry': 6}

        self.train_image_num = 283900#414798  # [74874, 134415, 25459, 14090, 6378, 3803, 24882, 3750, 33088, 11645, 82414]
        self.val_image_num = 3500  # [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]

    def load_data(self, mode=0, channel=3):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        # def _parse_example(example_string):
        #     feature_dict = tf.io.parse_single_example(example_string, feature_description)
        #     feature_dict['image'] = tf.io.decode_image(feature_dict['image'], channels=channel, expand_animations=False)
        #     if channel == 3:
        #         feature_dict['image'] = tf.reverse(feature_dict['image'], axis=[-1])
        #     return feature_dict['image'], feature_dict['label']
        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=channel)
            return feature_dict['image'], feature_dict['label']

        if mode == 0:
            raw_dataset = tf.data.TFRecordDataset(
                [os.path.join(self.tfrecord_dir, 'part{}_train_tfrecord'.format(part_id)) for part_id in
                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
            dataset = raw_dataset.map(_parse_example)
            return dataset

        else:
            raw_dataset = tf.data.TFRecordDataset(
                [os.path.join(self.tfrecord_dir, 'part{}_train_tfrecord'.format(part_id)) for part_id in
                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
            train_dataset = raw_dataset.map(_parse_example)
            raw_dataset = tf.data.TFRecordDataset(
                os.path.join(self.tfrecord_dir, 'val_tfrecord'))
            val_dataset = raw_dataset.map(_parse_example)
            return train_dataset, val_dataset

    def convert_to_tfrecord(self):
        example_per_tfrecord = 30000
        with open(self.train_csv, 'r') as csvfile:
            fer_rows = csv.reader(csvfile, delimiter=',')
            image_id = 0
            for row in islice(fer_rows, 1, None):
                # subdir, name = row[0].split('/')
                # if name.split('.')[1].lower() not in ['jpg','jpeg','png','bmp','tif']:
                #     print(subdir,name)
                if image_id % example_per_tfrecord == 0:
                    if image_id != 0:
                        writer.close()
                        print("part {} done".format(image_id // example_per_tfrecord))
                    part_id = image_id // example_per_tfrecord
                    writer = tf.io.TFRecordWriter(
                        os.path.join(self.tfrecord_dir, "part{}_train_tfrecord".format(part_id)))
                subdir, name = row[0].split('/')
                _, ex = name.split('.')
                name = _ + '.png'
                label = row[-3]
                if int(label) not in [0, 1, 2, 3, 4, 5, 6]:
                    continue
                image_path = os.path.join(self.write_path_train, label, name)
                if not os.path.exists(image_path):
                    continue
                image = open(image_path, 'rb').read()
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                image_id += 1

        if not os.path.exists(os.path.join(self.tfrecord_dir, 'val_tfrecord')):
            with tf.io.TFRecordWriter(os.path.join(self.tfrecord_dir, 'val_tfrecord')) as writer:
                with open(self.val_csv, 'r') as csvfile:
                    fer_rows = csv.reader(csvfile, delimiter=',')
                    for row in islice(fer_rows, 1, None):
                        subdir, name = row[0].split('/')
                        _, ex = name.split('.')
                        name = _ + '.png'
                        label = row[-3]
                        if int(label) not in [0, 1, 2, 3, 4, 5, 6]:
                            continue
                        image_path = os.path.join(self.write_path_val, label, name)
                        if not os.path.exists(image_path):
                            continue
                        image = open(image_path, 'rb').read()
                        feature = {
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)]))
                        }
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())

    def write_data(self):
        image_num = 0
        train_classes_num = [0] * 11
        val_classes_num = [0] * 11
        for path in [self.write_path_train, self.write_path_val]:
            for i in range(11):
                if not os.path.exists(os.path.join(path, str(i))):
                    os.mkdir(os.path.join(path, str(i)))
        index = 0
        for csv_file in [self.train_csv, self.val_csv]:
            with open(csv_file, 'r') as csvfile:
                fer_rows = csv.reader(csvfile, delimiter=',')
                for row in islice(fer_rows, 1, None):
                    subdir, name = row[0].split('/')
                    try:
                        [x1, y1, x2, y2] = [int(x) for x in row[1:5]]
                    except ValueError as e:
                        print('ValueError:', e)
                        print(row)
                    label = row[-3]
                    image_path = os.path.join(self.image_source, subdir, name)
                    image = cv.imread(image_path)
                    if image is None:
                        print("No such file {}".format(image_path))
                        continue
                    src_h, src_w, c = image.shape
                    w = x2 - x1
                    h = y2 - y1
                    x1 = max(0, int(x1 - 0.1 * w))
                    x2 = min(src_w, int(x2 + 0.1 * w))
                    y1 = max(0, int(y1 - 0.1 * h))
                    y2 = min(src_h, int(y2 + 0.1 * h))
                    face = image[y1:y2, x1:x2, :]
                    _, ex = name.split('.')
                    name = _ + '.png'
                    if index == 0:
                        cv.imwrite(os.path.join(self.write_path_train, label, name), face)
                        train_classes_num[int(label)] += 1
                    else:
                        cv.imwrite(os.path.join(self.write_path_val, label, name), face)
                        val_classes_num[int(label)] += 1
                    image_num += 1
            print(image_num)
            print(train_classes_num)
            print(val_classes_num)
            index += 1
        print("Done...")


class AffectNet_part:
    def __init__(self):
        self.train_csv = r'G:\deeplearning\FER datasets\AffectNet\AffectNet\Manually_Annotated_file_lists\training.csv'

        self.image_source = r'G:\deeplearning\FER datasets\AffectNet\train'

        self.write_path_even = r'G:\deeplearning\FER datasets\AffectNet\part\even'
        self.write_path_uneven = r'G:\deeplearning\FER datasets\AffectNet\part\uneven'

        self.tfrecord_path_even = r'G:\demo\cross_dataset\datasets\AffectNet\uniform_tfrecord_even'
        self.tfrecord_path_uneven = r'G:\demo\cross_dataset\datasets\AffectNet\uniform_tfrecord_uneven'

        self.index2label = {0: "neutral", 1: "happy", 2: "sad", 3: "surprise", 4: "fear", 5: "disgust", 6: "angry"}
        self.label2index = {'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3, 'fear': 4, 'disgust': 5, 'angry': 6}

        self.image_num_even = [2000, 2000, 2000, 2000, 2000, 2000, 2000]
        self.image_num_uneven = [500, 1000, 1500, 2000, 2500, 3000, 3500]

    def load_data(self, type=0, channels=1):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_png(feature_dict['image'], channels=channels)
            return feature_dict['image'], feature_dict['label']

        if type == 0:
            raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path_even)
            train_dataset = raw_dataset.map(_parse_example)
            return train_dataset
        else:
            raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path_uneven)
            train_dataset = raw_dataset.map(_parse_example)
            return train_dataset

    def convert_to_tfrecord(self):
        for write_path, tfrecord_path in [(self.write_path_even, self.tfrecord_path_even),
                                          (self.write_path_uneven, self.tfrecord_path_uneven)]:
            images_path = []
            for p in range(7):
                all_images_path = os.listdir(os.path.join(write_path, str(p)))
                all_images_labels = zip([p] * len(all_images_path), all_images_path)
                images_path.extend(all_images_labels)
            np.random.shuffle(images_path)
            print('train:', len(images_path))
            if not os.path.exists(tfrecord_path):
                with tf.io.TFRecordWriter(tfrecord_path) as writer:
                    for item in images_path:
                        label, filename = item[0], item[1]
                        image = open(os.path.join(write_path, str(label), filename), 'rb').read()
                        feature = {
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                        }
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())

    def write_data(self):
        for path in [self.write_path_even, self.write_path_uneven]:
            for i in range(7):
                if not os.path.exists(os.path.join(path, str(i))):
                    os.mkdir(os.path.join(path, str(i)))

        for i in range(7):
            src_list = os.listdir(os.path.join(self.image_source, str(i)))
            np.random.shuffle(src_list)

            even_list = random.sample(src_list, self.image_num_even[i])
            uneven_list = random.sample(src_list, self.image_num_uneven[i])

            for image_name in even_list:
                image = cv.imread(os.path.join(self.image_source, str(i), image_name))
                cv.imwrite(os.path.join(self.write_path_even, str(i), image_name), image)
            print("emotion {} for even done!".format(i))

            for image_name in uneven_list:
                image = cv.imread(os.path.join(self.image_source, str(i), image_name))
                cv.imwrite(os.path.join(self.write_path_uneven, str(i), image_name), image)
            print("emotion {} for uneven done!".format(i))


if __name__ == '__main__':
    dataset = ExpW()
    dataset.write_data()
