import tensorflow as tf
import cv2
import numpy as np
import socket
import struct
import time
import multiprocessing
import random

try:
    import cPickle as pickle
except:
    # Python 3
    import pickle

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

from utils import get_datasets

from io import BytesIO

#from tracker import network
from utils import bb_util
#from re3_utils.util import im_util
from utils import drawing
#from re3_utils.util import IOU
#from re3_utils.simulator import simulator
from utils import tf_util
import keras.backend as K

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

HOST = 'localhost'


PARALLEL_SIZE = 4
IMG_SIZE = 224

class Dataset(object):
    def __init__(self, num_unrolls, batch_size, memory_size, port, debug=False):

        self.num_unrolls = num_unrolls
        self.batch_size  = batch_size
        self.memory_size = memory_size
        self.port = port
        self.debug = debug
        self.image_paths = []
        self.key_lookup = dict()
        self.datasets = []
        #self.add_dataset('imagenet_video')
        self.add_dataset('kitti_tracking')

    def add_dataset(self, dataset_name):
        dataset_ind = len(self.datasets)
        dataset_gt = get_datasets.get_data_for_dataset(dataset_name, 'train')['gt']

        if self.debug:
            image_path = get_datasets.get_data_for_dataset(dataset_name, 'train')['image_paths']
            self.image_paths.append(image_path)

        for xx in range(dataset_gt.shape[0]):
            line = dataset_gt[xx,:].astype(int)
            # KEY = [data_idx, video_id, track_id, image_seq]
            self.key_lookup[(dataset_ind, line[4], line[5], line[6])] = xx
        self.datasets.append(dataset_gt)

    def getData(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, self.port))
        sock.sendall(('Ready' + '\n').encode('utf-8'))
        receivedBytes = sock.recv(4)
        messageLength = struct.unpack('>I', receivedBytes)[0]
        key = sock.recv(messageLength)
        key = pickle.loads(key)

        images = [None] * (self.num_unrolls + 1)
        for nn in range(self.num_unrolls + 1):
            image = BytesIO()
            # Connect to server and send data.

            # Get the array.
            received = 0
            receivedBytes = sock.recv(4)
            messageLength = struct.unpack('>I', receivedBytes)[0]
            while received < messageLength:
                receivedBytes = sock.recv(min(1024, messageLength - received))
                image.write(receivedBytes)
                received += len(receivedBytes)

            imageArray = np.fromstring(image.getvalue(), dtype=np.uint8)
            image.close()

            # Get shape.
            receivedBytes = sock.recv(4)
            messageLength = struct.unpack('>I', receivedBytes)[0]
            shape = sock.recv(messageLength)
            shape = pickle.loads(shape)
            imageArray = imageArray.reshape(shape)
            if len(imageArray.shape) < 3:
                imageArray = np.tile(imageArray[:,:,np.newaxis], (1,1,3))
            images[nn] = imageArray
        sock.close()
        return (key, images)

    def get_data_sequence(self):
        try:

            # Preallocate the space for the datas and labels.
            dataImage   = np.zeros((self.num_unrolls, self.memory_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            dataMotion  = np.zeros((self.num_unrolls, self.memory_size, 4), dtype=np.float32)

            labelImage  = np.zeros((self.num_unrolls, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            labelMotion = np.zeros((self.num_unrolls, 4), dtype=np.float32)

            # Read a new data sequence from batch cache and get the ground truth.
            (batchKey, images) = self.getData()
            # key = [data_idx, video_idx, track_id, image_seq]
            gtKey = batchKey
            initImageIndex = self.key_lookup[gtKey]
            if self.debug:
                print('Inital gtKey: ', gtKey)
                print('')

            # initial data
            # key = [data_idx, video_idx, track_id, image_seq]
            newKey = list(gtKey)
            newKey = tuple(newKey)
            imageIndex = self.key_lookup[newKey]
            bbox = self.datasets[newKey[0]][imageIndex, :4].copy()
            x1, y1, x2, y2 = bbox

            # image
            image = images[0]
            object = image[int(y1):int(y2), int(x1):int(x2)]
            object = cv2.resize(object, (IMG_SIZE, IMG_SIZE)) / 255.

            dataImage[:] = object

            # motion
            height, width, _ = image.shape
            """
            cx = float(int((x1 + x2) / 2) / width)
            cy = float(int((y1 + y2) / 2) / height)
            w = float((x2 - x1) / width)
            h = float((y2 - y1) / height)
            dataMotion[:] = [cx, cy, w, h]
            """

            x1 = float(x1 /width)
            y1 = float(y1 / height)
            x2 = float(x2 / width)
            y2 = float(y2 / height)

            dataMotion[:] = [x1, y1, x2, y2]

            # data
            for unroll in range(self.num_unrolls):

                if self.debug:
                    print('Unroll : ', unroll)
                    print('')
                    debug_image  = np.zeros((self.memory_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                    debug_motion = np.zeros((self.memory_size, 4), dtype=np.float32)

                    debug_image_label = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                    debug_motion_label = np.zeros((4), dtype=np.float32)

                for memory_idx in range(self.memory_size):

                    if unroll >= memory_idx:
                        # key = [data_idx, video_idx, track_id, image_seq]
                        newKey = list(gtKey)
                        newKey[3] += unroll - memory_idx
                        newKey = tuple(newKey)
                        imageIndex = self.key_lookup[newKey]
                        bbox = self.datasets[newKey[0]][imageIndex, :4].copy()
                        x1, y1, x2, y2 = bbox

                        # image
                        image = images[unroll - memory_idx]
                        object = image[int(y1):int(y2), int(x1):int(x2)]
                        object = cv2.resize(object, (IMG_SIZE, IMG_SIZE)) / 255.

                        dataImage[unroll, memory_idx] = object

                        # motion
                        height, width, _ = image.shape
                        """
                        cx = float(int((x1 + x2) / 2) / width)
                        cy = float(int((y1 + y2) / 2) / height)
                        w = float((x2 - x1) / width)
                        h = float((y2 - y1) / height)
                        dataMotion[:] = [cx, cy, w, h]
                        """

                        x1 = float(x1 / width)
                        y1 = float(y1 / height)
                        x2 = float(x2 / width)
                        y2 = float(y2 / height)

                        dataMotion[:] = [x1, y1, x2, y2]


                    if self.debug:
                            # debug each
                            debug_image1 = image.copy()
                            drawing.drawRect(debug_image1, [x1, y1, x2, y2], 2, [255, 0, 0])
                            #cv2.imshow('debug bbox', debug_image1)

                            path = self.image_paths[newKey[0]][newKey[-1]]
                            print('Memory idx  : ', memory_idx)
                            print('gtKey       : ', newKey)
                            print('bbox        : ', bbox)
                            print('bbox(float) : ', [x1, y1, x2, y2])
                            print('Image idx   : ', imageIndex)
                            print('Image path  : ', path)
                            print('')
                            debug_image[memory_idx] = object
                            debug_motion[memory_idx] = [x1, y1, x2, y2]

                            #cv2.waitKey(0)

                # label
                # key = [data_idx, video_idx, track_id, image_seq]
                newKey = list(gtKey)
                newKey[3] += unroll + 1
                newKey = tuple(newKey)
                imageIndex = self.key_lookup[newKey]
                bbox = self.datasets[newKey[0]][imageIndex, :4].copy()
                x1, y1, x2, y2 = bbox

                # image
                image = images[unroll + 1]
                object = image[int(y1):int(y2), int(x1):int(x2)]
                object = cv2.resize(object, (IMG_SIZE, IMG_SIZE)) / 255.

                labelImage[unroll] = object

                # motion
                height, width, _ = image.shape
                """
                cx = float(int((x1 + x2) / 2) / width)
                cy = float(int((y1 + y2) / 2) / height)
                w = float((x2 - x1) / width)
                h = float((y2 - y1) / height)
                dataMotion[:] = [cx, cy, w, h]
                """

                x1 = float(x1 / width)
                y1 = float(y1 / height)
                x2 = float(x2 / width)
                y2 = float(y2 / height)

                labelMotion[:] = [x1, y1, x2, y2]

                if self.debug:
                    # debug each
                    #debug_image1 = image.copy()
                    #drawing.drawRect(debug_image1, [x1, y1, x2, y2], 2, [255, 0, 0])
                    #cv2.imshow('debug bbox', debug_image1)

                    path = self.image_paths[newKey[0]][newKey[-1]]
                    print('[label]')
                    print('gtKey       : ', newKey)
                    print('bbox        : ', bbox)
                    print('bbox(float) : ', [x1, y1, x2, y2])
                    print('Image idx   : ', imageIndex)
                    print('Image path  : ', path)
                    print('')
                    debug_image_label = object
                    debug_motion_label = [x1, y1, x2, y2]

                    plots = []
                    for idx in range(self.memory_size):
                        #print('Memory idx :', idx, debug_motion[idx])
                        plots.append(dataImage[unroll, idx])

                    subplot = np.zeros((IMG_SIZE * self.memory_size, IMG_SIZE, 3), dtype=np.float32)
                    cv2.vconcat(tuple(plots), subplot)
                    cv2.imshow('external memory', subplot)
                    cv2.imshow('prediction', labelImage[unroll])
                    cv2.waitKey(0)

            dataImage = dataImage.reshape(([self.num_unrolls * self.memory_size] + list(dataImage.shape[2:])))
            dataMotion = dataMotion.reshape(([self.num_unrolls * self.memory_size] + list(dataMotion.shape[2:])))

            return (dataImage, dataMotion), (labelImage, labelMotion)

        except Exception as e:
            import traceback
            traceback.print_exc()
            import pdb
            pdb.set_trace()
            print('exception')


    def generator(self):
        while True:
            yield self.get_data_sequence()


    def get_dataset(self, batch_size):
        def get_data_generator(ind):
            dataset = tf.data.Dataset.from_generator(self.generator, ((tf.float32, tf.float32), (tf.float32, tf.float32)))
            dataset = dataset.prefetch(int(np.ceil(self.batch_size * 1.0 / PARALLEL_SIZE)))
            return dataset

        dataset = tf.data.Dataset.from_tensor_slices(list(range(PARALLEL_SIZE))).interleave(
                get_data_generator, cycle_length=PARALLEL_SIZE)

        dataset = dataset.batch(batch_size)
        dataset_iterator = dataset.make_one_shot_iterator()
        return dataset_iterator



if __name__ == '__main__':
    port = 9981
    num_unrolls = 10
    mem_size = 10
    debug = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    # dataset
    dataset = Dataset(num_unrolls, 1, mem_size, port, debug=debug)

    iteration = 0
    while True:
        iteration += 1
        print('iteration', iteration)
        dataset.get_data_sequence()




    """
    # initialization
    tf.Graph().as_default()
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))

    # dataset
    dataset = Dataset(num_unrolls, 1, mem_size, port)

    # network
    from Appearance.model.tf_customVGG16 import customVGG16 as network

    tf_image_input = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))
    tf_label_input = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))

    tf_image_output = network(inputs=tf_image_input, feature_size=256)
    tf_label_output = network(inputs=tf_label_input, feature_size=256, reuse=True)

    # Initialize the network and load saved parameters
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf_util.restore_from_dir(sess, os.path.join(str(os.path.join(os.path.dirname(__file__), os.path.pardir, 'logs')), 'checkpoints'))

    sess.graph.finalize()


    while True:

        (dataImage, dataMotion), (labelImage, labelMotion) = dataset.get_data_sequence()
        dataImage = np.reshape(dataImage, [1 * num_unrolls * mem_size, IMG_SIZE, IMG_SIZE, 3])

        image_data_feature = sess.run([tf_image_output], feed_dict={tf_image_input: dataImage})
        label_data_feature = sess.run([tf_label_output], feed_dict={tf_label_input: labelImage})

        dataImage = np.reshape(dataImage, [num_unrolls, mem_size, IMG_SIZE, IMG_SIZE, 3])

        image_data_feature = np.reshape(image_data_feature, [num_unrolls, mem_size, 256])
        label_data_feature = np.reshape(label_data_feature, [num_unrolls, 256])

        for unroll in range(num_unrolls):

            image_data = dataImage[unroll]
            memory = np.zeros((IMG_SIZE * mem_size, IMG_SIZE, 3), dtype=np.float32)
            label_image  = labelImage[unroll]
            plots = []
            print('Unroll                : %d' % unroll)


            for i in range(mem_size):
                plots.append(image_data[i])

                # cal dist
                dist = np.sum(np.abs(image_data_feature[unroll][i] - label_data_feature[unroll]), axis=0)
                print('Memory idx', i ,'distance : %.3f' %dist, 'feature_summation: ', np.sum(image_data_feature[unroll][i]))
            print('')

            cv2.vconcat(tuple(plots), memory)
            cv2.imshow('memory', memory)
            cv2.imshow('label', label_image)
            cv2.waitKey(0)

    """





