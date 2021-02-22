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

from Appearance.utils import get_datasets

from io import BytesIO

from Appearance.utils import bb_util
from Appearance.utils import drawing

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

HOST = 'localhost'

PARALLEL_SIZE = 4

IMG_SIZE = 160

class Dataset(object):
    def __init__(self, prefetch_size, port, debug=False):

        self.prefetch_size = prefetch_size
        self.port = port
        self.debug = debug

        self.key_lookup = dict()
        self.datasets = []

        self.add_dataset('kitti_tracking')

    def add_dataset(self, dataset_name):
        dataset_ind = len(self.datasets)
        dataset_gt = get_datasets.get_data_for_dataset(dataset_name, 'train')['gt']

        # gt = [video_id, main_img_seq, pos_img_seq, main_track_id, neg_track_id, main_x1, main_y1, main_x2, main_y2, pos_x1, pos_y1, pos_x2, pos_y2, neg_x1, neg_y1, neg_x2, neg_y2]
        for xx in range(dataset_gt.shape[0]):
            start_line = dataset_gt[xx,:].astype(int)
            self.key_lookup[(dataset_ind, start_line[0], start_line[1], start_line[2], start_line[3], start_line[4])] = xx
        self.datasets.append(dataset_gt)

    def getData(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, self.port))
        sock.sendall(('Ready' + '\n').encode('utf-8'))
        receivedBytes = sock.recv(4)
        messageLength = struct.unpack('>I', receivedBytes)[0]
        key = sock.recv(messageLength)
        key = pickle.loads(key)

        images = [None] * 2

        for nn in range(2):
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

    # Randomly jitter the box for a bit of noise.
    def add_noise(self, bbox, prevBBox, imageWidth, imageHeight):
        numTries = 0
        bboxXYWHInit = bb_util.xyxy_to_xywh(bbox)
        while numTries < 10:
            bboxXYWH = bboxXYWHInit.copy()
            centerNoise = np.random.laplace(0,1.0/5,2) * bboxXYWH[[2,3]]
            sizeNoise = np.clip(np.random.laplace(1,1.0/15,2), .6, 1.4)
            bboxXYWH[[2,3]] *= sizeNoise
            bboxXYWH[[0,1]] = bboxXYWH[[0,1]] + centerNoise
            if not (bboxXYWH[0] < prevBBox[0] or bboxXYWH[1] < prevBBox[1] or
                bboxXYWH[0] > prevBBox[2] or bboxXYWH[1] > prevBBox[3] or
                bboxXYWH[0] < 0 or bboxXYWH[1] < 0 or
                bboxXYWH[0] > imageWidth or bboxXYWH[1] > imageHeight):
                numTries = 10
            else:
                numTries += 1

        return self.fix_bbox_intersection(bb_util.xywh_to_xyxy(bboxXYWH), prevBBox, imageWidth, imageHeight)

    # Make sure there is a minimum intersection with the ground truth box and the visible crop.
    def fix_bbox_intersection(self, bbox, gtBox, imageWidth, imageHeight):
        if type(bbox) == list:
            bbox = np.array(bbox)
        if type(gtBox) == list:
            gtBox = np.array(gtBox)

        gtBoxArea = float((gtBox[3] - gtBox[1]) * (gtBox[2] - gtBox[0]))
        bboxLarge = bb_util.scale_bbox(bbox, CROP_PAD)
        while IOU.intersection(bboxLarge, gtBox) / gtBoxArea < AREA_CUTOFF:
            bbox = bbox * .9 + gtBox * .1
            bboxLarge = bb_util.scale_bbox(bbox, CROP_PAD)
        return bbox

    def get_data_sequence(self):

        try:
            # Preallocate the space for the images and labels.
            tImage = np.zeros((3, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            labels = np.zeros((2), dtype=np.float32)

            # Read a new data sequence from batch cache and get the ground truth.
            (batchKey, images) = self.getData()
            gtKey = batchKey

            # gtKey = [dataset_ind, video_id, main_img_seq, pos_img_seq, main_track_id, neg_track_id]
            keyIndex = self.key_lookup[gtKey]

            # self.datasets[gtKey[0]][imageIndex] =
            # [video_id, main_img_seq, pos_img_seq, main_track_id, neg_track_id, main_x1, main_y1, main_x2, main_y2, pos_x1, pos_y1, pos_x2, pos_y2, neg_x1, neg_y1, neg_x2, neg_y2]
            data = self.datasets[gtKey[0]][keyIndex].copy()
            video_id, main_img_seq, pos_img_seq, main_track_id, neg_track_id, main_x1, main_y1, main_x2, main_y2, pos_x1, pos_y1, pos_x2, pos_y2, neg_x1, neg_y1, neg_x2, neg_y2 = data

            # main object
            main_object = images[0][int(main_y1):int(main_y2), int(main_x1):int(main_x2)]
            main_object = cv2.resize(main_object, (IMG_SIZE, IMG_SIZE))
            tImage[0, ...] = main_object / 255.

            # pos object
            pos_object = images[1][int(pos_y1):int(pos_y2), int(pos_x1):int(pos_x2)]
            pos_object = cv2.resize(pos_object, (IMG_SIZE, IMG_SIZE))
            tImage[1, ...] = pos_object / 255.

            # neg object
            neg_object = images[1][int(neg_y1):int(neg_y2), int(neg_x1):int(neg_x2)]
            neg_object = cv2.resize(neg_object, (IMG_SIZE, IMG_SIZE))
            tImage[2, ...] = neg_object / 255.


            if self.debug:
                # Look at the inputs to make sure they are correct.
                image0 = images[0].copy()
                image1 = images[1].copy()

                # main
                drawing.drawRect(image0, [main_x1, main_y1, main_x2, main_y2], 2, [255, 0, 0])
                # pos
                drawing.drawRect(image1, [pos_x1, pos_y1, pos_x2, pos_y2], 2, [255, 0, 0])
                # neg
                drawing.drawRect(image1, [neg_x1, neg_y1, neg_x2, neg_y2], 2, [0, 255, 0])


                cv2.imshow('debug-main', image0)
                cv2.imshow('debug-next', image1)

                cv2.imshow('main', tImage[0])
                cv2.imshow('pos', tImage[1])
                cv2.imshow('neg', tImage[2])
                cv2.waitKey(0)


            #tImage = tImage.reshape([3] + list(tImage.shape[1:]))
            labels[0] = 1.0
            labels[1] = 0.0

            return tImage, labels

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
            dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32))
            dataset = dataset.prefetch(int(np.ceil(self.prefetch_size * 1.0 / PARALLEL_SIZE)))
            return dataset

        dataset = tf.data.Dataset.from_tensor_slices(list(range(PARALLEL_SIZE))).interleave(
                get_data_generator, cycle_length=PARALLEL_SIZE)

        dataset = dataset.batch(batch_size)
        dataset_iterator = dataset.make_one_shot_iterator()
        return dataset_iterator



if __name__ == '__main__':
    port = 9981
    debug = True
    """
    dataset = Dataset(1, port, debug)

    iteration = 0
    while True:
        iteration += 1
        print('iteration', iteration)
        dataset.get_data_sequence()
    """

    tf_dataset_obj = Dataset(prefetch_size=5, port=port)
    tf_dataset_iterator = tf_dataset_obj.get_dataset(5)
    imageBatch, labelsBatch = tf_dataset_iterator.get_next()
    sess = tf.Session()

    image_batch, label_batch = sess.run([imageBatch, labelsBatch])
    print(image_batch.shape)

