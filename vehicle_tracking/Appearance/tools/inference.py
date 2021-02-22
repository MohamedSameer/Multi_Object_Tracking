import argparse
import cv2
import os
import glob
import numpy as np
import tensorflow as tf
import Appearance.utils.tf_dataset as tf_dataset
# networks
from Appearance.model.tf_customVGG16 import customVGG16 as network
import Appearance.utils.tf_util as tf_util
import time

HOST = 'localhost'

IMG_SIZE = 160

def KL2(x, y):
    epsilon = 0.000001

    X = x + epsilon
    Y = y + epsilon

    divergence = np.sum(X * np.log(X/Y))

    return divergence

def main(FLAGS):
    learning_rate = FLAGS.learning_rate
    batch_size    = FLAGS.batch_size
    port          = FLAGS.port
    log_dir       = FLAGS.log_dir

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.cuda_visible_devices)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)

    # Tensorflow setup
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(log_dir + '/checkpoints'):
        os.mkdir(log_dir + '/checkpoints')

    tf.Graph().as_default()
    tf.logging.set_verbosity(tf.logging.INFO)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))

    # dataset
    dataset = tf_dataset.Dataset(1, port, debug=False)
    tf_dataset_obj = tf_dataset.Dataset(batch_size*3, port)

    # network
    tf_image_input  = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))
    tf_image_output = network(tf_image_input, feature_size=256)

    # Initialize the network and load saved parameters
    init = tf.global_variables_initializer()
    sess.run(init)
    tf_util.restore_from_dir(sess, os.path.join(log_dir, 'checkpoints'))

    sess.graph.finalize()

    var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size for v in
                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    print('total parameter size :', sum(var_sizes) / (1024 ** 2), 'MB')

    while True:
        image, labels = dataset.get_data_sequence()

        image_feature = sess.run([tf_image_output], feed_dict={tf_image_input : image})
        image_feature = np.reshape(image_feature, [3, -1])

        main_feature = image_feature[0]
        pos_feature  = image_feature[1]
        neg_feature  = image_feature[2]
        print(main_feature)
        print(pos_feature)

        pos_kl = KL2(pos_feature, main_feature)
        neg_kl = KL2(neg_feature, main_feature)

        pos_dist = np.sum(np.abs(main_feature - pos_feature), axis=0)
        neg_dist = np.sum(np.abs(main_feature - neg_feature), axis=0)

        print('Positive_dist :     %.3f' % pos_dist)
        print('Negative_dist :     %.3f' % neg_dist)
        print('Positive_KL   :     %.3f' % pos_kl)
        print('Negative_KL   :     %.3f' % neg_kl)
        print('')

        main_image = image[0]
        pos_image  = image[1]
        neg_image  = image[2]

        cv2.imshow('main', main_image)
        cv2.imshow('pos', pos_image)
        cv2.imshow('neg', neg_image)

        cv2.waitKey(0)





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training for Triplet network')
    parser.add_argument('-b', '--batch_size', action='store', default=1, type=int)
    parser.add_argument('-c', '--cuda_visible_devices', default=str(1), type=str, help='Device number or string')
    parser.add_argument('-r', '--restore', default=False, action='store_true')
    parser.add_argument('-p', '--port', default=9981, action='store', dest='port', type=int)
    parser.add_argument('-l', '--learning_rate', default=float(1e-5), type=float, action='store')
    parser.add_argument('-e', '--epoch', default=int(1e5), type=int, action='store')
    parser.add_argument('-s', '--log_dir', default=str(os.path.join(os.path.dirname(__file__), os.path.pardir, 'logs')), type=str)
    parser.add_argument('-d', '--debug', default=False, action='store_true')
    FLAGS = parser.parse_args()

    main(FLAGS)