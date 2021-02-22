import argparse
import cv2
import os
import glob
import numpy as np
import tensorflow as tf
import utils.tf_dataset as tf_dataset
from model import kl_network as network
import Appearance.utils.tf_util as tf_util
import time

HOST = 'localhost'

IMG_SIZE = 224
FEATURE_SIZE = 256

def KL_Divergence(x, y):

    # KL Divergence
    kl_score = np.multiply(x, np.log(x / y))

    # ignore NaN

    #mask = tf.constant([0.0], dtype=tf.float32, shape=[kl_score.get_shape().as_list()[0], kl_score.get_shape().as_list()[1]])
    #kl_score = tf.where(tf.is_nan(kl_score), mask, kl_score)

    kl_score = np.sum(kl_score, axis=0)
    #kl_score = tf.reduce_mean(kl_score, axis=0)

    return kl_score

def main(FLAGS):
    learning_rate = FLAGS.learning_rate
    num_unrolls   = FLAGS.num_unrolls
    mem_size      = FLAGS.mem_size
    batch_size    = FLAGS.batch_size
    port          = FLAGS.port
    log_dir       = FLAGS.log_dir

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.cuda_visible_devices)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    # Tensorflow setup
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(log_dir + '/checkpoints'):
        os.mkdir(log_dir + '/checkpoints')

    tf.Graph().as_default()
    tf.logging.set_verbosity(tf.logging.INFO)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))

    # dataset
    tf_dataset_obj = tf_dataset.Dataset(num_unrolls, batch_size, mem_size, port)
    tf_dataset_iterator = tf_dataset_obj.get_dataset(batch_size)
    trainBatch, labelBatch = tf_dataset_iterator.get_next()
    image_batch, motion_batch = trainBatch
    image_label, tf_motion_label = labelBatch

    image_batch = tf.reshape(image_batch, (batch_size * num_unrolls * mem_size, IMG_SIZE, IMG_SIZE, 3))
    motion_batch = tf.reshape(motion_batch, (batch_size * num_unrolls * mem_size, 4))

    image_label = tf.reshape(image_label, (batch_size * num_unrolls, IMG_SIZE, IMG_SIZE, 3))
    tf_motion_label = tf.reshape(tf_motion_label, (batch_size * num_unrolls, 4))

    learningRate = tf.placeholder(tf.float32)

    if FLAGS.debug:
        tf_predicted_appearance_feature, tf_predicted_bbox, tf_image_label_feature, tf_appearance_feature = network.training((image_batch, motion_batch), num_unrolls, mem_size, batch_size, image_label, debug=True, feature_size=FEATURE_SIZE)
    else:
        tf_predicted_appearance_feature, tf_predicted_bbox, tf_image_label_feature = network.training((image_batch, motion_batch), num_unrolls, mem_size, batch_size, image_label, debug=False, feature_size=FEATURE_SIZE)

    tf_loss, tf_appearance_loss, tf_motion_loss = network.loss((tf_predicted_appearance_feature, tf_predicted_bbox), (tf_image_label_feature, tf_motion_label))

    # train without CNN
    train_vars = tf.trainable_variables()
    train_vars = [var for var in train_vars if 'RAN' in var.name]

    train_op = network.train_optimizer(tf_loss, learning_rate, train_vars)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Initialize the network and load saved parameters
    sess.run(init)
    startIter = 0

    if FLAGS.restore:
        startIter = tf_util.restore_from_dir(sess, os.path.join(log_dir, 'checkpoints'))

    sess.graph.finalize()

    try:
        timeTotal = 0.000001
        numIters  = 0
        iteration = startIter

        # Run training iterations in the main thread
        while iteration < FLAGS.epoch:

            startSolver = time.time()

            if FLAGS.debug:
                _, loss, appearance_loss, motion_loss, predicted_appearance_feature, predicted_bbox, image_label_feature, batch_image, batch_image_label, image_data_feature, motion_label = \
                    sess.run([train_op, tf_loss, tf_appearance_loss, tf_motion_loss, tf_predicted_appearance_feature, tf_predicted_bbox, tf_image_label_feature, image_batch, image_label, tf_appearance_feature, tf_motion_label],
                    feed_dict={learningRate: learning_rate})
            else:
                #_, lossValue, a_Loss, m_Loss, a_score = sess.run([train_op, tf_loss, tf_appearance_loss, tf_motion_loss, tf_appearance_score], feed_dict={learningRate : learning_rate})
                _, loss, appearance_loss, motion_loss = sess.run([train_op, tf_loss, tf_appearance_loss, tf_motion_loss], feed_dict={learningRate: learning_rate})

            endSolver = time.time()

            numIters  += 1
            iteration += 1

            #print(a_dist)
            #print(iteration)
            timeTotal += (endSolver - startSolver)
            if (iteration -1) % 10 == 0:
                print('Iteration:         %d' %(iteration-1))
                print('Appearance Loss:   %.3f' % appearance_loss)
                print('Motion Loss:       %.3f' % motion_loss)
                print('Total Loss:        %.3f' % loss)
                #print('Average Time:    %.3f' % (timeTotal/ numIters))
                #print('Current Time:    %.3f' % (endSolver - startSolver))
                print('')

                if FLAGS.debug and (iteration -1) % 100 == 0:

                    dataImage                      = np.reshape(batch_image, [batch_size, num_unrolls, mem_size, IMG_SIZE, IMG_SIZE, 3]) # train image
                    labelImage                     = np.reshape(batch_image_label, [batch_size, num_unrolls, IMG_SIZE, IMG_SIZE, 3])     # label image
                    image_data_feature             = np.reshape(image_data_feature, [batch_size, num_unrolls, mem_size, FEATURE_SIZE])            # cnn result of train image
                    image_label_feature            = np.reshape(image_label_feature, [batch_size, num_unrolls, FEATURE_SIZE])
                    predicted_appearance_feature   = np.reshape(predicted_appearance_feature, [batch_size, num_unrolls, FEATURE_SIZE])

                    predicted_bbox                 = np.reshape(predicted_bbox, [batch_size, num_unrolls, 4])
                    motion_label                   = np.reshape(motion_label, [batch_size, num_unrolls, 4])

                    batch = 0
                    batch_dataImage           = dataImage[batch]
                    batch_label_image         = labelImage[batch]
                    batch_image_data_feature  = image_data_feature[batch]
                    batch_image_label_feature = image_label_feature[batch]
                    batch_predicted_appearance_feature = predicted_appearance_feature[batch]
                    batch_predicted_bbox = predicted_bbox[batch]
                    batch_motion_label   = motion_label[batch]

                    for unroll in range(num_unrolls):

                        image_data  = batch_dataImage[unroll]
                        memory      = np.zeros((int(IMG_SIZE/2) * mem_size, int(IMG_SIZE/2), 3), dtype=np.float32)
                        label_image = batch_label_image[unroll]
                        unroll_score = KL_Divergence(batch_image_label_feature[unroll],
                                                     batch_predicted_appearance_feature[unroll])

                        plots = []
                        print('Unroll                : %d' % unroll)
                        print('Prediction KL score   : %.3f' % unroll_score) # prediction with ground truth
                        print('GT Motion             : ', batch_motion_label[unroll])
                        print('Predicted Motion      : ', batch_predicted_bbox[unroll])

                        for i in range(mem_size):
                            plots.append(cv2.resize(image_data[i], (int(IMG_SIZE/2), int(IMG_SIZE/2))))

                            # cal dist
                            score = KL_Divergence(batch_image_label_feature[unroll], batch_image_data_feature[unroll][i])
                            #print('Memory idx', i, 'KL : %.3f' % score, 'feature_summation: ', np.sum(batch_image_data_feature[unroll][i]))
                        print('')

                        #cv2.vconcat(tuple(plots), memory)
                        #cv2.imshow('memory', memory)
                        #cv2.imshow('label', label_image)
                        #cv2.waitKey(1000)


            # save a checkpoint and remove a old one
            if iteration % 100 == 0 or iteration == FLAGS.epoch:
                checkpoint_file = os.path.join(log_dir, 'checkpoints', 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=iteration)

                files = glob.glob(log_dir + '/checkpoints/*')
                for file in files:
                    basename = os.path.basename(file)
                    if os.path.isfile(file) and str(iteration) not in file and 'checkpoint' not in basename:
                        os.remove(file)
    except:
        # save if error or killed by ctrl - c
        print('Saving...')
        checkpoint_file = os.path.join(log_dir, 'checkpoints', 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=iteration)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training for Triplet network')
    parser.add_argument('-b', '--batch_size', action='store', default=1, type=int)
    parser.add_argument('-n', '--num_unrolls', action='store', default=1, type=int)
    parser.add_argument('-m', '--mem_size', action='store', default=5, type=int)
    parser.add_argument('-c', '--cuda_visible_devices', default=str(0), type=str, help='Device number or string')
    parser.add_argument('-r', '--restore', default=False, action='store_true')
    parser.add_argument('-p', '--port', default=9981, action='store', dest='port', type=int)
    parser.add_argument('-l', '--learning_rate', default=float(1e-5), type=float, action='store')
    parser.add_argument('-e', '--epoch', default=int(1e10), type=int, action='store')
    parser.add_argument('-s', '--log_dir', default=str(os.path.join(os.path.dirname(__file__), os.path.pardir, 'logs')), type=str)
    parser.add_argument('-d', '--debug', default=False, action='store_true')
    FLAGS = parser.parse_args()

    main(FLAGS)