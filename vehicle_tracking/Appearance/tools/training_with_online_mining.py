import argparse
import cv2
import os
import glob
import numpy as np
import tensorflow as tf
import Appearance.utils.tf_dataset_online_mining as tf_dataset
import Appearance.model.online_triplet_wraper as network
import Appearance.utils.tf_util as tf_util
import time

HOST = 'localhost'

IMG_SIZE = 160
FETURE_SIZE = 256

def main(FLAGS):
    learning_rate = FLAGS.learning_rate
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
    tf_dataset_obj = tf_dataset.Dataset(batch_size, port)
    tf_dataset_iterator = tf_dataset_obj.get_dataset(batch_size)
    image_batch, label_batch = tf_dataset_iterator.get_next()
    image_batch = tf.reshape(image_batch, (batch_size, IMG_SIZE, IMG_SIZE, 3))
    label_batch = tf.reshape(label_batch, [batch_size])

    learningRate = tf.placeholder(tf.float32)

    tf_features = network.triplet_model(inputs=image_batch, batch_size=batch_size, train=True, reuse=False, feature_size=FETURE_SIZE)

    if FLAGS.debug:
        tf_loss, tf_pos_loss, tf_neg_loss, tf_kl_score, tf_pos_mask, tf_neg_mask, tf_positive_kl_score_matrix, tf_negative_kl_score_matrix = network.KL_loss(tf_features, label_batch, debug=True)
    else:
        tf_loss, tf_pos_loss, tf_neg_loss= network.KL_loss(tf_features, label_batch, debug=False)
    train_op = network.training(tf_loss, learningRate)

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
                _, lossValue, posLoss, negLoss, kl_score, pos_mask, neg_mask, labels, positive_kl_score_matrix, negative_kl_score_matrix = \
                    sess.run([train_op, tf_loss, tf_pos_loss, tf_neg_loss, tf_kl_score, tf_pos_mask, tf_neg_mask, label_batch, tf_positive_kl_score_matrix, tf_negative_kl_score_matrix], feed_dict={learningRate : learning_rate})
            else:
                _, lossValue, posLoss, negLoss = sess.run([train_op, tf_loss, tf_pos_loss, tf_neg_loss], feed_dict={learningRate : learning_rate})

            endSolver = time.time()

            numIters  += 1
            iteration += 1

            timeTotal += (endSolver - startSolver)
            if (iteration -1) % 10 == 0:
                print('Iteration:       %d' %(iteration-1))
                print('Positive Loss:   %.3f' % posLoss)
                print('Negative Loss:   %.3f' % negLoss)
                print('Total Loss:      %.3f' % lossValue)
                #print('Average Time:    %.3f' % (timeTotal/ numIters))
                #print('Current Time:    %.3f' % (endSolver - startSolver))
                print('')

            if FLAGS.debug and (iteration -1) % 100 == 0:
                print('Labels\n', labels)
                print('pos_mask\n', pos_mask)
                print('neg_mask\n', neg_mask)
                print('kl_score\n', kl_score)
                print('pos_kl_score_matrix\n', positive_kl_score_matrix)
                print('neg_kl_score_matrix\n', negative_kl_score_matrix)
                #print('Average Time:    %.3f' % (timeTotal/ numIters))
                #print('Current Time:    %.3f' % (endSolver - startSolver))
                print('')

            # save a checkpoint and remove a old one
            if iteration % 100 == 0 or iteration == FLAGS.epoch:
                checkpoint_file = os.path.join(log_dir, 'checkpoints', 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=iteration)

                """
                files = glob.glob(log_dir + '/checkpoints/*')
                for file in files:
                    basename = os.path.basename(file)
                    if os.path.isfile(file) and str(iteration) not in file and 'checkpoint' not in basename:
                        os.remove(file)
                """

    except:
        # save if error or killed by ctrl - c
        print('Saving...')
        checkpoint_file = os.path.join(log_dir, 'checkpoints', 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=iteration)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training for Triplet network')
    parser.add_argument('-b', '--batch_size', action='store', default=64, type=int)
    parser.add_argument('-c', '--cuda_visible_devices', default=str(1), type=str, help='Device number or string')
    parser.add_argument('-r', '--restore', default=False, action='store_true')
    parser.add_argument('-p', '--port', default=9980, action='store', dest='port', type=int)
    parser.add_argument('-l', '--learning_rate', default=float(1e-5), type=float, action='store')
    parser.add_argument('-e', '--epoch', default=int(1e5), type=int, action='store')
    parser.add_argument('-s', '--log_dir', default=str(os.path.join(os.path.dirname(__file__), os.path.pardir, 'logs')), type=str)
    parser.add_argument('-d', '--debug', default=False, action='store_true')
    FLAGS = parser.parse_args()

    main(FLAGS)