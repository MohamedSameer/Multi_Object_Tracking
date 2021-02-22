import tensorflow as tf
from Appearance.model.tf_customVGG16 import customVGG16
from utils import tf_util
import numpy as np
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

FEATURE_SIZE = 512
IMG_SIZE = 224

msra_initializer = tf.contrib.layers.variance_scaling_initializer()

def inference(inputs, memory_size, train, batch_size, prevLstmState=None, reuse=None ):
    # Data should be in order [B x M x H x W x C, B x M x Motion_vectors] where M is the size of external memory

    variable_list = []

    images  = inputs[0]
    motions = inputs[1]

    if reuse is not None and not reuse:
        reuse = None


def training(inputs, num_unrolls, mem_size, batch_size, labels, debug=False, train=False, reuse=None, feature_size=256):
    # Data should be inorder ( [B x N x M , Image(fixed)] , [B x N x M ,Motion(4)] ), where B: batch size, N : number of unrolls, M : external memory size

    # image_data  = [batch x num_unrolls x mem_size, IMG_SIZE, IMG_SIZE, 3]
    # motion_data = [batch x num_unrolls x mem_size, 4]
    image_data, motion_feature = inputs

    # extract appearance feature
    # appearance_feature = [batch x num_unrolls x mem_size, FEATURE_SIZE]
    appearance_feature = customVGG16(inputs=image_data, feature_size=feature_size, train=False, reuse=None)

    # appearance = [Batch, num_unrolls, mem_size, features]
    appearance_feature = tf.reshape(appearance_feature, [batch_size, num_unrolls, mem_size, appearance_feature.get_shape().as_list()[-1]])

    # motion     = [Batch, num_unrolls, mem_size, 4(x, y, w, h)]
    motion_feature = tf.reshape(motion_feature, [batch_size, num_unrolls, mem_size, motion_feature.get_shape().as_list()[-1]])

    # RAN
    if debug:
        predicted_appearance_feature, predicted_bbox, appearance_alpha, motion_alpha = RAN(appearance_feature, motion_feature, batch_size, num_unrolls, mem_size, reuse=False, debug=debug)
    else:
        predicted_appearance_feature, predicted_bbox = RAN(appearance_feature, motion_feature, batch_size, num_unrolls, mem_size, reuse=False, debug=debug)

    # extractor feature vector of image labels
    # image_label_feature = [batch * num_unroll, features]
    image_label_feature = customVGG16(inputs=labels, feature_size=feature_size, train=False, reuse=True)

    if debug:
        return predicted_appearance_feature, predicted_bbox, image_label_feature, appearance_alpha, motion_alpha, appearance_feature
    else:
        return predicted_appearance_feature, predicted_bbox, image_label_feature


def RAN(appearance_external_memory, motion_external_memory, batch_size, num_unrolls, mem_size, prev_lstm_state1=None, prev_lstm_state2=None, reuse=None, debug=False):
    # external_memory should be in order [B, N, M, F] where B: batch size, N : number of unrolls, M : external memory size
    # appearance external memory = [Batch, num_unrolls, mem_size, FEATURE_SIZE]
    # motion external memory     = [Batch, num_unrolls, mem_size, 4(x,y,w,h)]

    if reuse is not None and not reuse:
        reuse = None

    feature_size = appearance_external_memory.get_shape().as_list()[-1]

    if mem_size == 1:
        cell_ratio = 1
    else:
        cell_ratio = int(mem_size/2)

    with tf.variable_scope('RAN', reuse=reuse):

        # Update LSTM
        swap_memory = num_unrolls > 1
        with tf.variable_scope('appearance_lstm'):
            # Initialize LSTM cell
            appearance_lstm = tf.contrib.rnn.LSTMCell(feature_size, use_peepholes=True, initializer=msra_initializer, reuse=reuse)

            if prev_lstm_state1 is not None:
                appearance_lstm_state = tf.contrib.rnn.LSTMStateTuple(prev_lstm_state1[0], prev_lstm_state1[1])
            else:
                appearance_lstm_state = appearance_lstm.zero_state(batch_size, dtype=tf.float32)
            # rehsape memory dimension to [batch, unroll, mem_size x FEATURE_SIZE]
            appearance_external_memory_to_lstm = tf.reshape(appearance_external_memory, [batch_size, num_unrolls, -1])

            # forward memory to LSTM
            appearance_output, appearance_lstm_state = tf.nn.dynamic_rnn(appearance_lstm, appearance_external_memory_to_lstm, initial_state=appearance_lstm_state, swap_memory=swap_memory)
            # appearance_output = [batch_size * num_unroll, LSTM_Cell_size]
            appearance_output = tf.reshape(appearance_output, [batch_size * num_unrolls, -1])

        with tf.variable_scope('motion_lstm'):
            # Initialization LSTM cell
            motion_lstm = tf.contrib.rnn.LSTMCell(32, use_peepholes=True, initializer=msra_initializer, reuse=reuse)

            if prev_lstm_state2 is not None:
                motion_lstm_state = tf.contrib.rnn.LSTMStateTuple(prev_lstm_state2[0], prev_lstm_state2[1])
            else:
                motion_lstm_state = motion_lstm.zero_state(batch_size, dtype=tf.float32)

            # reshape memory dimension to [batch, num_unroll, mem_size x 4]
            motion_external_memory_to_lstm = tf.reshape(motion_external_memory, [batch_size, num_unrolls, -1])

            # forward memory to LSTM
            motion_output, motion_lstm_state = tf.nn.dynamic_rnn(motion_lstm, motion_external_memory_to_lstm, initial_state=motion_lstm_state, swap_memory=swap_memory)
            # motion_output = [batch_size * num_unroll, LSTM_Cell_size]
            motion_output = tf.reshape(motion_output, [batch_size * num_unrolls, -1])

        # get AR parameters
        with tf.variable_scope('fc_appearance'):
            # appearance_AR_param = [batch x num_unroll, mem_size + feature_size]
            appearance_AR_param = tf.layers.dense(appearance_output, mem_size + feature_size, activation=None, use_bias=True,
                                  bias_initializer=tf.zeros_initializer(), reuse=reuse)

            # appearance_alpha = [batch * num_unrolls, mem_size]
            appearance_alpha = tf.nn.softmax(appearance_AR_param[:, :mem_size])
            # appearance_noise = [batch * num_unrolls, feature_size]
            appearance_noise = appearance_AR_param[:, mem_size:]

        with tf.variable_scope('fc_motion'):
            motion_AR_param = tf.layers.dense(motion_output, mem_size + 4, activation=None,
                                                  use_bias=True, bias_initializer=tf.zeros_initializer(), reuse=reuse)

            # motion_alpha = [batch * num_unrolls, 4]
            motion_alpha = tf.nn.softmax(motion_AR_param[:, :mem_size])
            # motion_noise = [batch * num_unrolls, feature_size]
            motion_noise = motion_AR_param[:, mem_size:]

        # predict next frame feature vector
        # appearance
        appearance_external_memory   = tf.reshape(appearance_external_memory, [batch_size * num_unrolls, mem_size, -1])
        appearance_alpha             = tf.expand_dims(appearance_alpha, axis=-1)
        predicted_appearance_feature = tf.reduce_sum(tf.math.multiply(appearance_alpha, appearance_external_memory) ,axis=1)
        # add noise
        predicted_appearance_feature = tf.nn.softmax(tf.add(predicted_appearance_feature, appearance_noise))

        # motion
        motion_external_memory = tf.reshape(motion_external_memory, [batch_size * num_unrolls, mem_size, -1])
        motion_alpha           = tf.expand_dims(motion_alpha, axis=-1)
        predicted_bbox         = tf.reduce_sum(tf.math.multiply(motion_alpha, motion_external_memory) ,axis=1)
        # add noise
        predicted_bbox         = tf.add(predicted_bbox, motion_noise)

        if debug:
            if prev_lstm_state1 is not None:
                return predicted_appearance_feature, predicted_bbox, appearance_lstm_state, motion_lstm_state, appearance_alpha, motion_alpha
            else:
                return predicted_appearance_feature, predicted_bbox, appearance_alpha, motion_alpha
        else:
            if prev_lstm_state1 is not None:
                return predicted_appearance_feature, predicted_bbox, appearance_lstm_state, motion_lstm_state
            else:
                return predicted_appearance_feature, predicted_bbox


def KL_Divergence(x, y):

    # KL Divergence
    kl_score = tf.math.multiply(x, tf.math.log(x / y))

    # ignore NaN
    mask = tf.constant([0.0], dtype=tf.float32, shape=[kl_score.get_shape().as_list()[0], kl_score.get_shape().as_list()[1]])
    kl_score = tf.where(tf.is_nan(kl_score), mask, kl_score)

    kl_score = tf.reduce_sum(kl_score, axis=1)
    kl_score = tf.reduce_mean(kl_score, axis=0)

    return kl_score


def loss(outputs, labels):
    predicted_appearance, predicted_motion = outputs
    appearance_label, motion_label = labels

    appearance_loss = KL_Divergence(appearance_label, predicted_appearance)
    motion_loss     = tf.reduce_mean(tf.reduce_sum(tf.abs(motion_label - predicted_motion), axis=1), axis=0)

    total_loss = appearance_loss + motion_loss

    return total_loss, appearance_loss, motion_loss


def train_optimizer(loss, learning_rate, var_list=None):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    with tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()

    if var_list is None:
        var_list = tf.trainable_variables()

    train_op = optimizer.minimize(loss, var_list=var_list, global_step=global_step, colocate_gradients_with_ops=True)

    return train_op


if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batchSize = 2
    num_unrolls = 10
    mem_size = 5

    image_data = tf.placeholder(tf.float32, shape=(batchSize * num_unrolls * mem_size, 160, 160, 3))
    motion_data = tf.placeholder(tf.float32, shape=(batchSize * num_unrolls * mem_size, 4))

    image_label = tf.placeholder(tf.float32, (batchSize * num_unrolls, IMG_SIZE, IMG_SIZE, 3))
    motion_label = tf.placeholder(tf.float32, (batchSize * num_unrolls, 4))

    predicted_appearance_feature, predicted_bbox, image_label_feature, appearance_alpha, motion_alpha = training((image_data, motion_data), num_unrolls, mem_size, batchSize, image_label, debug=True)
    total_loss, appearance_loss, motion_loss = loss((predicted_appearance_feature, predicted_bbox), (image_label_feature, motion_label))



