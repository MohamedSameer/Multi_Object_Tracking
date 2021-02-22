import tensorflow as tf
import numpy as np
from Appearance.utils import tf_util

#regulizer = tf.contrib.layers.l2_regulizer(scale=0.0001)
regulizer = None

def customVGG16(inputs, feature_size, train=False, reuse=None):

    with tf.variable_scope('customVGG16', reuse=reuse):

        with tf.variable_scope('block1'):
                                    #input, num, filter_size
            block1_conv1 = tf.layers.conv2d(inputs, 32, [3, 3], padding='same', activation=tf.nn.relu, use_bias=True, bias_initializer=tf.zeros_initializer(),
                                            name='conv1', kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=regulizer , trainable=train, reuse=reuse)
            block1_conv2 = tf.layers.conv2d(block1_conv1, 32, [3, 3], padding='same', activation=tf.nn.leaky_relu, use_bias=False,
                                            name='conv2', kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=regulizer, trainable=train, reuse=reuse)

            block1_pool = tf.layers.max_pooling2d(block1_conv2, [2, 2], [2, 2], name='pool')

        with tf.variable_scope('block1_skip'):

            conv1_skip = tf.layers.conv2d(block1_pool, 16, [1, 1], padding='same', activation=tf.nn.relu, use_bias=True, bias_initializer=tf.zeros_initializer(),
                                          name='conv1', kernel_initializer=tf.initializers.he_normal(), kernel_regularizer=regulizer, trainable=train, reuse=reuse)

            conv1_skip = tf.transpose(conv1_skip, perm=[0,3,1,2])
            conv1_skip_flat = tf_util.remove_axis(conv1_skip, [2,3])

        with tf.variable_scope('block2'):
            # input, num, filter_size
            block2_conv1 = tf.layers.conv2d(block1_pool, 64, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv1',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)
            block2_conv2 = tf.layers.conv2d(block2_conv1, 64, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv2',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)

            block2_pool = tf.layers.max_pooling2d(block2_conv2, [2, 2], [2, 2], name='pool')

        with tf.variable_scope('block3'):
            # input, num, filter_size
            block3_conv1 = tf.layers.conv2d(block2_pool, 128, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv1',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)
            block3_conv2 = tf.layers.conv2d(block3_conv1, 128, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv2',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)
            block3_conv3 = tf.layers.conv2d(block3_conv2, 128, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv3',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)

            block3_pool = tf.layers.max_pooling2d(block3_conv3, [2, 2], [2, 2], name='pool')

        with tf.variable_scope('block3_skip'):
            conv3_skip = tf.layers.conv2d(block3_pool, 64, [1, 1], padding='same', activation=tf.nn.relu,
                                          use_bias=True, bias_initializer=tf.zeros_initializer(),
                                          name='conv1',
                                          kernel_initializer=tf.initializers.he_normal(),
                                          kernel_regularizer=regulizer, trainable=train, reuse=reuse)

            conv3_skip = tf.transpose(conv3_skip, perm=[0, 3, 1, 2])
            conv3_skip_flat = tf_util.remove_axis(conv3_skip, [2, 3])

        with tf.variable_scope('block4'):
            # input, num, filter_size
            block4_conv1 = tf.layers.conv2d(block3_pool, 256, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv1',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)
            block4_conv2 = tf.layers.conv2d(block4_conv1, 256, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv2',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)
            block4_conv3 = tf.layers.conv2d(block4_conv2, 256, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv3',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)

            block4_pool = tf.layers.max_pooling2d(block4_conv3, [2, 2], [2, 2], name='pool')

        with tf.variable_scope('block5'):
            # input, num, filter_size
            block5_conv1 = tf.layers.conv2d(block4_pool, 256, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv1',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)
            block4_conv2 = tf.layers.conv2d(block5_conv1, 256, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv2',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)
            block4_conv3 = tf.layers.conv2d(block4_conv2, 256, [3, 3], padding='same', activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                     name='conv3',
                                     kernel_initializer=tf.initializers.he_normal(),
                                     kernel_regularizer=regulizer, trainable=train, reuse=reuse)

            block5_pool = tf.layers.max_pooling2d(block4_conv3, [2, 2], [2, 2], name='pool')
            block5_pool_skip = tf.transpose(block5_pool, perm=[0, 3, 1, 2])
            block5_pool_skip_flat = tf_util.remove_axis(block5_pool_skip, [2, 3])

        with tf.variable_scope('big_concat'):
            # Concat all skip layers
            skip_concat = tf.concat([conv1_skip_flat, conv3_skip_flat, block5_pool_skip_flat], 1)
            """
            skip_concat2 = tf.layers.dense(skip_concat, feature_size, activation=tf.nn.relu,  use_bias=True, bias_initializer=tf.zeros_initializer(),
                                  kernel_initializer=tf.initializers.he_normal(),
                                  kernel_regularizer=regulizer, trainable=train, reuse=reuse, name='fc_skip_concat')
            """

        with tf.variable_scope('fc_out'):
            out = tf.layers.dense(skip_concat, feature_size, activation=tf.nn.softmax, use_bias=True, bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=regulizer, trainable=train, reuse=reuse, name='fc_out')

        return out


if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batchSize = 5

    image = tf.placeholder(tf.float32, shape=(5, 160, 160, 3))

    customVGG16(inputs=image, feature_size=256)



