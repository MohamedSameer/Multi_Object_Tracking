import tensorflow as tf
#from Appearance.model import alexnet as network
from Appearance.model.tf_customVGG16 import customVGG16 as network

IMG_SIZE = 160

def triplet_model(inputs=None, batch_size=None, train=False, reuse=None):
    # Data should be B x 3 x H x W x C where 3 is (main, pos, neg)
    if batch_size is None:
        batch_size = int(inputs.get_shape().as_list()[0] / 3 )

    #inputs = tf.reshape(inputs, [batch_size, 3, IMG_SIZE, IMG_SIZE, 3])
    #main_image_batch = tf.reshape(inputs[:, 0, :], [batch_size, IMG_SIZE, IMG_SIZE, 3])
    #pos_image_batch = tf.reshape(inputs[:, 1, :], [batch_size, IMG_SIZE, IMG_SIZE, 3])
    #neg_image_batch = tf.reshape(inputs[:, 2, :], [batch_size, IMG_SIZE, IMG_SIZE, 3])

    #feature_vector = network.VGG16(inputs=inputs, feature_size=256)
    feature_vector = network(inputs=inputs, feature_size=256, train=train, reuse=reuse)

    feature_vector_shape = feature_vector.get_shape().as_list()
    feature_vector = tf.reshape(feature_vector, [batch_size, 3, feature_vector_shape[-1]])

    main_feature_vector = feature_vector[:, 0]
    pos_feature_vector  = feature_vector[:, 1]
    neg_feature_vector  = feature_vector[:, 2]

    return main_feature_vector, pos_feature_vector, neg_feature_vector


def compute_euclidean_distance(x,y):

    #d = tf.square(tf.subtract(x,y))
    #d = tf.sqrt(tf.reduce_sum(d, axis=1))
    d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=1)

    return d

def get_var_list():
    return tf.trainable_variables()

def KL_Divergence(x, y):

    # KL Divergence
    kl_score = tf.math.multiply(x, tf.math.log(x / y))

    # ignore NaN
    mask = tf.constant([0.0], dtype=tf.float32, shape=[kl_score.get_shape().as_list()[0], kl_score.get_shape().as_list()[1]])
    kl_score = tf.where(tf.is_nan(kl_score), mask, kl_score)

    kl_score = tf.reduce_sum(kl_score, axis=1)
    kl_score = tf.reduce_mean(kl_score, axis=0)

    return kl_score

def KL_loss(main, positive, negative, labels, debug=False):

    with tf.name_scope('KL_Divergence_loss'):

        # KL Divergence
        kl_score_pos = tf.math.multiply(main, tf.math.log(main / positive))
        kl_score_neg = tf.math.multiply(main, tf.math.log(main / negative))

        # ignore NaN
        mask = tf.constant([0.0], dtype=tf.float32, shape=[kl_score_pos.get_shape().as_list()[0], kl_score_pos.get_shape().as_list()[1]])
        kl_score_pos = tf.where(tf.is_nan(kl_score_pos), mask, kl_score_pos)
        kl_score_neg = tf.where(tf.is_nan(kl_score_neg), mask, kl_score_neg)

        kl_score_pos = tf.reduce_sum(kl_score_pos, axis=1)
        kl_score_neg = tf.reduce_sum(kl_score_neg, axis=1)

        #positive_loss = tf.reduce_mean(tf.abs(tf.exp(-kl_score_pos) - labels[:, 0]))
        #negative_loss = tf.reduce_mean(tf.abs(tf.exp(-kl_score_neg) - labels[:, 1]))

        positive_loss = tf.reduce_mean(kl_score_pos)
        negative_loss = tf.reduce_mean(tf.maximum(1.0 - kl_score_neg, 0.0))

        loss = positive_loss + negative_loss

        # L2 Loss on variables.
    with tf.variable_scope('l2_weight_penalty'):
        l2_weight_penalty = 0.00001 * tf.add_n([tf.nn.l2_loss(v)
                                               for v in get_var_list()])

        loss = loss + l2_weight_penalty


    if debug:
        return loss, positive_loss, negative_loss
    else:
        return loss

def triplet_loss(main, positive, negative, labels, debug=False):

    with tf.name_scope('triplet_loss'):

        p_distance = compute_euclidean_distance(main, positive)
        n_distance = compute_euclidean_distance(main, negative)
        #out = K.exp(-K.sum(K.abs(x1 - x2), axis=1))

        positive_loss = tf.reduce_mean(tf.abs(tf.exp(-p_distance) - labels[:, 0]))
        negative_loss = tf.reduce_mean(tf.abs(tf.exp(-n_distance) - labels[:, 1]))

        loss = positive_loss + negative_loss

        # L2 Loss on variables.
    with tf.variable_scope('l2_weight_penalty'):
        l2_weight_penalty = 0.0001 * tf.add_n([tf.nn.l2_loss(v)
                                               for v in get_var_list()])

        loss = loss + l2_weight_penalty

    if debug:
        return loss, positive_loss, negative_loss
    else:
        return loss


def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    with tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()

    train_op = optimizer.minimize(loss, global_step=global_step, colocate_gradients_with_ops=True)

    return train_op

if __name__ == '__main__':

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    images = tf.placeholder(tf.float32, shape=(5*3, IMG_SIZE, IMG_SIZE, 3))
    labels = tf.placeholder(tf.float32, shape=(5, 2))

    main, pos, neg = triplet_model(images, 5)

    loss = KL_loss(main, pos, neg, labels)

    op = training(loss, 0.005)


