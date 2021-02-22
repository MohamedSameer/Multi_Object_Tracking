import tensorflow as tf
#from Appearance.model.tf_customVGG16 import customVGG16 as network
from Appearance.model.tf_customVGG16 import customVGG16 as network

IMG_SIZE = 224


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask

def triplet_model(inputs=None, batch_size=None, train=False, reuse=None, feature_size=512):
    # Data should be B x H x W x C
    if batch_size is None:
        batch_size = int(inputs.get_shape().as_list()[0] / 3 )

    feature_vector = network(inputs=inputs, feature_size=feature_size, train=train, reuse=reuse)

    return feature_vector

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

def KL_loss(features, labels, debug=False):

    with tf.name_scope('KL_Divergence_loss'):

        # get positive and negative mask from labels
        pos_mask = _get_anchor_positive_triplet_mask(labels)
        neg_mask = _get_anchor_negative_triplet_mask(labels)

        # compute KL Divergence from all batch [batch, batch]
        # front
        feature_1 = tf.tile(features, [1, features.get_shape().as_list()[0]])
        feature_1 = tf.reshape(feature_1, [features.get_shape().as_list()[0] * features.get_shape().as_list()[0], -1])

        # rear
        feature_2 = tf.broadcast_to(features, [features.get_shape().as_list()[0], features.get_shape().as_list()[0], features.get_shape().as_list()[1]])
        feature_2 = tf.reshape(feature_2, [features.get_shape().as_list()[0] * features.get_shape().as_list()[0], -1])

        # KL-Divergence
        kl_score = tf.math.multiply(feature_1, tf.math.log(feature_1 / feature_2))

        # ignore NaN
        mask = tf.constant([0.0], dtype=tf.float32, shape=[kl_score.get_shape().as_list()[0], kl_score.get_shape().as_list()[1]])
        kl_score = tf.where(tf.is_nan(kl_score), mask, kl_score)
        kl_score = tf.reduce_sum(kl_score, axis=1)
        kl_score = tf.reshape(kl_score, [features.get_shape().as_list()[0], features.get_shape().as_list()[0]])

        # matrix
        positive_kl_score_matrix = kl_score * tf.to_float(pos_mask)
        negative_kl_score_matrix = kl_score * tf.to_float(neg_mask)
        #negative_kl_score_matrix = tf.abs(1.0 - kl_score * tf.to_float(neg_mask))

        positive_loss = tf.reduce_sum(positive_kl_score_matrix) / tf.reduce_sum(tf.to_float(pos_mask))
        negative_loss = tf.reduce_sum(tf.maximum((1.0 - negative_kl_score_matrix) * tf.to_float(neg_mask), 0.0)) / tf.reduce_sum(tf.to_float(neg_mask))
        #negative_loss = tf.reduce_sum(tf.maximum(negative_kl_score_matrix, 0.0))  #

        loss = positive_loss + negative_loss

        # L2 Loss on variables.
    with tf.variable_scope('l2_weight_penalty'):
        l2_weight_penalty = 0.00001 * tf.add_n([tf.nn.l2_loss(v)
                                               for v in get_var_list()])

        loss = loss + l2_weight_penalty


    if debug:
        return loss, positive_loss, negative_loss, kl_score, pos_mask, neg_mask, positive_kl_score_matrix, negative_kl_score_matrix
    else:
        return loss, positive_loss, negative_loss


def training(loss, learning_rate, var_list=None):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    with tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()

    if var_list is None:
        var_list = tf.trainable_variables()

    train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list, colocate_gradients_with_ops=True)

    return train_op

if __name__ == '__main__':

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    images = tf.placeholder(tf.float32, shape=(5, IMG_SIZE, IMG_SIZE, 3))
    labels = tf.placeholder(tf.float32, shape=(5))

    features = triplet_model(images, 5, train=True)

    loss = KL_loss(features, labels)

    op = training(loss, 0.005)


