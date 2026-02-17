import tensorflow as tf

def pairwise_rank_loss(y_true, y_pred):
    """
    Differentiable ranking loss
    Encourages correct ordering instead of value prediction
    Optimizes Information Coefficient directly
    """

    # reshape
    y_true = tf.reshape(y_true, (-1, 1))
    y_pred = tf.reshape(y_pred, (-1, 1))

    # pairwise differences
    diff_true = y_true - tf.transpose(y_true)
    diff_pred = y_pred - tf.transpose(y_pred)

    # sign matrix: +1 if i should rank higher than j
    sign = tf.sign(diff_true)

    # logistic ranking loss
    loss = tf.math.log(1 + tf.exp(-sign * diff_pred))

    # ignore equal returns
    mask = tf.not_equal(sign, 0)
    loss = tf.boolean_mask(loss, mask)

    return tf.reduce_mean(loss)

