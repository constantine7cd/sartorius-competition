from functools import partial

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


def _contrastive_loss(y_true, y_pred, mask, temperature):
    if tf.reduce_sum(tf.cast(mask, tf.float32)) == 0:
        return 0.

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    scores = tf.matmul(y_pred, y_pred, transpose_b=True)
    scores = scores / temperature

    logits_max = tf.reduce_max(scores, axis=1, keepdims=True)
    logits = scores - logits_max

    logits_exp = tf.exp(logits)
    logits_exp_sum = tf.reduce_sum(logits_exp, axis=1, keepdims=True)

    log_probs = logits - tf.math.log(tf.maximum(logits_exp_sum, 1e-8))

    adj = tf.math.equal(y_true, tf.transpose(y_true))
    adj = tf.cast(adj, tf.float32)

    loss = log_probs * adj
    loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(adj, axis=1)

    loss = -tf.reduce_mean(loss)

    return loss


def _generate_sample_mask(shape, sample_size):
    prob = tf.cast(sample_size / shape[1], tf.float32)

    return tf.random.uniform(shape) <= prob


def contrastive_loss(y_true, y_pred,
                     normalize_preds=True, temperature=0.2, sample_size=5000):
    def wrapper(inputs):
        y_true, y_pred, mask = inputs

        return _contrastive_loss(y_true, y_pred, mask, temperature)

    if normalize_preds:
        y_pred = tf.nn.l2_normalize(y_pred, axis=-1)

    b, h, w, dim = tf.unstack(tf.shape(y_pred))
    sp_size = h * w

    y_true = tf.reshape(y_true, [b, sp_size, 1])
    y_pred = tf.reshape(y_pred, [b, sp_size, dim])

    if sample_size is not None:
        mask = _generate_sample_mask([b, sp_size], sample_size)
    else:
        mask = tf.ones([b, sp_size], dtype=tf.float32)

    loss = tf.map_fn(
        wrapper,
        (y_true, y_pred, mask),
        fn_output_signature=tf.float32
    )

    return tf.reduce_mean(loss)


class ContrastiveLoss(LossFunctionWrapper):
    def __init__(
        self,
        normalize_preds=True,
        temperature=0.2,
        sample_size=5000
    ):
        super().__init__(
            contrastive_loss,
            reduction=tf.keras.losses.Reduction.NONE,
            name='ContrastiveLoss',
            normalize_preds=normalize_preds,
            temperature=temperature,
            sample_size=sample_size
        )
