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
        mask = tf.ones([b, sp_size], dtype=tf.bool)

    loss = tf.map_fn(
        wrapper,
        (y_true, y_pred, mask),
        fn_output_signature=tf.float32
    )

    return tf.reduce_mean(loss)


def flatten(x):
    return tf.reshape(x, [-1])


def _generate_random_patch_mask(shape, w_size):
    b, h, w = shape

    x_coord = tf.random.uniform((b, 1), 0, w - w_size, dtype=tf.int64)
    x_coord = x_coord + tf.range(w_size, dtype=tf.int64)[None, :]

    y_coord = tf.random.uniform((b, 1), 0, h - w_size, dtype=tf.int64)
    y_coord = y_coord + tf.range(w_size, dtype=tf.int64)[None, :]

    def build_indices(x_coord, y_coord):
        xx, yy = tf.meshgrid(x_coord, y_coord)
        indices = tf.concat(
            [flatten(yy)[:, None], flatten(xx)[:, None]], axis=-1)

        return indices

    def compute_mask(inputs):
        x_coord, y_coord = inputs

        indices = build_indices(x_coord, y_coord)
        mask = tf.SparseTensor(indices,
                               tf.ones(indices.shape[0], dtype=tf.bool),
                               [h, w])

        return tf.sparse.to_dense(mask)

    mask = tf.map_fn(
        compute_mask,
        (x_coord, y_coord),
        fn_output_signature=tf.bool
    )

    return mask


def contrastive_loss_on_patches(y_true, y_pred,
                                normalize_preds=True,
                                temperature=0.2,
                                window_size=80):
    def wrapper(inputs):
        y_true, y_pred, mask = inputs

        return _contrastive_loss(y_true, y_pred, mask, temperature)

    if normalize_preds:
        y_pred = tf.nn.l2_normalize(y_pred, axis=-1)

    b, h, w, _ = tf.unstack(tf.shape(y_pred, out_type=tf.int64))

    if window_size is not None:
        mask = _generate_random_patch_mask([b, h, w], window_size)
    else:
        mask = tf.ones([b, h, w], dtype=tf.bool)

    loss = tf.map_fn(
        wrapper,
        (y_true, y_pred, mask),
        fn_output_signature=tf.float32
    )

    return tf.reduce_mean(loss)


def random_choice(x, sample_size):
    # TODO: test it
    n = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(n))[:sample_size]

    return tf.gather(x, indices)


def pad_bboxes(bboxes, pad):
    """
    :param bboxes: tensor of shape [B, 4], where each
        bbox is defined as min_row, min_col, max_row, max_col
    :param pad: int, defines padding size
    :return: tensor of shape [B, 4], padded bounding boxes
    """

    min_row, min_col, max_row, max_col = tf.unstack(bboxes, axis=-1)

    bboxes_pad = tf.stack([
        min_row - pad,
        min_col - pad,
        max_row + pad,
        max_col + pad
    ], axis=-1)

    return bboxes_pad


def cut_bboxes(bboxes, max_area, shape):
    def select_random_sub_bbox(bbox, max_are):
        # Possible variants:
        # Select by left side, by right, by top and by bottom

        index = tf.random.uniform([], maxval=4)

        # TODO: finish this part

    bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 3] - bboxes[:, 1])

    pass


def build_mask_from_bboxes(shape, bboxes):
    h, w = shape

    def build_indices(x_coord, y_coord):
        pass

    indices = build_indices()
    mask = tf.SparseTensor(indices,
                           tf.ones(indices.shape[0], dtype=tf.bool),
                           [h, w])

    return tf.sparse.to_dense(mask)


def _generate_bboxes_mask(shape, bboxes, bbox_pad,
                          max_bbox_area, num_bboxes):

    bboxes_sample = random_choice(bboxes, num_bboxes)
    bboxes_pad = pad_bboxes(bboxes_sample, bbox_pad)
    bboxes_cut = cut_bboxes(bboxes_pad, max_bbox_area, shape)
    mask = build_mask_from_bboxes(shape, bboxes_cut)

    return mask


def contrastive_loss_with_bboxes(
    y_true,
    y_pred,
    bboxes,
    bbox_pad=4,
    max_bbox_area=8000,
    num_bboxes_sample=8,
    normalize_preds=True,
    temperature=0.2,
):
    pass


def _sample_indices(min_idx, max_idx, num_samples, replace=False):
    if not replace:
        indices = tf.range(min_idx, max_idx, dtype=tf.int32)
        indices = tf.random.shuffle(indices)

        return indices[:num_samples]

    return tf.random.categorical(
        tf.ones([1, max_idx - min_idx + 1]),
        num_samples,
        dtype=tf.int32
    )[0]


def _tf_morphs(func, x, kernel):
    inputs = tf.cast(x[None], tf.int32)

    return func(
        inputs,
        kernel,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',
        dilations=[1, 1, 1, 1])


def tf_dilate(x, kernel=tf.ones([7, 7, 1], dtype=tf.int32)):
    return _tf_morphs(tf.nn.dilation2d, x, kernel) - 1


def tf_erode(x, kernel=tf.ones([7, 7, 1], dtype=tf.int32)):
    return _tf_morphs(tf.nn.erosion2d, x, kernel) + 1


def _contour_mask(x, kernel=tf.ones([7, 7, 1], dtype=tf.int32)):
    dilated = tf.cast(tf_dilate(x, kernel), tf.bool)
    eroded = tf.cast(tf_erode(x, kernel), tf.bool)

    return tf.logical_and(dilated, tf.logical_not(eroded))[0]


def _build_masks(y_true, cell_ids):
    def build_m(cell_id):
        return _contour_mask(y_true == cell_id)

    return tf.map_fn(build_m, cell_ids, fn_output_signature=tf.bool)


def _contrastive_loss_with_contours(y_true, y_pred, num_cells_sample, temperature):
    def wrapper(mask):
        return _contrastive_loss(y_true, y_pred, mask, temperature)

    cell_ids = _sample_indices(1, tf.reduce_max(y_true), num_cells_sample)
    masks = _build_masks(y_true, cell_ids)

    return tf.map_fn(wrapper, masks[..., 0], fn_output_signature=tf.float32)


def contrastive_loss_with_contours(
    y_true,
    y_pred,
    num_cells_sample=10,
    normalize_preds=True,
    temperature=0.2,
):
    def wrapper(inputs):
        y_true, y_pred = inputs

        return _contrastive_loss_with_contours(y_true, y_pred,
                                               num_cells_sample, temperature)

    if normalize_preds:
        y_pred = tf.nn.l2_normalize(y_pred, axis=-1)

    loss = tf.map_fn(
        wrapper,
        (y_true, y_pred),
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


class ContrastiveLossOnPatches(LossFunctionWrapper):
    def __init__(
        self,
        normalize_preds=True,
        temperature=0.2,
        window_size=85
    ):
        super().__init__(
            contrastive_loss_on_patches,
            reduction=tf.keras.losses.Reduction.NONE,
            name='ContrastiveLossOnPatches',
            normalize_preds=normalize_preds,
            temperature=temperature,
            window_size=window_size
        )


class ContrastiveLossWithContours(LossFunctionWrapper):
    def __init__(
        self,
        normalize_preds=True,
        temperature=0.2,
        num_cells_sample=10
    ):
        super().__init__(
            contrastive_loss_with_contours,
            reduction=tf.keras.losses.Reduction.NONE,
            name='ContrastiveLossOnPatches',
            normalize_preds=normalize_preds,
            temperature=temperature,
            num_cells_sample=num_cells_sample
        )
