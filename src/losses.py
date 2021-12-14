import numpy as np
import tensorflow as tf
from skimage.measure import regionprops
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
    n = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(n))[:sample_size]

    return tf.gather(x, indices)


def pad_bboxes(bboxes, pad, shape):
    """
    :param bboxes: tensor of shape [B, 4], where each
        bbox is defined as min_row, min_col, max_row, max_col
    :param pad: int, defines padding size
    :return: tensor of shape [B, 4], padded bounding boxes
    """

    h, w = shape

    min_row, min_col, max_row, max_col = tf.unstack(
        bboxes[:, None, :], axis=-1)

    bboxes_pad = tf.concat([
        tf.clip_by_value(min_row - pad, 0, h),
        tf.clip_by_value(min_col - pad, 0, w),
        tf.clip_by_value(max_row + pad, 0, h),
        tf.clip_by_value(max_col + pad, 0, w)
    ], axis=-1)

    return bboxes_pad


# def cut_bboxes(bboxes, max_area, shape):
#     def select_random_sub_bbox(bbox, max_are):
#         # Possible variants:
#         # Select by left side, by right, by top and by bottom

#         index = tf.random.uniform([], maxval=4)

#         # TODO: finish this part

#     bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 3] - bboxes[:, 1])

#     pass


def _subsample_mask(mask, max_area):
    mask_area = tf.reduce_sum(tf.cast(mask, tf.float32))

    if mask_area < max_area:
        return mask

    p_sample = max_area / mask_area

    mask_subsample = tf.random.uniform(tf.shape(mask)) <= p_sample

    return tf.logical_and(mask, mask_subsample)


def _build_mask_from_bbox(shape, bbox):
    h, w = shape

    def build_indices():
        min_row, min_col, max_row, max_col = tf.unstack(bbox, axis=-1)

        y_coords = tf.range(min_row, max_row, dtype=tf.int64)
        x_coords = tf.range(min_col, max_col, dtype=tf.int64)

        xx, yy = tf.meshgrid(x_coords, y_coords)
        indices = tf.concat(
            [flatten(yy)[:, None], flatten(xx)[:, None]], axis=-1)

        return indices

    indices = build_indices()
    mask = tf.SparseTensor(indices,
                           tf.ones(tf.shape(indices)[0], dtype=tf.bool),
                           [h, w])

    return tf.sparse.to_dense(mask)


def _generate_bbox_mask(shape, bboxes, bbox_pad,
                        max_bbox_area, num_bboxes=1):
    """
        returns: tf.tensor of shape [h, w, 1]
    """

    bboxes_sample = random_choice(bboxes, num_bboxes)
    bboxes_pad = pad_bboxes(bboxes_sample, bbox_pad, shape)

    mask = _build_mask_from_bbox(shape, bboxes_pad[0])
    mask = _subsample_mask(mask, max_bbox_area)

    return mask


def _py_compute_bboxes(segmentation):
    rprops = regionprops(segmentation)
    bboxes = [rprop.bbox for rprop in rprops]
    bboxes = np.array(bboxes, dtype=np.int32)

    return bboxes


@tf.function
def _tf_compute_bboxes(segmentation):
    bboxes = tf.numpy_function(
        _py_compute_bboxes,
        [tf.squeeze(segmentation)],
        tf.int32
    )

    return tf.reshape(bboxes, [-1, 4])


def contrastive_loss_with_bboxes(
    y_true,
    y_pred,
    bbox_pad=4,
    max_bbox_area=8000,
    num_bboxes_sample=1,
    normalize_preds=True,
    temperature=0.2
):
    """
        Current implementation supports only batch size = 1
        and one bounding box to sample
    """

    batch, h, w, _ = tf.unstack(tf.shape(y_true))

    # if batch > 1:
    #     print(f"batch size: {batch}")
    #     err = 'only batch size = 1 is supported for now'
    #     raise NotImplementedError(err)

    if num_bboxes_sample > 1:
        err = 'multiple bounding boxes support not implemented yet'
        raise NotImplementedError(err)

    if normalize_preds:
        y_pred = tf.nn.l2_normalize(y_pred, axis=-1)

    bboxes = _tf_compute_bboxes(y_true)
    mask = _generate_bbox_mask(
        (h, w),
        bboxes,
        bbox_pad,
        max_bbox_area,
        num_bboxes_sample
    )

    loss = _contrastive_loss(y_true[0], y_pred[0], mask, temperature)

    return loss


def _sample_indices(min_idx, max_idx, num_samples, replace=False):
    n_sample = tf.reduce_min([max_idx - min_idx + 1, num_samples])

    if not replace:
        indices = tf.range(min_idx, max_idx + 1, dtype=tf.int32)
        indices = tf.random.shuffle(indices)

        return indices[:n_sample]

    return tf.random.categorical(
        tf.ones([1, max_idx - min_idx + 1]),
        n_sample,
        dtype=tf.int32
    )[0] + min_idx


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


class ContrastiveLossWithBboxes(LossFunctionWrapper):
    def __init__(
        self,
        normalize_preds=True,
        temperature=0.2,
        bbox_pad=5,
        max_bbox_area=10000,
        num_bboxes_sample=1
    ):
        super().__init__(
            contrastive_loss_with_bboxes,
            reduction=tf.keras.losses.Reduction.NONE,
            name='ContrastiveLossWithBboxes',
            normalize_preds=normalize_preds,
            temperature=temperature,
            bbox_pad=bbox_pad,
            max_bbox_area=max_bbox_area,
            num_bboxes_sample=num_bboxes_sample
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


class ContrastiveLossPatchesContours(LossFunctionWrapper):
    def __init__(
        self,
        normalize_preds=True,
        temperature=0.2,
        window_size=85,
        num_cells_sample=10
    ):
        loss_patches = ContrastiveLossOnPatches(
            normalize_preds, temperature, window_size)
        loss_contours = ContrastiveLossWithContours(
            normalize_preds, temperature, num_cells_sample)

        def loss_fn(y_true, y_pred):
            return loss_patches(y_true, y_pred) + loss_contours(y_true, y_pred)

        super().__init__(
            loss_fn,
            reduction=tf.keras.losses.Reduction.NONE,
            name='ContrastiveLossPatchesContours'
        )
