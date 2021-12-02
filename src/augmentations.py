from functools import partial

import tensorflow as tf


class Augmentation:
    def __init__(self, name, fn, params, apply_mask, prob) -> None:
        self.fn = partial(fn, **params) if params else fn
        self.prob = prob
        self.name = name
        self.apply_mask = apply_mask

    def __call__(self, *images):
        return tf.cond(
            tf.random.uniform([], 0, 1, tf.float32) > (1 - self.prob),
            lambda: tuple((self.fn(image) for image in images)),
            lambda: images
        )


class RandomBrigthness(Augmentation):
    def __init__(self, max_delta, prob) -> None:
        params = {
            'max_delta': max_delta
        }

        super().__init__('RandomBrigthness',
                         tf.image.random_brightness,
                         params, False, prob)


# TODO: add contrast, saturation, hue

class RandomCrop(Augmentation):
    def __init__(self, crop_shape) -> None:
        self.crop_shape = crop_shape

        super().__init__('RandomCrop',
                         tf.image.crop_to_bounding_box,
                         None, True, 1.0)

    def __call__(self, *images):
        shape = tf.shape(images[0])[:2]
        dh, dw = tf.unstack(shape - self.crop_shape + 1)

        h = tf.random.uniform(shape=[], maxval=dh, dtype=tf.int32)
        w = tf.random.uniform(shape=[], maxval=dw, dtype=tf.int32)

        crop_fn = partial(self.fn,
                          offset_height=h,
                          offset_width=w,
                          target_height=self.crop_shape[0],
                          target_width=self.crop_shape[1])

        images = [crop_fn(image) for image in images]

        return tuple(images)


class RandomLeftRightFlip(Augmentation):
    def __init__(self, prob) -> None:
        super().__init__('RandomLeftRightFlip',
                         tf.image.flip_left_right,
                         None, True, prob)
