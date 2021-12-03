import math
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE


def _read_image_opencv(path):
    return cv2.imread(
        path.decode('UTF-8'),
        cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
    )


@tf.function
def _read_image(path, parse_shape=(520, 704, 1), dtype=tf.uint8):
    image = tf.numpy_function(_read_image_opencv, [path], dtype)
    image = tf.reshape(image, parse_shape)

    return image


@tf.function
def _resize_image(image, resize_shape=(512, 512)):
    return tf.image.resize(image, resize_shape, method='bilinear')


class SartoriusDataset:
    def __init__(
        self,
        root_dir: Path,
        batch_size: int = 16,
        parse_shape: tuple = (520, 704),
        resize_shape: tuple = None,
        resize_image_only: bool = False,
        augmentations: list = None,
        shuffle: bool = True,
        shuffle_buffer_size: int = int(1e5),
        random_state: int = None
    ) -> None:

        self.image_parse_shape = parse_shape + (1,)
        self.mask_parse_shape = parse_shape + (1,)

        self.resize_shape = resize_shape
        self.resize_image_only = resize_image_only
        self.augmentations = augmentations or []
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.seed = random_state

        img_paths = [str(p) for p in root_dir.glob('*.png')]
        mask_paths = [p.replace('.png', '_mask.tif') for p in img_paths]

        self.img_paths = img_paths
        self.mask_paths = mask_paths

        dataset = tuple(tf.data.Dataset.from_tensor_slices(p)
                        for p in [img_paths, mask_paths])
        dataset = tf.data.Dataset.zip(dataset)

        dataset = self.shuffle_dataset(dataset)
        dataset = self.parse_dataset(dataset)
        dataset = self.preprocess_dataset(dataset)
        dataset = self.batch_dataset(dataset)

        self.dataset = dataset
        self.n_steps = int(math.ceil(len(img_paths) / batch_size))

    def shuffle_dataset(self, dataset: tf.data.Dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        return dataset

    def parse_dataset(self, dataset: tf.data.Dataset):
        def parse(image_path, mask_path):
            image = _read_image(image_path, self.image_parse_shape)
            image = tf.cast(image, tf.float32) / 255.

            mask = _read_image(mask_path, self.mask_parse_shape, dtype=tf.uint16)
            mask = tf.cast(mask, tf.int32)

            return image, mask

        return dataset.map(parse, num_parallel_calls=AUTOTUNE)

    def preprocess_dataset(self, dataset: tf.data.Dataset):
        def preprocess(image, mask):
            for augmentation in self.augmentations:
                if augmentation.apply_mask:
                    image, mask = augmentation(image, mask)
                else:
                    image = augmentation(image)

            image = tf.clip_by_value(image, 0., 1.)

            if self.resize_shape is not None:
                image = _resize_image(image, self.resize_shape)

                if not self.resize_image_only:
                    mask = _resize_image(mask, self.resize_shape)

            return image, mask

        return dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

    def batch_dataset(self, dataset: tf.data.Dataset):
        return dataset.batch(self.batch_size)
