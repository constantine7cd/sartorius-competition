import datetime
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from attrdict import AttrDict

sys.path.append('..')
sys.path.append('../Mask_RCNN')

from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config

from src.dataset import SartoriusDetectionDataset


class SartoriusConfig(Config):
    NAME = "sartorius"

    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2  # Including background

    STEPS_PER_EPOCH = 600
    VALIDATION_STEPS = 4 // IMAGES_PER_GPU

    IMAGE_RESIZE_MODE = "none"
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = np.array([0.5])  # TODO: change it

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 800


class SartoriusInferenceConfig(SartoriusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "none"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


def train(model, config, root_dir: Path):
    """Train the model."""
    # Training dataset.
    dataset_train = SartoriusDetectionDataset()
    dataset_train.load_sartorius_dataset(root_dir/'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SartoriusDetectionDataset()
    dataset_val.load_sartorius_dataset(root_dir/'val')
    dataset_val.prepare()

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='all')


def detect(model, results_dir: Path, root_dir: Path):
    """Run detection on images in the given directory."""

    # Create directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(results_dir, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = SartoriusDetectionDataset()
    dataset.load_sartorius_dataset(root_dir/'val')
    dataset.prepare()

    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]

        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir,
                    dataset.image_info[image_id]["id"]))


if __name__ == '__main__':
    train_config = AttrDict({
        'command': 'train',
        'logs': 'experiments/detection/logs/test_run_on_images',
        'weights': None,
        'dataset': Path('sartorius-cell-instance-segmentation/train')
    })

    cfg = train_config

    if cfg.command == "train":
        config = SartoriusConfig()
    else:
        config = SartoriusDetectionDataset()
    config.display()

    # Create model
    if cfg.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=cfg.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=cfg.logs)

    weights_path = cfg.weights

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if cfg.command == "train":
        train(model, config, cfg.dataset)
    elif cfg.command == "detect":
        detect(model, cfg.results_dir, cfg.dataset)
    else:
        print("'{}' is not recognized. "
            "Use 'train' or 'detect'".format(cfg.command))
