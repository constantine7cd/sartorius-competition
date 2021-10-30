import tensorflow as tf
from pathlib import Path


class SartoriousDataset:
    def __init__(
        self, 
        root_dir: Path, 
        batch_size: int = 16, 
        crop_size: tuple = (384, 384), 
        augmentations: list = None, 
        random_state: int = None
    ) -> None:
        
        img_paths = [str(p) for p in root_dir.glob('.png')]
        mask_paths = [p.replace('.png', '_mask.tif') for p in img_paths]

        print(img_paths[:3])
        print(mask_paths[:3])

        dataset = tuple(tf.data.Dataset.from_tensor_slices(p) 
                        for p in [img_paths, mask_paths])
        dataset = tf.data.Dataset.zip(dataset)
    
        self.dataset = dataset


dataset = SartoriousDataset(root_dir='sartorius-cell-instance-segmentation/train')