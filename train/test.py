import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from config import LaneConfig
# from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DEFAULT_DATASET_DIR = os.path.join(ROOT_DIR, "..", "ICME2022_Training_Dataset")
DATASET_IMAGES_SUBSET = {
    "train": "images",
    "val": "images_real_world"
}
DATASET_LABELS_SUBSET = {
    "train": os.path.join("labels", "class_labels"),
    "val": "labels_real_world"
}


if __name__ == "__main__":
    dataset_dir = DEFAULT_DATASET_DIR
    subset = "train"
    dataset_labels_dir = os.path.join(dataset_dir, DATASET_LABELS_SUBSET[subset])
    # dataset_labels_dir = os.path.join(dataset_dir, DATASET_LABELS_SUBSET[subset])
    # print(os.listdir(dataset_labels_dir))
    np.set_printoptions(threshold=np.inf)

    for file in os.listdir(dataset_labels_dir):
        if not os.path.isfile(os.path.join(dataset_labels_dir, file)):
            continue
        
        (name, ext) = os.path.splitext(file)
            
        image_path = os.path.join(dataset_labels_dir, file)
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        if len(image.shape) == 3:
            image = image[..., 0]

        mask = np.zeros((image.shape[0], image.shape[1], LaneConfig.NUM_CLASSES), dtype=np.uint8)

        for (ridx, row) in enumerate(image):
            for (idx, label) in enumerate(row):
                mask[ridx, idx, label] = 1
        