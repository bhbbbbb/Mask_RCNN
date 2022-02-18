"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from config import LaneConfig
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DEFAULT_DATASET_DIR = os.path.join(ROOT_DIR, "..", "ICME2022_Training_Dataset")
DATASET_IMAGES_SUBSET = {
    "train": "images",
    "train_test": "images_test",
    "val": "images_real_world"
}
DATASET_LABELS_SUBSET = {
    "train": os.path.join("labels", "class_labels"),
    "train_test": os.path.join("labels_test", "class_labels"),
    "val": "labels_real_world"
}
LABEL_POSTFIX = {
    "train": "_lane_line_label_id.png",
    "val": ".png",
}


import time

############################################################
#  Dataset
############################################################

class LaneDataset(utils.Dataset):

    def load_lane(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("lane", 1, "main_lane")
        self.add_class("lane", 2, "alter_lane")
        self.add_class("lane", 3, "double_line")
        self.add_class("lane", 4, "dashed_line")
        self.add_class("lane", 5, "single_line")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_labels_dir = os.path.join(dataset_dir, DATASET_LABELS_SUBSET[subset])
        dataset_images_dir = os.path.join(dataset_dir, DATASET_IMAGES_SUBSET[subset])
        postfix = LABEL_POSTFIX[subset]

        # imread_time_sum = 0
        # set_labels_time_sum = 0

        files = os.listdir(dataset_images_dir)
        for (fidx, file) in enumerate(files):
            if not os.path.isfile(os.path.join(dataset_images_dir, file)):
                continue
            
            (name, ext) = os.path.splitext(file)
                
            image_path = os.path.join(dataset_images_dir, file)
            label_path = os.path.join(dataset_labels_dir, name + postfix)
            for i in range(1, 100):
                try:
                    # imread_time = time.monotonic_ns()
                    label = skimage.io.imread(label_path)
                    # imread_time = time.monotonic_ns() - imread_time
                    break
                except MemoryError as err:
                    sleep_time = i ** 2
                    print(f"{err} occurs at fidx={fidx} for file: {name}, try again after {sleep_time}s")
                    time.sleep(sleep_time)
                    if i == 99: raise "give up"
            
            height, width = label.shape[:2]

            mask = np.zeros([label.shape[0], label.shape[1], 0], dtype=bool)

            # set_labels_time = time.monotonic_ns()
            if len(label.shape) == 3: label = label[..., 0]
            for i in range(1, LaneConfig.NUM_CLASSES):
                filiter = label == i
                filiter = filiter[..., np.newaxis]
                mask = np.concatenate((mask, filiter), axis=2)

            # for (ridx, row) in enumerate(label):
            #     for (idx, class_label) in enumerate(row):
            #         if len(label.shape) == 3:
            #             mask[ridx, idx, class_label[0]] = 1
            #         else:
            #             mask[ridx, idx, class_label] = 1
            # set_labels_time = time.monotonic_ns() - set_labels_time

            self.add_image(
                "lane",
                image_id=name,  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                mask=mask)
            
            # imread_time_sum = imread_time_sum + imread_time
            # set_labels_time_sum = set_labels_time_sum + set_labels_time
            # if fidx % 5 == 0:
            #     print("------------------------------------------")
            #     print(f"current spent time: imread: {imread_time}, set: {set_labels_time}")
            #     times = fidx + 1
            #     print(f"average spent time: {imread_time_sum / times}, {set_labels_time_sum / times}")
            #     print("------------------------------------------")


            if fidx % 100 == 0:
                print("{}/{}({} %) {} is added.".format(fidx, len(files), 100 * fidx / len(files), name))


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]

        assert image_info["source"] == "lane"
        # if image_info["source"] != "lane":
        #     return super(self.__class__, self).load_mask(image_id)

        mask = image_info["mask"]
        image_info["mask"] = None

        # Return mask, and array of class IDs of each instance. Since we have
        return mask.astype(bool), [i for i in range(1, LaneConfig.NUM_CLASSES)]

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lane":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LaneDataset()
    dataset_train.load_lane(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LaneDataset()
    dataset_val.load_lane(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        default=DEFAULT_DATASET_DIR,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LaneConfig()
    else:
        class InferenceConfig(LaneConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
