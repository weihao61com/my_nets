import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import glob


# Root directory of the project
ROOT_DIR = os.path.abspath("/media/weihao/DISK0/Object_detection/Mask_RCNN-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
import shutil


def save_region(image, regions, scores, ids, out_dir):
    print("Total detection", len(regions))
    for a in range(len(regions)):
        if (ids[a]) != 1:
            continue
        r = regions[a]
        area = (r[2]-r[0])*(r[3]-r[1])
        print('area {0:4d} {1:8d}'.format(a, area))
        img = image[r[0]:r[2], r[1]:r[3], :]
        file_name = '_{}_{}_{}.jpg'.format(a, ids[a], int(scores[a]*10000))
        skimage.io.imsave(out_dir + file_name, img)

# matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#filename = os.path.join(IMAGE_DIR, random.choice(file_names))

out_folder = '/home/weihao/Projects/tmp/images/'
if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
os.mkdir(out_folder)

#filename = '/home/weihao/Downloads/IMG_1134.JPG'

in_dir = '/home/weihao/Downloads/*'
for filename in glob.glob(in_dir):
    if filename[-3:] in ['jpg', 'JPG']:
        print('image', filename)
        image = skimage.io.imread(filename)

        # Run detection
        results = model.detect([image], verbose=1)
        basename = os.path.basename(filename)[:-4]

        # Visualize results
        r = results[0]
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                            class_names, r['scores'])

        save_region(image, r['rois'], r['scores'], r['class_ids'], out_folder + basename)

