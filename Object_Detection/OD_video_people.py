import os
import sys
import cv2
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import glob
import pickle

SIZE = 416
# Root directory of the project
ROOT_DIR = os.path.abspath("/home/weihao/GITHUB/Mask_RCNN-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
import shutil
import xml.etree.ElementTree as ET
import datetime


def get_values(c):
    for y in c:
        for x in y:
            if x.tag == 'xmin':
                xmin = int(x.text)
            if x.tag == 'xmax':
                xmax = int(x.text)
            if x.tag == 'ymin':
                ymin = int(x.text)
            if x.tag == 'ymax':
                ymax = int(x.text)
    return np.array([ymin, xmin, ymax, xmax])


def set_values(vs, c):
    for y in c:
        for x in y:
            if x.tag == 'xmin':
                x.text = '{}'.format(vs[1])
            if x.tag == 'xmax':
                x.text = '{}'.format(vs[3])
            if x.tag == 'ymin':
                x.text = '{}'.format(vs[0])
            if x.tag == 'ymax':
                x.text = '{}'.format(vs[2])


def recenter(shape, r):
    x = int((r[2]+r[0])/2)
    y = int((r[3]+r[1])/2)

    sz = SIZE

    w = r[2] - r[0]
    if w > sz:
        sz = w
    h = r[3] - r[1]
    if h > sz:
        sz = h

    if shape[0]<sz:
        sz = shape[0]
    if shape[1]<sz:
        sz = shape[1]

    sz = int(sz/2)

    if x<sz:
        x = sz
    if y<sz:
        y = sz
    if x>shape[0]-sz:
        x=shape[0]-sz
    if y>shape[1]-sz:
        y = shape[1]-sz

    return x-sz, y-sz, x+sz, y+sz


def inside(vs, region):

    if vs[2]>region[2]:
        return None
    if vs[3]>region[3]:
        return None

    vs[0] -= region[0]
    vs[1] -= region[1]
    vs[2] -= region[0]
    vs[3] -= region[1]

    if vs[0] < 0:
        return None
    if vs[1] < 0:
        return None

    return vs


def save_region(image, regions, scores, ids):

    print("Total detection", len(regions))
    for a in range(len(regions)):
        if (ids[a]) != 1:
            continue
        r0 = regions[a]
        r = r0 #recenter(image.shape, r0)draw
        w = r[2]-r[0]
        h = r[3]-r[1]
        #if h<100:
        #    continue
        area = (r[2]-r[0])*(r[3]-r[1])
        print('region {0:4d} {4:4d} {5:4d}, {1:4d} {2:4d}, {3:8d}'.format(a, w, h, area, r[0], r[1]))
        img = image[r[0]:r[2], r[1]:r[3], :]

        cv2.line(image, (r[1], r[0]), (r[3], r[0]), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(image, (r[3], r[0]), (r[3], r[2]), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(image, (r[3], r[2]), (r[1], r[2]), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(image, (r[1], r[2]), (r[1], r[0]), (0, 0, 255), 1, cv2.LINE_AA)


def get_anno(in_dir, basename):
    filename = os.path.join(in_dir, 'annotations', basename[:-3]+'xml')

    if not os.path.exists(filename):
        return None

    tree = ET.parse(filename)

    return tree

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def redo(m):
    msk = []
    regions = m['rois']
    for a in range(len(regions)):
        box = regions[a]
        mask = m['masks'][:, :, a]
        submask = mask[box[0]:box[2], box[1]:box[3]]
        msk.append(submask)
    m['masks'] = msk

    return m


if __name__ == '__main__':

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = InferenceConfig()
    config.display()


    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)


    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['person']

    filename = '/media/weihao/DISK0/flickr_images/BOLDERBoulder_10K_livestream_at_finish_line.mp4'
    if len(sys.argv)>1:
        filename = sys.argv[1]

    cap = cv2.VideoCapture(filename)

    out_folder = '/media/weihao/DISK0/flickr_images/testing_square'

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.mkdir(out_folder)
    ig_folder = os.path.join(out_folder, 'images')
    os.mkdir(ig_folder)
    skip = 23500
    nt = 0

    while nt<skip:
        hasFrame, frame = cap.read()
        nt += 1

    scale = 4
    md = {}
    objs = 0
    t0 = datetime.datetime.now()
    while True:
        hasFrame, frame = cap.read()
        print(nt, datetime.datetime.now()-t0)
        if not hasFrame:
            break

        # Run detection
        results = model.detect([frame], verbose=1)
        r = results[0]
        save_region(frame, r['rois'], r['scores'], r['class_ids'])
        objs += len(r['rois'])
        md[nt] = redo(r)
        cv2.imshow('t',frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        nt += 1
        if nt % 500 == 0:
            with open(filename[:-4] + '_{}.p'.format(nt), 'wb') as fp:
                pickle.dump(md, fp)
            md= {}


    print('total frames', nt)
    print('total object', objs)
    print('Processing time', datetime.datetime.now()-t0)
    with open(filename[:-4] + '_{}.p'.format(nt), 'wb') as fp:
        pickle.dump(md, fp)
