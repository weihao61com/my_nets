import sqlite3
import struct
from sortedcontainers import SortedDict
import numpy as np
import math
import cv2
import os
import sys
import pickle
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("last2")
logger.setLevel(logging.INFO)

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/..'.format(this_file_path))
sys.path.append('{}/../../../GITHUB/colmap/scripts/python'.format(this_file_path))

from utils import Utils, PinholeCamera, HOME
# from db import COLMAPDatabase
# from colmapdeplib.sparse_model_mngr import Sparse_Model_Mngr

from colmap_script.db import Colmap_DB
from read_model import read_model
from image_pairing.pose_ana import load_truth


def get_id(filename):
    strs = filename.split('.')[0]
    strs = strs.split('-')[1]
    return int(strs)


if __name__ == '__main__':

    key = 'heads'  # office" #heads
    mode = 'Test'
    poses_dic, cam, key = load_truth(key, mode)
    poses_by_name = {}
    for id in poses_dic.values()[0]:
        pose = poses_dic.values()[0][id]
        poses_by_name[os.path.basename(pose.filename)] = pose

    project_dir = '{}/tmp/{}_{}'.format(HOME, key, mode)
    print(project_dir)

    db_file = '{}/proj.db'.format(project_dir)

    output_file = '{}/pairs.p'.format(project_dir)
    model_file = '{}/sparse/align_0'.format(project_dir)

    logger.info("Model {}".format(model_file))
    cameras, images, p3D = read_model(path=model_file, ext='.bin')

    logger.info("DB file {}".format(db_file))
    db = Colmap_DB(db_file)
    #db.get_image_match(0,2000)
    #db = Colmap_DB(db_file)
    db.get_image_list()
    db.get_image_feature()
    db.get_image_match(0, 2000)

    output = []
    total_match = 0
    for imgs in db.matches:
        matches = db.matches[imgs]
        read_matches = []
        img0 = db.imagelist[imgs[0]]
        img1 = db.imagelist[imgs[1]]
        img0_md = images[imgs[0]]
        img1_md = images[imgs[1]]
        tr0 = poses_by_name[img0.name]
        tr1 = poses_by_name[img1.name]
        nf = 0
        nt = 0
        for m in matches:
            if img0_md.point3D_ids[m[0]] > -1:
                if img1_md.point3D_ids[m[1]] > -1:
                    id0 = img0_md.point3D_ids[m[0]]
                    id1 = img1_md.point3D_ids[m[1]]
                    if id0 != id1:
                        # print('What', id0, id1)
                        nf += 1
                    else:
                        p3 = p3D[id0].xyz
                        k0 = img0.key_points[m[0]]
                        k1 = img1.key_points[m[1]]
                        mh = (p3, [k0.x, k0.y], [k1.x, k1.y], m[0], m[1])
                        read_matches.append(mh)
                        nt += 1
        output.append((tr0, tr1, read_matches))
        total_match += len(read_matches)
        #if len(output)>=500:
        #    break

    print(len(output), total_match/len(output))
    with open(output_file, 'wb') as fp:
        pickle.dump(output, fp)
