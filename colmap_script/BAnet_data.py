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
from utils import Utils, PinholeCamera, HOME
from db import Colmap_DB
# from colmapdeplib.sparse_model_mngr import Sparse_Model_Mngr
from database import COLMAPDatabase
# from . import colmap_reader
from colmap_ana import read_model
from image_pairing.pose_ana import load_truth


def get_id(filename):
    strs = filename.split('.')[0]
    strs = strs.split('-')[1]
    return int(strs)


if __name__ == '__main__':

    key = 'heads'  # office" #heads
    key = '00'  # office" #heads
    mode = 'Train'
    # mode = 'Test'
    poses_dic, cam, key = load_truth(key, mode)
    poses_by_name = {}
    A = list(poses_dic.values())
    id_list = A[0]
    for id in id_list:
        pose = id_list[id]
        poses_by_name[os.path.basename(pose.filename)] = pose

    project_dir = '{}/tmp/{}_{}'.format(HOME, key, mode)
    print(project_dir)

    db_file = '{}/proj.db'.format(project_dir)
    output_file = '{}/pairs_2.p'.format(project_dir)
    model_file = '{}/sparse/align_0'.format(project_dir)

    logger.info("Model {}".format(model_file))
    cameras, images, points3D = read_model(path=model_file, ext='bin')

    logger.info("DB file {}".format(db_file))
    db = Colmap_DB(db_file)
    db.get_image_list()
    db.get_image_feature()
    db.get_image_match(0, 2000)

    # mm = Sparse_Model_Mngr(model_dir=model_file)
    # images = mm.read_images()

    # p3D = mm.read_points3d()

    output_images = {}
    for id in images:
        img = images[id]
        output_images[id-1] = (img.qvec, img.tvec)

    output_poses = {}
    output_matches = {}
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
                        p3 = points3D[id0].xyz
                        k0 = img0.key_points[m[0]]
                        k1 = img1.key_points[m[1]]
                        mh = (p3, [k0.x, k0.y], [k1.x, k1.y], m)
                        read_matches.append(mh)
                        nt += 1
        # output.append((tr0, tr1, read_matches))
        if tr0.id not in output_poses:
            output_poses[tr0.id] = tr0
        if tr1.id not in output_poses:
            output_poses[tr1.id] = tr1
        output_matches[(tr0.id, tr1.id)] = read_matches
        total_match += len(read_matches)
        #if len(output)>=500:
        #    break

    print("num_image, num_image_match, num_point_match/image_pair",
          len(output_images), len(output_poses), len(output_matches), total_match/len(output_matches))
    print("Out file", output_file)
    with open(output_file, 'wb') as fp:
        pickle.dump((output_images, output_matches, output_poses), fp)
