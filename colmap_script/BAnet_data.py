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
import datetime

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
    key = '02'  # office" #heads
    mode = 'Test'
    # mode = 'Test'
    poses_dic, cam, key = load_truth(key, mode)

    project_dir = '{}/tmp/{}_{}'.format(HOME, key, mode)
    print(project_dir)

    db_file = '{}/proj.db'.format(project_dir)
    output_file = '{}/pairs_2.p'.format(project_dir)
    model_file = '{}/sparse/align_0'.format(project_dir)

    print("Model {}".format(model_file))
    cameras, images, points3D = read_model(path=model_file, ext='bin')

    print("DB file {}".format(db_file))
    db = Colmap_DB(db_file)
    db.get_image_list()
    db.get_image_feature()
    db.get_image_match(0, 2000)

    name_id = {}
    for id in images:
        img = images[id]
        name_id[img.name] = id

    # fix ID
    poses_by_id = {}
    A = list(poses_dic.values())
    id_list = A[0]
    for id in id_list:
        pose = id_list[id]
        name = os.path.basename(pose.filename)
        if name in name_id:
            pose.id = name_id[name]
            poses_by_id[name_id[name]] = pose

    output_images = {}
    images_by_id = {}
    for id in images:
        img = images[id]
        images_by_id[id] = img
        output_images[id] = (img.qvec, img.tvec)

    output_poses = {}
    output_matches = {}
    total_match = 0
    T0 = datetime.datetime.now()
    nm = 0
    for imgs in db.matches:
        matches = db.matches[imgs]
        read_matches = []
        id0 = imgs[0]
        id1 = imgs[1]
        img0 = db.imagelist[id0]
        img1 = db.imagelist[id1]
        img0_md = images_by_id[id0]
        img1_md = images_by_id[id1]
        tr0 = poses_by_id[id0]
        tr1 = poses_by_id[id1]
        nf = 0
        nt = 0

        for m in matches:
            if img0_md.point3D_ids[m[0]] > -1:
                if img1_md.point3D_ids[m[1]] > -1:
                    id30 = img0_md.point3D_ids[m[0]]
                    id31 = img1_md.point3D_ids[m[1]]
                    if id30 != id31:
                        # print('What', id0, id1)
                        nf += 1
                    else:
                        p3 = points3D[id30].xyz
                        k0 = img0.key_points[m[0]]
                        k1 = img1.key_points[m[1]]

                        mh = (p3, [k0.x, k0.y], [k1.x, k1.y], m)
                        if id0 == 1 and m[0] < 100:
                            print(imgs,  mh)
                        if id1 == 1 and m[1] < 100:
                            print(imgs,  mh)

                        read_matches.append(mh)
                        nt += 1
        nm += 1
        #if nm%10==0:
        #    print("{} {}/{}".format(datetime.datetime.now()-T0, nm, len( db.matches)))

        if id0 not in output_poses:
            output_poses[id0] = tr0
        if id1 not in output_poses:
            output_poses[id1] = tr1
        output_matches[(id0, id1)] = read_matches
        total_match += len(read_matches)
        #if len(output)>=500:
        #    break

    print("num_image, num_image_match, num_point_match/image_pair",
          len(output_images), len(output_poses), len(output_matches), total_match/len(output_matches))
    print("Out file", output_file)
    with open(output_file, 'wb') as fp:
        pickle.dump((output_images, output_matches, output_poses), fp)
