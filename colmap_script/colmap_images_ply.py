import sys
import numpy as np
import argparse
import evo.core.transformations as etr

from colmap_ana import read_model
from utils import Utils


def process(model_file, output_file, rgb_file):
    cameras, images, points3D = read_model(path=model_file, ext='bin')

    points = []
    for image_id in images:
        q = images[image_id].qvec
        t = images[image_id].tvec
        q_obj = etr.quaternion_matrix(q)

        rot_mat = q_obj[:3, :3]
        # q_obj[:3, 3] = t
        # q_obj_inv =

        rot_mat_inv = -1.0 * np.linalg.inv(rot_mat)
        cen = np.dot(rot_mat_inv, t)
        points.append(cen)

    Utils.create_ply(points, output_file)

    cloud = []
    rgb = []
    for id in points3D:
        p = points3D[id]
        cloud.append(p.xyz)
        rgb.append(p.rgb)

    Utils.create_ply(cloud, rgb_file, rgb)

if __name__ == '__main__':

    model_file = '/home/weihao/Projects/tmp/kitti_00_Train/sparse/align_0'
    output_file = model_file + '/images.ply'
    rgb_file = model_file + '/sparse.ply'

    process(model_file, output_file, rgb_file)
