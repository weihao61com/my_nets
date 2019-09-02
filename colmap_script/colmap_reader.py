import sys
import numpy as np
import argparse

from bluenotelib.common.quaternion import Quaternion
from colmapdeplib.sparse_model_mngr import Sparse_Model_Mngr


def process(model_file):

    mm = Sparse_Model_Mngr(model_dir=model_file)
    images = mm.read_images()
    p3D = mm.read_points3d()

    print(len(p3D))
    for p in p3D:
        point = p3D[p]
        img_ids = point.image_ids
        pnt_ids = point.point2d_idxs
        xyz = point.xyz
        for img_id in img_ids:
            img = images[img_id]
            q = img.qvec

    points = []
    for image_id in images:
        image_name = images[image_id].name
        q = images[image_id].qvec
        t = images[image_id].tvec
        q_obj = Quaternion(qw=q[0], qx=q[1], qy=q[2], qz=q[3])
        rot_mat = q_obj.to_rotation_matrix(q_obj)
        rot_mat_inv = -1.0 * np.linalg.inv(rot_mat)
        cen = np.dot(rot_mat_inv, t)
        points.append(cen)



if __name__ == '__main__':

    model_file = '/Users/weihao/Projects/tmp/heads_Train/sparse/align_0'

    process(model_file)
