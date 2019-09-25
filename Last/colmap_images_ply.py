import sys
import numpy as np
import argparse

from bluenotelib.common.quaternion import Quaternion
from colmapdeplib.sparse_model_mngr import Sparse_Model_Mngr


def process(model_file, output_file, name_filter):

    mm = Sparse_Model_Mngr(model_dir=model_file)
    images = mm.read_images(name_filter=name_filter)

    points = []
    for image_id in images:
        q = images[image_id].qvec
        t = images[image_id].tvec
        q_obj = Quaternion(qw=q[0], qx=q[1], qy=q[2], qz=q[3])
        rot_mat = q_obj.to_rotation_matrix(q_obj)
        rot_mat_inv = -1.0 * np.linalg.inv(rot_mat)
        cen = np.dot(rot_mat_inv, t)
        points.append(cen)


    ply_header = '''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    end_header
    '''.format(len(points))

    with open(output_file, 'w') as fp:
        fp.write(ply_header)
        for p in points:
            fp.write('{} {} {}\n'.format(p[0], p[1], p[2]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--model_file', required=True)
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('-f', '--name_filter', required=False)
    args = parser.parse_args()

    process(args.model_file, args.output_file, args.name_filter)
