import cv2
import sqlite3
import numpy as np
import os
import sys

from db_view import get_rows, get_data
this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/..'.format(this_file_path))
from utils import PinholeCamera, Utils


def get_H(conn, table_name, id=7, fmt='d'):
    rows = get_rows(conn, table_name)
    Hs = {}
    for row in rows:
        if row[id] is not None:
            data1 = get_data(row[id], fmt)
            Hs[row[0]] = np.array(data1).reshape((3, 3))
    return Hs


if __name__ == "__main__":
    project_dir = '/home/weihao/Projects'
    key = 'heads'  # office" #heads
    mode = 'Test'
    db = '{}/colmap_features/{}_{}/proj.db'.format(project_dir, key, mode)

    conn = sqlite3.connect(db)
    Hs = get_H(conn, 'two_view_geometries')

    focal = 525.0
    cam = PinholeCamera(640.0, 480.0, focal, focal, 320.0, 240.0)

    nt = 0
    for idx in Hs:
        As, Rs, Ts, Ns = cv2.decomposeHomographyMat(Hs[idx], cam.mx)

        for a in range(As):
            print idx, Utils.rotationMatrixToEulerAngles(Rs[a])
        nt += 1
        if nt>10:
            break