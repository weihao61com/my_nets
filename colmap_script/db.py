import sqlite3
import struct
from sortedcontainers import SortedDict
import numpy as np
import math


def camera_params(data):
    return struct.unpack('dddd', data)


def get_rows(conn, table_name, verbose = True):
    c = conn.cursor()
    cameras = c.execute('select * from {}'.format(table_name))
    c_names = list(map(lambda x: x[0], cameras.description))

    rows = c.fetchall()
    if verbose:
        print '\n\ttable', table_name
        for name in c_names:
            print '\t', name
        print '\tdata length', len(rows)
    return rows


def view_cameras_table(conn):
    rows = get_rows(conn, 'cameras')

    nt = 0
    for row in rows:
        print '\t', row
        data1 = camera_params(row[4])
        print '\tparams', data1
        nt += 1
        if nt>5:
            break

def get_data(data, fmt):
    length = 4
    if fmt=='d':
        length = 8
    if fmt=='B':
        length = 1
    ft = "<{}{}".format(len(data)/length, fmt)
    d = struct.unpack(ft, data)
    return d

def view_data_table(conn, table_name, id=None, fmt=None):
    rows = get_rows(conn, table_name)
    if id is not None:
        nt = 0
        for row in rows:
            print('\t{}'.format(row))
            if fmt is not None:
                data1 = get_data(row[id], fmt)
                print('\t{}'.format(data1[:12]))
            nt += 1
            if nt>5:
                break


class KeyPoint:

    def __init__(self, p):
        self.x = p[0]
        self.y = p[1]
        self.angle = math.atan2(p[3], p[2])
        self.r = np.sqrt(p[2]*p[2]+p[3]*p[3])
        self.descriptor = None

    def add_descriptor(self, p):
        self.descriptor = p

def image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return int(image_id1-1), int(image_id2-1)

class ImageFeature:

    def __init__(self, imagename):
        self.name = imagename
        self.key_points = SortedDict()

    def add_key_point(self, p):
        data = get_data(p[3], 'f')
        length = int(p[1])
        width = int(p[2])
        kps = np.array(data).reshape((length, width))

        id = 0
        for kp in kps:
            self.key_points[id] = KeyPoint(kp)
            id += 1
            #print kp[0], kp[1], kp[2], kp[3], kp[4], kp[5]
        #print ('\n')
        #print len(data)

    def add_descriptor(self, p):
        data = get_data(p[3], 'B')
        length = int(p[1])
        width = int(p[2])
        decs = np.array(data).reshape((length, width))
        id = 0
        for dec in decs:
            self.key_points[id].add_descriptor(dec)
            id += 1

class Colmap_DB:

    def __init__(self, db_name, verbose=False):
        self.name = db_name
        self.imagelist = SortedDict()

        if verbose:
            conn = sqlite3.connect(db_name)
            res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            for name in res:
                print name[0]

            view_data_table(conn, 'cameras', 4, 'd')
            view_data_table(conn, 'sqlite_sequence')
            view_data_table(conn, 'images',3, None)
            view_data_table(conn, 'keypoints', 3, 'd')
            view_data_table(conn, 'descriptors', 3, 'I')
            view_data_table(conn, 'matches',1)
            view_data_table(conn, 'inlier_matches',1)

    def get_image_match(self):
        conn = sqlite3.connect(self.name)
        rows = get_rows(conn, 'matches', False)
        print 'Total match', len(rows)
        for row in rows:
            print image_ids(long(row[0]))

    def get_image_list(self):
        conn = sqlite3.connect(self.name)
        rows = get_rows(conn, 'images', False)

        for row in rows:
            self.imagelist[row[0]] = ImageFeature(row[1])
            #print('\t{}'.format(row))

        print 'Total image', len(self.imagelist)

    def get_image_feature(self):
        conn = sqlite3.connect(self.name)
        kp = get_rows(conn, 'keypoints', False)
        for p in kp:
            self.imagelist[p[0]].add_key_point(p)

        des = get_rows(conn, 'descriptors', False)
        for p in des:
            self.imagelist[p[0]].add_descriptor(p)


if __name__ == "__main__":
    db = '/home/weihao/Projects/colmap_features/proj1/proj1.db'
    c = Colmap_DB(db)

    c.get_image_list()
    c.get_image_feature()

    c.get_image_match()

