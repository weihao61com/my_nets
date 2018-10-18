import sqlite3
import struct


def camera_params(data):
    return struct.unpack('dddd', data)


def get_rows(conn, table_name):
    c = conn.cursor()
    print '\n\ttable', table_name
    cameras = c.execute('select * from {}'.format(table_name))
    c_names = list(map(lambda x: x[0], cameras.description))
    for name in c_names:
        print '\t', name
    rows = c.fetchall()
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


if __name__ == "__main__":
    import sys
    db = '/home/weihao/Projects/colmap_features/fire_Test/proj.db'
    if len(sys.argv)>1:
        db =sys.argv[1]
    conn = sqlite3.connect(db)
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for name in res:
        print name[0]

    view_data_table(conn, 'cameras', 4, 'd')
    view_data_table(conn, 'sqlite_sequence')
    view_data_table(conn, 'images',3, None)
    view_data_table(conn, 'keypoints', 3, 'd')
    view_data_table(conn, 'descriptors', 3, 'I')
    view_data_table(conn, 'matches',3, 'I')
    view_data_table(conn, 'two_view_geometries',3, 'I')
    view_data_table(conn, 'two_view_geometries',5, 'd')
    view_data_table(conn, 'two_view_geometries',6, 'd')
    view_data_table(conn, 'two_view_geometries',7, 'd')

