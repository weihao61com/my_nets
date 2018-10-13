import sqlite3
import struct


def camera_params(data):
    return struct.unpack('dddd', data)


def get_rows(conn, table_name):
    c = conn.cursor()
    print '\ttable', table_name
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
    ft = "<{}{}".format(len(data)/4, fmt)
    d = struct.unpack(ft, data)
    return d

def view_data_table(conn, table_name, fmt):
    rows = get_rows(conn, table_name)

    nt = 0
    for row in rows:
        length = int(row[1])
        print'\t', nt, row, len(row[3])/length
        data = get_data(row[3], fmt)
        print '\t', data[:6]
        nt += 1
        if nt>5:
            break


def view_table(conn, table_name):
    rows = get_rows(conn, table_name)
    nt = 0
    for row in rows:
        print'\t', nt, row
        nt += 1
        if nt>5:
            break


if __name__ == "__main__":
    db = '/Users/weihao/Downloads/proj1.db'
    conn = sqlite3.connect(db)
    # c = conn.cursor()
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for name in res:
        print name[0]

    view_cameras_table(conn)
    # view_table(conn, 'sqlite_sequence')
    view_table(conn, 'images')
    view_data_table(conn, 'keypoints', 'f')
    #view_data_table(conn, 'descriptors', 'I')
    #view_data_table(conn, 'matches')
    #view_table(conn, 'inlier_matches')

