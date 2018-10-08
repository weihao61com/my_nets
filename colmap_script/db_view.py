import sqlite3
import struct


def view_table(conn, table_name):
    c = conn.cursor()

    cameras = c.execute('select * from {}'.format(table_name))
    c_names = list(map(lambda x: x[0], cameras.description))
    for name in c_names:
        print '\t', name

    rows = c.fetchall()

    nt = 0
    for row in rows:
        print(row)
        data1 = struct.unpack('dddff', row[4])
        print(data1)
        nt += 1
        if nt>10:
            break


if __name__ == "__main__":
    db = '/home/weihao/Downloads/proj1.db'
    conn = sqlite3.connect(db)
    # c = conn.cursor()
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for name in res:
        print name[0]

    view_table(conn, 'cameras')

