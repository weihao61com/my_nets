import sys
import tensorflow as tf
import datetime
import pickle
import os
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("fc")

sys.path.append('..')
sys.path.append('.')
from utils import Utils, Config, HOME
from network import Network
from distance import distance_cal


def transfor_T(pose, r, w2c=True):
    if w2c:
        return pose.inv.dot(np.concatenate((r, [1])))[:3]
    return pose.Q4.dot(np.concatenate((r, [1])))[:3]


def find_idx(xy, id_list):
    idx = 0
    for id in id_list:
        if abs(xy[0]-id[0])<0.0001:
            if abs(xy[1]-id[1])<0.0001:
                return idx
        idx += 1
    return -1


class DataSet:
    def __init__(self, p_file, cfg, imf=None):
        if p_file is not None:
            with open(os.path.join(HOME, p_file), 'rb') as fp:
                self.data = pickle.load(fp, encoding='latin1')
                # self.data = pickle.load(fp)
                self.id_list = {}
                # print(self.data[0])
        else:
            with open(os.path.join(HOME, imf), 'rb') as fp:
                A = pickle.load(fp)
                self.data = A[0]
                # self.data = pickle.load(fp)
                self.id_list = A[1]
                # print(self.data[0][2])
        self.cfg = cfg
        nt = 0
        x0 = 140
        y0 = 100
        win = 1
        for d in self.data:
            d[0].add_inv(1)
            d[1].add_inv(1)

            nt += len(d[2])
        logger.info("Total match {}".format(nt))
        self.nt = nt
        self.dists_dic = {}
        self.out = None


    def run_data_0(self, data, inputs, sess, xy):
        logger.info("run_data_0")
        rst = data[0]
        truth = data[1]
        ids = data[2]
        cmap = data[3]
        trans = data[4]

        feed = {}
        mode = abs(self.cfg.mode)
        nt = 0
        print('rst:', len(rst))
        for rt in rst:
            feed[inputs['data_{}'.format(mode)]] = np.array(rt)
            r = sess.run(xy[mode-1], feed_dict=feed)
            sz = r.shape
            xyz0 = trans[nt].dot(np.concatenate((r, np.ones((sz[0],1))), axis=1 ).transpose())
            idds = ids[nt]
            length = len(idds[2])
            for a in range(length):
                xyz = xyz0[:3, a]
                id1 = idds[0]
                id2 = idds[1]
                id = idds[2][a]
                #if id1==10 and id[0]==307:
                #    print(id1, id[0], xyz, truth[nt], cmap[nt])
                # print(xyz, idds[:2], id)
                self.add_xyz(xyz, id1, id[0])
                self.add_xyz(xyz, id2, id[1])

            nt += 1

        for id1 in self.dists_dic:
            for id2 in self.dists_dic[id1]:
                dists = self.dists_dic[id1][id2]
                if len(dists)>50:
                    print(id1, id2, len(dists))
                    for d in dists:
                        print(d[0], d[1], d[2])

    def add_xyz(self, xyz, id1, id2):
        if id1 not in self.dists_dic:
            self.dists_dic[id1] = {}
        if id2 not in self.dists_dic[id1]:
            self.dists_dic[id1][id2] = []
        self.dists_dic[id1][id2].append(xyz)

    def get_avg(self, filename):
        x = []
        for d in self.data:
            for a in d[2]:
                x.append(a[1])
                x.append(a[2])

        x = np.array(x)
        avx = np.mean(x, axis=0)
        stx = np.std(x, axis=0)

        print(avg_file)
        with open(filename, 'wb') as fp:
            pickle.dump((avx, stx), fp)

    def get_id(self, xy, id):
        #if id not in self.id_list:
        #    self.id_list[id] = []
        idx1 = find_idx(xy, self.id_list[id])
        if idx1==-1:
            idx1 = len(self.id_list[id])
            self.id_list[id].append(xy)
        return idx1

    def subtract_avg(self, filename, save_im=True):
        with open(filename, 'br') as fp:
            av, st = pickle.load(fp, encoding='latin1')
            print(av,st)
            # raise Exception()
            # av, st = pickle.load(fp)
            nt = 0
            for d in self.data:
                id1 = d[0].id
                id2 = d[1].id
                if id1==840 and id2==872:
                    print(id1, id2)
                if id2==840 and id1==872:
                    print(id1, id2)
                if id1 not in self.id_list:
                    self.id_list[id1] = []
                if id2 not in self.id_list:
                    self.id_list[id2] = []
                for a in range(len(d[2])):
                    x = np.array(d[2][a][1])
                    y = np.array(d[2][a][2])
                    if save_im:
                        idx1 = self.get_id(x, id1)
                        idx2 = self.get_id(y, id2)
                    else:
                        idx1 = 0
                        idx2 = 0
                    x = (x - av) / st
                    y = (y - av) / st
                    d[2][a] = (d[2][a][0], x, y, idx1, idx2)
                # print(id1, id2, len(d[2]), len(self.id_list[id1]), len(self.id_list[id2]), len(self.id_list))

                if nt%2000==0:
                    # (self.data[0][2])
                    logger.info('subtract_avg {} {} {}'.format(nt, len(self.id_list), len(self.data)))
                nt += 1

        if save_im:
            tmp_file = 'c:/Projects/tmp/t_pairs.p'
            with open(tmp_file, 'bw') as fp:
                pickle.dump((self.data, self.id_list), fp)


    def prepare(self, count=None, clear=False):
        #logger.info("Prepare data")
        if self.cfg.mode == 1:
            return self.prepare_n(count, 1, clear)
        elif self.cfg.mode == 2:
            return self.prepare_n(count, 2, clear)
        elif self.cfg.mode == 3:
            return self.prepare_n(count, 3, clear)
        elif self.cfg.mode == -1:
            return self.prepare4()
        elif self.cfg.mode == -2:
            return self.prepare5()
        elif self.cfg.mode == -3:
            return self.prepare6()
        else:
            raise Exception('Unknown mode {}'.format(cfg.mode))

    def prepare_n(self, count, n, clear):

        if count:
            r = float(count) / self.nt

        if self.out is None:
            out = []
            nc = 0
            for d in self.data:
                P1 = d[0]
                P2 = d[1]
                A1, T1 = Utils.get_relative(P1, P2)
                A2, T2 = Utils.get_relative(P2, P1)
                length = len(d[2])
                for b in range(length):
                    a = d[2][b]
                    #if count is None or random.random() < r/2:
                    #if len(out)<count:
                    xyz = a[0]
                    p1 = a[1]
                    p2 = a[2]

                    if n==1:
                        data = np.concatenate((A1, T1, p1, p2))
                        out.append((data, (P1, P2, a[3], a[4], xyz)))
                        data = np.concatenate((A2, T2, p2, p1))
                        out.append((data, (P2, P1, a[4], a[3], xyz)))
                    elif n==2:
                        data = np.concatenate((xyz0, A1, p1, p2))
                        out.append((data, T1))
                        data = np.concatenate((xyz0, A2, p2, p1))
                        out.append((data, T2))
                    elif n==3:
                        data = np.concatenate((xyz0, T1, A1[1], A[2], p1, p2))
                        out.append((data, [A1[0]]))
                    elif n==4:
                        data = np.concatenate((xyz0, T1, A1[0], A[2], p1, p2))
                        out.append((data, A1[1]))
                    elif n==5:
                        data = np.concatenate((xyz0, T1, A1[1], A[2], p1, p2))
                        out.append((data, A1[2]))
                    else:
                        raise Exception("Unknown n={}".format(n))

                    #if nc%100000==0 and nc>0:
                    #    logger.info("prepare {} {}".format(n, len(out)))

            self.out = out
            self.data = None

        np.random.shuffle(self.out)
        # logger.info("prepare {} {} {}".format(n, len(out), nc))
        if clear:
            self.data = None

        rst = []
        truth = []
        for b in self.out:
            if count is None or random.random() < r/2:
                rst.append(b[0])
                truth.append(b[1])
        return np.array(rst), truth

    def get_data4(self, P1, P2, d, swap):
        A, T = Utils.get_relative(P1, P2)
        data_array = []
        id_array = []
        for a in d[2]:
            # xyz = a[0]
            if swap:
                p1 = a[2]
                p2 = a[1]
                id1 = a[4]
                id2 = a[3]
            else:
                p1 = a[1]
                p2 = a[2]
                id1 = a[3]
                id2 = a[4]
            #xyz0 = np.linalg.inv(P1.Q4).dot(np.concatenate((xyz, [1])))[:3]
            data = np.concatenate((A, T, p1, p2))
            #out.append((data, T, (P1.id, P2.id), T0))
            data_array.append(data)
            id_array.append((id1, id2))

        if swap:
            tran = d[4].tran[:3]
        else:
            tran = d[3].tran[:3]
        return data_array, P2.tran, (P2.id, P1.id, id_array), tran, P1.Q4

    def get_data5(self, d, swap):
        dr = 0
        P1 = d[0]
        P2 = d[1]
        Q1 = d[0+dr]
        Q2 = d[1+dr]
        if swap:
            P1 = d[1]
            P2 = d[0]
            Q1 = d[1+dr]
            Q2 = d[0+dr]
        # A, T = Utils.get_relative(P1, P2)
        A, T = Utils.get_relative(Q1, Q2)
        data_array = []
        for a in d[2]:
            xyz = a[0]
            p1 = a[1]
            p2 = a[2]
            if swap:
                p1 = a[2]
                p2 = a[1]

            xyz0 = Q1.inv.dot(np.concatenate((xyz, [1])))[:3]
            data = np.concatenate((xyz0, A, p1, p2))
            #out.append((data, T, (P1.id, P2.id), T0))
            data_array.append(data)

        if swap:
            tran = d[3].tran[:3]
        else:
            tran = d[4].tran[:3]
        return data_array, P2.tran, (P2.id, P1.id), tran, Q1.Q4

    def get_data6(self, d, swap):
        dr = 3
        P1 = d[0]
        P2 = d[1]
        Q1 = d[0+dr]
        Q2 = d[1+dr]
        if swap:
            P1 = d[1]
            P2 = d[0]
            Q1 = d[1+dr]
            Q2 = d[0+dr]
        A, T = Utils.get_relative(Q1, Q2)
        data_array = []
        for a in d[2]:
            xyz = a[0]
            p1 = a[1]
            p2 = a[2]
            if swap:
                p1 = a[2]
                p2 = a[1]

            xyz0 = Q1.inv.dot(np.concatenate((xyz, [1])))[:3]
            data = np.concatenate((xyz0, T, A[:2], p1, p2))
            #out.append((data, T, (P1.id, P2.id), T0))
            data_array.append(data)

        if swap:
            tran = Utils.get_A(d[3].m3x3)
        else:
            tran = Utils.get_A(d[4].m3x3)
        return data_array, Utils.get_A(P2.m3x3), (P2.id, P1.id), tran, Q1.Q4


    def prepare4(self):
        ID0 = 0
        ID1 = 100
        out = []
        for d in self.data:
            P1 = d[0]
            P2 = d[1]
            # len(out)==17:
            #    print()
            if ID1 > P2.id >= ID0:
                out.append(self.get_data4(P1, P2, d, False))
            if ID1 > P1.id >= ID0:
                out.append(self.get_data4(P2, P1, d, True))

        rst = []
        truth = []
        ids = []
        Ts = []
        trans = []
        for b in out:
            if len(b[0])>1:
                rst.append(b[0])
                truth.append(b[1])
                ids.append(b[2])
                Ts.append(b[3])
                trans.append(b[4])
        return rst, np.array(truth), ids, Ts, trans

    def prepare5(self):
        ID0 = -600
        ID1 = 6100
        out = []
        for d in self.data:
            P1 = d[0]
            P2 = d[1]
            if ID1 > P2.id >= ID0:
                out.append(self.get_data5(d, False))
            if ID1 > P1.id >= ID0:
                out.append(self.get_data5(d, True))
            if len(out)%1000==0:
                logger.info("prepare5 {} from {}".format(len(out), len(self.data)))

        rst = []
        truth = []
        ids = []
        Ts = []
        trans = []
        for b in out:
            if len(b[0])>1:
                rst.append(b[0])
                truth.append(b[1])
                ids.append(b[2])
                Ts.append(b[3])
                trans.append(b[4])
        return rst, np.array(truth), ids, Ts, trans

    def prepare6(self):
        ID0 = -588
        ID1 = 6000
        out = []
        for d in self.data:
            P1 = d[0]
            P2 = d[1]
            # len(out)==17:
            #    print()
            if ID1 > P2.id >= ID0:
                out.append(self.get_data6(d, False))
            if ID1 > P1.id >= ID0:
                out.append(self.get_data6(d, True))

        self.data = None
        rst = []
        truth = []
        ids = []
        Ts = []
        trans = []
        for b in out:
            if len(b[0])>1:
                rst.append(b[0])
                truth.append(b[1])
                ids.append(b[2])
                Ts.append(b[3])
                trans.append(b[4])
        return rst, np.array(truth), ids, Ts, trans


class P1Net1(Network):

    def setup(self):
        pass

    def real_setup(self, nodes, num_output):
        self.feed('data')
        for a in range(len(nodes)):
            name = 'fc_{}'.format(a)
            self.fc(nodes[a], name=name)
            # self.dropout(0.4, name='drop_{}'.format(a))
        self.fc(num_output, relu=False, name='output')


def avg_file_name(p):
    basename = os.path.basename(p)
    pathname = os.path.dirname(p)
    return pathname + '_' + basename + '_avg.p'


def run_data(data, inputs, sess, xy, fname, cfg):
    rst = data[0]
    truth = data[1]
    feed = {}
    feed[inputs['data_{}'.format(cfg.mode)]] = np.array(rst)
    r = sess.run(xy[cfg.mode-1], feed_dict=feed)

    rt = 2000.0 / len(rst)
    filename = 'c:/tmp/{}.csv'.format(fname)
    if sys.platform == 'darwin':
        filename = '/Users/weihao/tmp/{}.csv'.format(fname)
    elif sys.platform == 'win32':
        filename = 'c:/tmp/{}.csv'.format(fname)

    fp = open(filename, 'w')
    rs = []
    for d in range(len(rst)):
        mm = r[d, :]
        t = truth[d]
        r0 = np.linalg.norm(mm - t)
        if random.random() < rt:
            for a in range(len(t)):
                if a > 0:
                    fp.write(',')
                fp.write('{},{}'.format(t[a], mm[a]))
            fp.write(',{}\n'.format(r0))
    fp.close()
    diff = r - np.array(truth)
    dist = np.linalg.norm(diff, axis=1)
    return np.mean(dist * dist), np.median(dist)


def run_data_t(data, inputs, sess, xy, fname, cfg):
    rst = data[0]
    truth = data[1]
    ids = data[2]
    cmap = data[3]
    feed = {}
    mode = abs(cfg.mode)
    r = []
    for a in range(len(rst)):
        ins = np.array(rst[a])
        feed[inputs['data_{}'.format(mode)]] = ins
        rs = sess.run(xy[mode-1], feed_dict=feed)
        r.append(np.median(rs, axis=0))

    filename = '/home/weihao/tmp/{}.csv'.format(fname)
    if sys.platform == 'darwin':
        filename = '/Users/weihao/tmp/{}.csv'.format(fname)
    elif sys.platform == 'win32':
        filename = 'c:/tmp/{}.csv'.format(fname)
    fp = open(filename, 'w')
    rs = {}
    for d in range(len(rst)):
        id = ids[d]
        if id not in rs:
            rs[id] = []
        mm = r[d]
        rs[id] = (mm, truth[d], cmap[d])

    for id in rs:
        val = rs[id]
        mm = val[0]
        t = val[1]
        c = val[2]
        r0 = np.linalg.norm(mm-t)
        r1 = np.linalg.norm(c-t)
        fp.write('{},'.format(id)) #[0], id[1]))
        for a in range(len(t)):
            fp.write('{},{},{},'.format(t[a], mm[a], c[a]))
        fp.write('{},{}\n'.format(r0, r1))
    fp.close()
    diff = r - np.array(truth)
    dist = np.linalg.norm(diff, axis=1)
    diff = np.array(cmap) - np.array(truth)
    dist0 = np.linalg.norm(diff, axis=1)
    return np.sqrt(np.mean(dist * dist)), np.median(dist), np.sqrt(np.mean(dist0 * dist0)), np.median(dist0)


def save_ply(rst_dic, filename, fraction, idx=0):
    nt = 0
    xyz = []
    for image_id in rst_dic:
        for point_id in rst_dic[image_id]:
            nt += 1
            if random.random()<fraction:
                xyz.append(rst_dic[image_id][point_id][idx])

    print("PLY {} out of {}".format(len(xyz), nt))
    Utils.create_ply(filename, xyz)


if __name__ == '__main__':

    config_file = "config.json"

    if len(sys.argv) > 2:
        config_file = sys.argv[2]

    cfg = Config(config_file)

    if len(sys.argv) > 1:
        mode = int(sys.argv[1])
        cfg.mode = mode

    iterations = 200
    logger.info("LR {} num_out {} mode {}".format(cfg.lr, cfg.num_output, cfg.mode))

    input_dic = {}
    output = []
    num_output = cfg.num_output[0]
    feature_len = 13 - num_output
    output = tf.compat.v1.placeholder(tf.float32, [None, num_output])
    input_dic['data'] =  tf.compat.v1.placeholder(tf.float32, [None, feature_len])

    net = P1Net1(input_dic)
    net.real_setup(cfg.nodes[0], num_output)

    xy = net.layers['output']
    loss = tf.reduce_sum(tf.square(tf.subtract(xy, output)))
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.lr, beta1=0.9,
                                          beta2=0.999, epsilon=0.00000001,
                                          use_locking=False, name='Adam').minimize(loss)
    # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    avg_file = avg_file_name(cfg.netFile)

    if hasattr(cfg, 'inter_file'):
        tr = DataSet(None,  cfg, cfg.inter_file)
    else:
        tr = DataSet(cfg.tr_data[0], cfg)
        tr.get_avg(avg_file)
        tr.subtract_avg(avg_file, save_im=True)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if cfg.mode == 0:
            saver.save(sess, cfg.netFile)
            exit(0)

        saver.restore(sess, cfg.netFile)

        t00 = datetime.datetime.now()
        st1 = ''

        img0 = 40
        id0 = 75
        img1 = 766
        id1 = 32
        img2 = 250
        id2 = 30
        threshold = 10000.0
        rst_dic = None,
        for a in range(iterations):

            for lp in range(cfg.loop):
                tr_pre = tr.prepare(200000, clear=True)

                rst_dic = {}
                data = tr_pre[0]
                truth = tr_pre[1]
                length = data.shape[0]

                for c in range(0, length, cfg.batch_size):
                    dd = data[c:c + cfg.batch_size]
                    feed = {input_dic['data']: dd}
                    A = sess.run(xy, feed_dict=feed)
                    T = truth[c:c + cfg.batch_size]
                    total = len(T)
                    for b in range(total):
                        a = A[b, :]
                        t = T[b]
                        xyz = transfor_T(t[0], a, w2c=False)
                        xyz0 = t[4] # transfor_T(t[0].Q4, t[4], w2c=False)
                        xyz = (xyz, xyz0)

                        if t[0].id not in rst_dic:
                            rst_dic[t[0].id] = {}
                        if t[2] not in rst_dic[t[0].id]:
                            rst_dic[t[0].id][t[2]] = []
                        rst_dic[t[0].id][t[2]].append(xyz)
                        #if t[0].id==img0 and t[2]==id0:
                        #    print(t[1].id, t[3], xyz)

                        if t[1].id not in rst_dic:
                            rst_dic[t[1].id] = {}
                        if t[3] not in rst_dic[t[1].id]:
                            rst_dic[t[1].id][t[3]] = []
                        rst_dic[t[1].id][t[3]].append(xyz)
                        #if t[1].id==img0 and t[3]==id0:
                        #    print(t[0].id, t[2], xyz)

                # print
                err_str = ''
                error = 0
                total_count = 0
                for image_id in rst_dic:
                    for point_id in rst_dic[image_id]:
                        xyzs = rst_dic[image_id][point_id]
                        count = len(xyzs)
                        xyz0 = xyzs[0][1]

                        xyzs = np.array(xyzs)
                        #if np.linalg.norm(np.std(xyzs[:,1,:], axis=0))>0.001:
                        #    print(xyzs)
                        #    raise Exception()

                        xyzs = xyzs[:, 0, :]
                        #
                        #if count>1:
                        #     print(image_id, point_id, count)
                        #     print(xyzs)
                        # (xyzs.shape)
                        if xyzs.shape[0]<3:
                            mm = np.mean(xyzs, axis=0)
                        else:
                            mm = np.median(xyzs, axis=0)
                        rst_dic[image_id][point_id] = (mm, xyz0, count)
                        st = np.std(xyzs, axis=0)
                        #if count>10:
                        #    err_str += '\n{} {} {} {} {}'.format(image_id, mm, xyz0, st, count)
                        if image_id==img0 and point_id==id0:
                            err_str += '\n{} {} {} {} {}'.format(image_id, mm, xyz0, st, count)
                        if image_id==img1 and point_id==id1:
                            err_str += '\n{} {} {} {} {}'.format(image_id, mm, xyz0, st, count)
                        if image_id==img2 and point_id==id2:
                            err_str += '\n{} {} {} {} {}'.format(image_id, mm, xyz0, st, count)
                        error += np.sum(st*st)
                        total_count += count
                print(err_str)

                t_loss = 0
                t_count = 0
                distances = []
                for c in range(0, length, cfg.batch_size):
                    dx = data[c:c + cfg.batch_size]
                    dd = [] #data[c:c + cfg.batch_size]
                    th = []
                    for d in range(c, c+dx.shape[0]):
                        t = truth[d]
                        th1 = rst_dic[t[0].id][t[2]]
                        th2 = rst_dic[t[1].id][t[3]]

                        dist = np.linalg.norm(np.array(th1[0]-th2[0]))
                        if dist<threshold:
                            dd.append(data[d])
                            if cfg.usetruth==0:
                                xyz = (th1[0]+th2[0])/2
                                #if xyz[2]<0.5:
                                #    xyz[2]=0.5
                            else:
                                xyz = th1[1]
                            xyz = transfor_T(t[0], xyz)
                            if xyz[2]<0.4 and cfg.usetruth==0:
                                #if cfg.usetruth!=0:
                                #    print(t[0].id, t[1].id, xyz, t[2:])
                                xyz[2] = 0.4
                            th.append(xyz)
                        distances.append(dist)
                    feed = {input_dic['data']: np.array(dd), output: np.array(th)}
                   # _ = sess.run(opt, feed_dict=feed)
                    A,_ = sess.run([loss, opt], feed_dict=feed)
                    t_loss += A
                    t_count += len(th)

                threshold = distance_cal(distances, 10, False)
                logger.info("Err {0:.6f} {1:.6f} {2}".format(error/total_count, t_loss/t_count, threshold))
            saver.save(sess, cfg.netFile)

            with open('c:\\tmp\\distance.p', 'wb') as fp:
                pickle.dump(distances, fp)

            # save_ply(rst_dic, 'c:\\tmp\\points.ply', 0.2)
