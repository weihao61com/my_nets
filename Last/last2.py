import sys
import tensorflow as tf
import datetime
import pickle
import os
import numpy as np
import random
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("last2")
logger.setLevel(logging.INFO)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
#logger.setFormatter(formatter)

sys.path.append('..')
sys.path.append('.')
from utils import Utils, Config, HOME
from network import Network
from distance import distance_cal
import evo.core.transformations as etr



def data_form(xyz, T, A0, A1, A2, p1, p2, P1, P2, a3, a4):
    data = np.concatenate((xyz, T, [A1], [A2], p1, p2))
    return data, (P1, P2, a3, a4, A0)

def data_form_1(xyz, T, A, p1, p2, P1, P2, a3, a4):
    data = np.concatenate((xyz, T, [A[2]], p1, p2))
    return data, (P1, P2, a3, a4, A[:2])

def data_form_2(xyz, T, A, p1, p2, P1, P2, a3, a4):
    data = np.concatenate((xyz, T, A[:2], p1, p2))
    return data, (P1, P2, a3, a4, A[2])

def data_form_3(xyz, T, A, p1, p2, a3, a4):
    data = np.concatenate((T, A, p1, p2))
    return data, (a3, a4, xyz)

def find_idx(xy, id_list):
    idx = 0
    for id in id_list:
        if abs(xy[0]-id[0])<0.0001:
            if abs(xy[1]-id[1])<0.0001:
                return idx
        idx += 1
    return -1


class DataSet2:
    def __init__(self, p_file, cfg, imf=None, mx=None):
        if p_file is not None:
            with open(p_file, 'rb') as fp:
                d = pickle.load(fp, encoding='latin1')
                self.images = d[0]
                self.matches = d[1]
                self.poses = d[2]
                print("Total data {} {} in {}".format(len(self.images), len(self.matches), p_file))
        else:
            with open(os.path.join(HOME, imf), 'rb') as fp:
                A = pickle.load(fp)
                self.data = A[0]
                # self.data = pickle.load(fp)
                self.id_list = A[1]
        self.cfg = cfg
        nt = 0
        for id in self.images:
            img = self.images[id]
            pose = self.poses[id]
            Q = etr.quaternion_matrix(img[0])
            self.poses[id].Q4 = etr.quaternion_matrix(img[0])
            self.poses[id].Q4[:3, 3] = img[1]
            # print(self.poses[id].Q4)
            self.poses[id].add_inv()
        self.out = None

    def get_avg(self, filename):
        x = []
        for d in self.matches:
            mm = self.matches[d]
            for a in mm:
                x.append(a[1])
                x.append(a[2])

        x = np.array(x)
        avx = np.mean(x, axis=0)
        stx = np.std(x, axis=0)

        print("Average: ", avx)
        print("STD:     ", stx)
        print(filename)
        with open(filename, 'wb') as fp:
            pickle.dump((avx, stx), fp)

    def subtract_avg(self, filename):
        with open(filename, 'br') as fp:
            av, st = pickle.load(fp, encoding='latin1')
            print(av, st)
            # raise Exception()
            # av, st = pickle.load(fp)
            nt = 0
            for d in self.matches:
                mm = self.matches[d]

                for a in range(len(mm)):
                    # for a in range(len(d[2])):
                    x = np.array(mm[a][1])
                    y = np.array(mm[a][2])

                    idx1 = d[0]
                    idx2 = d[1]
                    x = (x - av) / st
                    y = (y - av) / st
                    mm[a] = (mm[a][0], x, y, d, mm[a][3])
                # print(id1, id2, len(d[2]), len(self.id_list[id1]), len(self.id_list[id2]), len(self.id_list))

                nt += 1
                if nt%2000==0:
                    # (self.data[0][2])
                    print('subtract_avg {} {}'.format(nt, len(self.matches)))

    def prepare(self, count=None, clear=False):
        #print("Prepare data")
        if self.cfg.mode > 0:
            return self.prepare_n(count, self.cfg.mode, clear)
        elif self.cfg.mode == -1:
            out = self.prepare4(2)
            rst = []
            truth = []
            for b in out:
                #if count is None or random.random() < r/2:
                rst.append(b[0])
                truth.append(b[1])
            # print("load: {}".format(len(truth)))
            return np.array(rst), truth
        else:
            raise Exception('Unknown mode {}'.format(cfg.mode))

    def prepare4(self,  r):
        if self.out is None:
            out = []
            num = 20

            for imgs in self.matches:
                id1 = imgs[0]
                id2 = imgs[1]
                if id1 > num or id2 > num:
                    continue
                P1 = self.poses[id1]
                P2 = self.poses[id2]
                A1, T1 = Utils.get_relative(P1, P2)
                A2, T2 = Utils.get_relative(P2, P1)
                for a in self.matches[imgs]:
                    if random.random() < r:
                        xyz = a[0]
                        p1 = a[1]
                        p2 = a[2]
                        xyz1 = Utils.transfor_T(P1, xyz)
                        xyz2 = Utils.transfor_T(P2, xyz)
                        xyz1 = Utils.xyz_tran(xyz1)
                        xyz2 = Utils.xyz_tran(xyz2)
                        a0 = [a[3][0], a[4][0]]
                        a1 = [a[3][1], a[4][1]]
                        out.append(data_form_3(xyz1, T1, A1, p1, p2, a0, a1))
                        out.append(data_form_3(xyz2, T2, A2, p2, p1, a1, a0))
            self.out = out
        return self.out


class DataSet:
    def __init__(self, p_file, cfg, imf=None, mx=None):
        if p_file is not None:
            with open(p_file, 'rb') as fp:
                self.data = pickle.load(fp, encoding='latin1')
                if mx:
                    self.data = self.data[:mx]
                # self.data = pickle.load(fp)
                self.id_list = {}
                # print(self.data[0])
                print("Total data {}".format(len(self.data)))
        else:
            with open(os.path.join(HOME, imf), 'rb') as fp:
                A = pickle.load(fp)
                self.data = A[0]
                # self.data = pickle.load(fp)
                self.id_list = A[1]
        self.cfg = cfg
        nt = 0
        for d in self.data:
            d[0].add_inv()
            d[1].add_inv()
            nt += len(d[2])
        print("Total match {}".format(nt))
        self.nt = nt
        self.dists_dic = {}
        self.out = None



    def run_data_0(self, data, inputs, sess, xy):
        print("run_data_0")
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

        print(filename)
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
                        ids = d[2][a][3]
                        idx1 = ids[0]
                        idx2 = ids[1]
                    x = (x - av) / st
                    y = (y - av) / st
                    d[2][a] = (d[2][a][0], x, y, idx1, idx2, d[3])
                # print(id1, id2, len(d[2]), len(self.id_list[id1]), len(self.id_list[id2]), len(self.id_list))

                nt += 1
                if nt%2000==0:
                    # (self.data[0][2])
                    print('subtract_avg {} {} {}'.format(nt, len(self.id_list), len(self.data)))

        if save_im:
            tmp_file = '{}/tmp/t_pairs.p'.format(HOME)
            with open(tmp_file, 'bw') as fp:
                pickle.dump((self.data, self.id_list), fp)


    def prepare(self, count=None, clear=False):
        #print("Prepare data")
        if self.cfg.mode > 0:
            return self.prepare_n(count, self.cfg.mode, clear)
        elif self.cfg.mode == -1:
            out = self.prepare4(self.data, 2)
            rst = []
            truth = []
            for b in out:
                #if count is None or random.random() < r/2:
                rst.append(b[0])
                truth.append(b[1])
            # print("load: {}".format(len(truth)))
            return np.array(rst), truth
        else:
            raise Exception('Unknown mode {}'.format(cfg.mode))

    @staticmethod
    def prepare_data(sdata, n, r):
        out = []
        nc = 0
        for d in sdata:
            P1 = d[0]
            P2 = d[1]

            # A10, T10 = Utils.get_relative(P1, P2)
            # A20, T20 = Utils.get_relative(P2, P1)

            P1.Q4 = Utils.get_pose(d[3][0], d[3][1])
            P1.inv = np.linalg.inv(P1.Q4)
            P2.Q4 = Utils.get_pose(d[3][2], d[3][3])
            P2.inv = np.linalg.inv(P2.Q4)
            # Q_org = poses_by_name[img.name]
            # Q_org.add_inv()
            # xyz = []
            # xyz_c = []
            # for p in D3:
            #     xyz.append(p.xyz)
            #     # c = Utils.transfor_T(Q_org, p.xyz)
            #     c = Q.dot(np.concatenate((p.xyz, [1])))[:3]
            #     if c[2]<-15:
            #         print()
            #     xyz_c.append(c)
            A1, T1 = Utils.get_relative(P1, P2)
            A2, T2 = Utils.get_relative(P2, P1)
            length = len(d[2])
            for b in range(length):
                a = d[2][b]
                if random.random() < r:
                    # if len(out)<count:
                    xyz = a[0]
                    p1 = a[1]
                    p2 = a[2]
                    xyz1 = Utils.transfor_T(P1, xyz)
                    xyz2 = Utils.transfor_T(P2, xyz)
                    if n == 1:
                        out.append(data_form_1(xyz1, T1, A1, p1, p2, P1, P2, a[3], a[4]))
                        out.append(data_form_1(xyz2, T2, A2, p2, p1, P2, P1, a[4], a[3]))
                        # data = np.concatenate((A1, T1, p1, p2))
                        # out.append((data, (P1, P2, a[3], a[4], xyz1)))
                        # data = np.concatenate((A2, T2, p2, p1))
                        # out.append((data, (P2, P1, a[4], a[3], xyz2)))
                    elif n == 2:
                        out.append(data_form_2(xyz1, T1, A1, p1, p2, P1, P2, a[3], a[4]))
                        out.append(data_form_2(xyz2, T2, A2, p2, p1, P2, P1, a[4], a[3]))
                        # data = np.concatenate((xyz1, A1, [T1[2]], p1, p2))
                        # out.append((data, (P1, P2, a[3], a[4], T1[:2])))
                        # data = np.concatenate((xyz2, A2, [T2[2]], p2, p1))
                        # out.append((data, (P2, P1, a[4], a[3], T2[:2])))
                    elif n == 3:
                        out.append(data_form_1(xyz1, A1, T1, p1, p2, P1, P2, a[3], a[4]))
                        out.append(data_form_1(xyz2, A2, T2, p2, p1, P2, P1, a[4], a[3]))
                        # data = np.concatenate((xyz1, A1, T1[:2], p1, p2))
                        # out.append((data, (P1, P2, a[3], a[4], T1[2])))
                        # data = np.concatenate((xyz2, A2, T2[:2], p2, p1))
                        # out.append((data, (P2, P1, a[4], a[3], T2[2])))
                    elif n == 4:
                        out.append(data_form_2(xyz1, A1, T1, p1, p2, P1, P2, a[3], a[4]))
                        out.append(data_form_2(xyz2, A2, T2, p2, p1, P2, P1, a[4], a[3]))
                        #out.append(data_form(xyz1, T1, A1[0], A1[1], A1[2], p1, p2, P1, P2, a[3], a[4]))
                        #out.append(data_form(xyz2, T2, A2[0], A2[1], A2[2], p2, p1, P2, P1, a[4], a[3]))
                        #xyz1, A1 = A1, xyz1
                        #xyz2, A2 = A2, xyz2
                        #out.append(data_form(xyz1, T1, A1[0], A1[1], A1[2], p1, p2, P1, P2, a[3], a[4]))
                        #out.append(data_form(xyz2, T2, A2[0], A2[1], A2[2], p2, p1, P2, P1, a[4], a[3]))
                        # data = np.concatenate((xyz1, T1, [A1[1]], [A1[2]], p1, p2))
                        # out.append((data, (P1, P2, a[3], a[4], A1[0])))
                        # data = np.concatenate((xyz2, T2, [A2[1]], [A2[2]], p2, p1))
                        # out.append((data, (P2, P1, a[4], a[3], A2[0])))
                    elif n == 5:
                        xyz1 = Utils.xyz_tran(xyz1)
                        xyz2 = Utils.xyz_tran(xyz2)
                        out.append(data_form_1(T1, A1, xyz1, p1, p2, P1, P2, a[3], a[4]))
                        out.append(data_form_1(T2, A2, xyz2, p2, p1, P2, P1, a[4], a[3]))
                        #xyz1, A1 = A1, xyz1
                        #xyz2, A2 = A2, xyz2
                        #out.append(data_form(xyz1, T1, A1[1], A1[0], A1[2], p1, p2, P1, P2, a[3], a[4]))
                        #out.append(data_form(xyz2, T2, A2[1], A2[0], A2[2], p2, p1, P2, P1, a[4], a[3]))
                        # data = np.concatenate((xyz1, T1, [A1[0]], [A1[2]], p1, p2))
                        # out.append((data, (P1, P2, a[3], a[4], A1[1])))
                        # data = np.concatenate((xyz2, T2, [A2[0]], [A2[2]], p2, p1))
                        # out.append((data, (P2, P1, a[4], a[3], A2[1])))
                    elif n == 6:
                        xyz1 = Utils.xyz_tran(xyz1)
                        xyz2 = Utils.xyz_tran(xyz2)
                        out.append(data_form_2(T1, A1, xyz1, p1, p2, P1, P2, a[3], a[4]))
                        out.append(data_form_2(T2, A2, xyz2, p2, p1, P2, P1, a[4], a[3]))
                        #xyz1, A1 = A1, xyz1
                        #xyz2, A2 = A2, xyz2
                        #out.append(data_form(xyz1, T1, A1[2], A1[0], A1[1], p1, p2, P1, P2, a[3], a[4]))
                        #out.append(data_form(xyz2, T2, A2[2], A2[0], A2[1], p2, p1, P2, P1, a[4], a[3]))
                        # data = np.concatenate((xyz1, T1, [A1[0]], [A1[1]], p1, p2))
                        # out.append((data, (P1, P2, a[3], a[4], A1[2])))
                        # data = np.concatenate((xyz2, T2, [A2[0]], [A2[1]], p2, p1))
                        # out.append((data, (P2, P1, a[4], a[3], A2[2])))
                    else:
                        raise Exception("Unknown n={}".format(n))

                # if nc%100000==0 and nc>0:
                #    print("prepare {} {}".format(n, len(out)))
        return out

    def prepare_n(self, count, n, clear):

        if count:
            r = float(count) / self.nt

        if self.out is None:
            self.out = DataSet.prepare_data(self.data, n, r)

        np.random.shuffle(self.out)
        if clear:
            self.data = None
            self.id_list = None

        rst = []
        truth = []
        for b in self.out:
            #if count is None or random.random() < r/2:
            rst.append(b[0])
            truth.append(b[1])
        # print("load: {}".format(len(truth)))
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


    def prepare4(self, sdata,  r):
        if self.out is None:
            out = []
            nc = 0
            num=20

            for d in sdata:
                P1 = d[0]
                P2 = d[1]
                length = len(d[2])
                if P1.id>num or P2.id>num:
                    continue
                # print('Match:', P1.id, P2.id, length)
                A1, T1 = Utils.get_relative(P1, P2)
                A2, T2 = Utils.get_relative(P2, P1)
                for b in range(length):
                    a = d[2][b]
                    if random.random() < r:
                        # if len(out)<count:
                        xyz = a[0]
                        p1 = a[1]
                        p2 = a[2]
                        xyz1 = Utils.transfor_T(P1, xyz)
                        xyz2 = Utils.transfor_T(P2, xyz)
                        xyz1 = Utils.xyz_tran(xyz1)
                        xyz2 = Utils.xyz_tran(xyz2)
                        out.append(data_form_3(xyz1, T1, A1, p1, p2, P1, P2, a[3], a[4]))
                        out.append(data_form_3(xyz2, T2, A2, p2, p1, P2, P1, a[4], a[3]))

                    # if nc%100000==0 and nc>0:
                    #    print("prepare {} {}".format(n, len(out)))

            self.out = out
        return self.out


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
                print("prepare5 {} from {}".format(len(out), len(self.data)))

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

    def real_setup(self, nodes, num_outputs):
        nt = 1
        for num_output in num_outputs:
            self.feed('data_{}'.format(nt))
            for a in range(len(nodes)):
                name = 'fc_{}_{}'.format(a, nt)
                self.fc(nodes[a], name=name)
            # self.dropout(0.4, name='drop_{}'.format(a))
            self.fc(num_output, relu=False, name='output_{}'.format(nt))
            nt += 1


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


def create_net(cfg):

    input_dic = {}
    outputs = []
    nt = 1
    num_outputs = cfg.num_output
    for num_output in num_outputs:
        output = tf.compat.v1.placeholder(tf.float32, [None, num_output])
        outputs.append(output)
        input_dic['data_{}'.format(nt)] = tf.compat.v1.placeholder(tf.float32, [None, 10])
        # input_dic['data_{}'.format(nt)] = tf.compat.v1.placeholder(tf.float32, [None, 13-num_output])
        nt += 1

    net = P1Net1(input_dic)
    net.real_setup(cfg.nodes[0], num_outputs)

    xys = []
    losses = []
    opts = []
    nt = 1
    for num_output in num_outputs:
        xy = net.layers['output_{}'.format(nt)]
        loss = tf.reduce_sum(tf.square(tf.subtract(xy, outputs[nt-1])))
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.lr, beta1=0.9,
                                          beta2=0.999, epsilon=0.00000001,
                                          use_locking=False, name='Adam').minimize(loss)
        # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)
        xys.append(xy)
        losses.append(loss)
        opts.append(opt)
        nt += 1

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    return init, saver, input_dic, outputs, losses, opts, xys


def run(cfg, iterations):

    #terations = 50
    print("LR {} num_out {} mode {}".format(cfg.lr, cfg.num_output, cfg.mode))
    #
    # input_dic = {}
    # outputs = []
    # nt = 1
    # num_outputs = cfg.num_output
    # for num_output in num_outputs:
    #     output = tf.compat.v1.placeholder(tf.float32, [None, num_output])
    #     outputs.append(output)
    #     input_dic['data_{}'.format(nt)] =  tf.compat.v1.placeholder(tf.float32, [None, 13-num_output])
    #     nt += 1
    #
    # net = P1Net1(input_dic)
    # net.real_setup(cfg.nodes[0], num_outputs)
    #
    # xys = []
    # losses = []
    # opts = []
    # nt = 1
    # for num_output in num_outputs:
    #     xy = net.layers['output_{}'.format(nt)]
    #     loss = tf.reduce_sum(tf.square(tf.subtract(xy, outputs[nt-1])))
    #     opt = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.lr, beta1=0.9,
    #                                       beta2=0.999, epsilon=0.00000001,
    #                                       use_locking=False, name='Adam').minimize(loss)
    #     # opt = tf.train.GradientDescentOptimizer(learning_rate=cfg.lr).minimize(loss)
    #     xys.append(xy)
    #     losses.append(loss)
    #     opts.append(opt)
    #     nt += 1
    #
    # init = tf.compat.v1.global_variables_initializer()
    # saver = tf.compat.v1.train.Saver()
    #

    net = create_net(cfg)
    init = net[0]
    saver = net[1]
    input_dic = net[2]
    outputs = net[3]
    losses = net[4]
    opts = net[5]
    xyz = net[6]

    avg_file = Utils.avg_file_name(cfg.netFile)
    if hasattr(cfg, 'inter_file'):
        tr = DataSet(None,  cfg, cfg.inter_file)
    else:
        tr = DataSet(cfg.tr_data[0], cfg)
        tr.get_avg(avg_file)
        tr.subtract_avg(avg_file, save_im=False)
        te = DataSet(cfg.te_data[0], cfg)
        te.subtract_avg(avg_file, save_im=False)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        if cfg.mode == 0:
            saver.save(sess, cfg.netFile)
            exit(0)

        saver.restore(sess, cfg.netFile)

        t00 = datetime.datetime.now()
        st1 = ''

        mode = cfg.mode
        print(outputs[mode-1])
        for a in range(iterations):

            t_loss = 0
            t_count = 0
            te_loss = 0
            te_count = 0
            filename = '/home/weihao/Projects/tmp/rst_{}.csv'.format(cfg.mode)

            for lp in range(cfg.loop):
                tr_pre = tr.prepare(2000000, clear=True)

                data = tr_pre[0]
                truth = tr_pre[1]
                length = data.shape[0]

                for c in range(0, length, cfg.batch_size):
                    dd = data[c:c + cfg.batch_size]
                    th = []
                    for d in range(c, c+dd.shape[0]):
                        t = truth[d][4]
                        th.append(t)
                    th = np.array(th)
                    if len(th.shape)==1:
                        lenght = len(th)
                        th = th.reshape((lenght,1))
                    # print(th.shape, len(th.shape))
                    feed = {input_dic['data_{}'.format(mode)]: np.array(dd), outputs[mode-1]: th}
                    # _ = sess.run(opt, feed_dict=feed)
                    A,_ = sess.run([losses[mode-1], opts[mode-1]], feed_dict=feed)
                    t_loss += A
                    t_count += len(th)

            tr_pre = te.prepare(100000, clear=True)

            fp = open(filename, 'w')

            data = tr_pre[0]
            truth = tr_pre[1]
            length = data.shape[0]
            td0 = None
            for c in range(0, length, cfg.batch_size):
                dd = data[c:c + cfg.batch_size]
                th = []
                ids = []
                for d in range(c, c + dd.shape[0]):
                    td = truth[d]
                    th.append(td[4])
                    ids.append((td[0].id, td[1].id, td[2], td[3]))
                th = np.array(th)
                if len(th.shape) == 1:
                    lenght = len(th)
                    th = th.reshape((lenght, 1))
                # print(th.shape, len(th.shape))
                feed = {input_dic['data_{}'.format(mode)]: np.array(dd)}
                # _ = sess.run(opt, feed_dict=feed)
                A = sess.run(xyz[mode - 1], feed_dict=feed)
                sz = A.shape
                diff = np.linalg.norm(A-th, axis=1)

                c=0
                i = ids[c]
                fp.write('{},{},{},{},'.format(i[0], i[1], i[2], i[3]))
                for b in range(sz[1]):
                    fp.write('{},{},'.format(th[c, b], A[c, b]))
                fp.write('{}\n'.format(diff[c]))
                te_loss += np.sum(diff)
                te_count += len(th)
                if td0 is None:
                    td0 = diff
                else:
                    td0 = np.concatenate((td0, diff))

            fp.close()

            print("Err {2} {0:.6f} {1:.6f} {3}".
                        format(t_loss/t_count, te_loss/te_count, a, np.median(np.array(td0))))
            saver.save(sess, cfg.netFile)


if __name__ == '__main__':

    config_file = "config.json"

    cfg = Config(config_file)
    cfg.num_output = list(map(int, cfg.num_output.split(',')))

    iterations = 10

    if len(sys.argv)>1:
        cfg.mode = int(sys.argv[1])

    if len(sys.argv)>2:
        iterations = int(sys.argv[2])

    run(cfg, iterations)
