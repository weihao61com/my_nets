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
from last1 import P1Net1, DataSet, transfor_T


if __name__ == '__main__':

    config_file = "config.json"

    if len(sys.argv) > 2:
        config_file = sys.argv[2]

    cfg = Config(config_file)

    print("LR {} num_out {} mode {}".format(cfg.lr, cfg.num_output, cfg.mode))

    input_dic = {}
    output = []
    num_output = cfg.num_output[0]
    feature_len = 13 - num_output
    output = tf.compat.v1.placeholder(tf.float32, [None, num_output])
    input_dic['data'] =  tf.compat.v1.placeholder(tf.float32, [None, feature_len])

    net = P1Net1(input_dic)
    net.real_setup(cfg.nodes[0], num_output)

    xy = net.layers['output']

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    avg_file = Utils.avg_file_name(cfg.netFile)

    if hasattr(cfg, 'inter_file'):
        tr = DataSet(None,  cfg, cfg.inter_file)
    else:
        tr = DataSet(cfg.tr_data[0], cfg)
        tr.subtract_avg(avg_file, save_im=True)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver.restore(sess, cfg.netFile)

        t00 = datetime.datetime.now()

        tr_pre = tr.prepare( clear=True)

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

                if t[1].id not in rst_dic:
                    rst_dic[t[1].id] = {}
                if t[3] not in rst_dic[t[1].id]:
                    rst_dic[t[1].id][t[3]] = []
                rst_dic[t[1].id][t[3]].append(xyz)


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

                xyzs = xyzs[:, 0, :]
                mm = np.mean(xyzs, axis=0)

                rst_dic[image_id][point_id] = (mm, xyz0, count)

        threshold = 0.03
        clu_dic = {}
        for c in range(0, length, cfg.batch_size):
            dx = data[c:c + cfg.batch_size]
            dd = []
            th = []
            for d in range(c, c+dx.shape[0]):
                t = truth[d]
                th1 = rst_dic[t[0].id][t[2]]
                th2 = rst_dic[t[1].id][t[3]]

                dist = np.linalg.norm(np.array(th1[0]-th2[0]))
                if dist<threshold:

                    img_id = t[0].id
                    pnt_id = t[2]
                    if img_id not in clu_dic:
                        clu_dic[img_id] = {}
                    if pnt_id not in clu_dic[img_id] :
                        clu_dic[img_id][pnt_id] = set()
                    clu_dic[img_id][pnt_id].add((t[1].id, t[3]))

                    img_id = t[1].id
                    pnt_id = t[3]
                    if img_id not in clu_dic:
                        clu_dic[img_id] = {}
                    if pnt_id not in clu_dic[img_id] :
                        clu_dic[img_id][pnt_id] = set()
                    clu_dic[img_id][pnt_id].add((t[0].id, t[2]))


        clusters = []
        #for id in clu_dic:
        #    print(clu_dic[id])
        while len(clu_dic)>0:
            ids = list(clu_dic.keys())
            id = ids[0]
            p_ids = clu_dic[id]
            del clu_dic[id]
            #print(p_ids)
            for p in p_ids:
                p_list = set()
                p_list.add((id, p))
                new_list = p_ids[p]
                while len(new_list)>0:
                    for l in new_list:
                        p_list.add(l)
                    new_list = []
                    for a in p_list:
                        if a[0] in clu_dic:
                            # print(a)
                            new_list = new_list + list(clu_dic[a[0]][a[1]])
                            clu_dic[a[0]][a[1]] = []
                if len(p_list)>1:
                    #if len(p_list)>30:
                    #    print('{}'.format(p_list))
                    clusters.append(p_list)

        print("Cluster: {}".format(len(clusters)))
        filename = 'c:/Projects/tmp/cloud.p'
        with open(filename, 'wb') as fp:
            pickle.dump(clusters, fp)