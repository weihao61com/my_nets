import pickle
import numpy as np
import cv2
import shutil
import os
import glob
from sortedcontainers import SortedDict


def read_p_file(filename):
    ss = SortedDict()
    for f in glob.glob(filename[:-4] + '*.p'):
        basename = os.path.basename(f)[:-2]
        id = basename.split('_')[-1]
        ss[int(id)] = f

    return ss


def cal_diff(b10, b20, b12, b22):
    if b22 < b10:
        return 0
    if b12 < b20:
        return 0
    return min(b22, b12) - max(b20, b10)


def cal_overlap(b1, b2):
    dx = cal_diff(b1[0], b2[0], b1[2], b2[2])
    if dx == 0:
        return 0
    dy = cal_diff(b1[1], b2[1], b1[3], b2[3])
    if dy == 0:
        return 0

    s = dx*dy
    s1 = (b1[3]-b1[1])*(b1[2]-b1[0])
    s2 = (b2[3]-b2[1])*(b2[2]-b2[0])
    return float(s)/(s1+s2-s)


class Detection:
    def __init__(self, f, b):
        self.frame = f
        self.box = b
        self.persion_id = None

if __name__=='__main__':
    import sys

    filename = sys.argv[1] #'/media/weihao/DISK0/flickr_images/out.mp4'# '/home/weihao/tmp/out.mp4'
    out_dir = sys.argv[2] #'/home/weihao/tmp/ps'


    persons = []
    nps = {}
    pid = 0

    p_files = read_p_file(filename)
    for p in p_files:
        p_file = p_files[p]
        print(p, p_file)
        with open(p_file, 'rb') as fp:
            dm = pickle.load(fp)

        for a in dm:
            md = dm[a]
            rs = md['rois']

            n1 = 0
            for p in rs:

                val = Detection(a, p)
                w = p[3]-p[1]
                if w<50:
                    continue
                n2 = 0
                match = False
                for b in nps:
                    r = nps[b][-1].box
                    o = cal_overlap(p, r)
                    if o>0.4:
                        match = True
                        nps[b].append(val)
                        break
                        #print(a, n1, n2, o)
                    n2 += 1
                if not match:
                    nps[pid] = [val]
                    pid += 1
                n1 += 1

            i1 = []
            for b in nps:
                r = nps[b][-1].frame
                if r!=a:
                    i1.append(b)
                    persons.append(nps[b])

            for i in i1:
                del nps[i]

    for n in nps:
        persons.append(nps[n])

    print("total person", len(persons))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    rm = []
    for p in persons:
        if len(p)<4:
            rm.append(p)
        else:
            max_w=0
            min_w=1e6
            for a in p:
                b = a.box
                w = b[3]-b[1]
                if w>max_w:
                    max_w = w
                if w<min_w:
                    min_w = w
            if max_w<80:
                rm.append(p)
            elif min_w>100:
                rm.append(p)
            else:
                print(min_w, max_w)

    n = 0
    for r in rm:
        n += 1
        persons.remove(r)

    print("total person", len(persons))

    frame_ids = {}
    pid = 0
    for ps in persons:
        os.mkdir(out_dir + '/{}'.format(pid))

        for p in ps:
            if p.frame not in frame_ids:
                frame_ids[p.frame] = []
            p.person_id = pid
            frame_ids[p.frame].append(p)
        pid += 1


    cap = cv2.VideoCapture(filename)
    nt = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        delay = 1
        if nt in frame_ids:

            clone = frame.copy()
            for m in frame_ids[nt]:
                pid = m.person_id
                b = m.box
                ig = clone[b[0]:b[2], b[1]:b[3], :]
                outname = '{}/{}/I{}.jpg'.format(out_dir, pid, nt)
                cv2.imwrite(outname, ig)
                r = b
                cv2.line(frame, (r[1], r[0]), (r[3], r[0]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(frame, (r[3], r[0]), (r[3], r[2]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(frame, (r[3], r[2]), (r[1], r[2]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(frame, (r[1], r[2]), (r[1], r[0]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "{}".format(pid), (r[1], r[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
            delay = 16
            #
            # if nt not in dm:
            #      break
            #
            # md = dm[nt]
            # for r in md['rois']:
            #     if r[3]-r[1]>0:
            #         cv2.line(frame, (r[1], r[0]), (r[3], r[0]), (0, 0, 255), 1, cv2.LINE_AA)
            #         cv2.line(frame, (r[3], r[0]), (r[3], r[2]), (0, 0, 255), 1, cv2.LINE_AA)
            #         cv2.line(frame, (r[3], r[2]), (r[1], r[2]), (0, 0, 255), 1, cv2.LINE_AA)
            #         cv2.line(frame, (r[1], r[2]), (r[1], r[0]), (0, 0, 255), 1, cv2.LINE_AA)


        cv2.imshow('t',frame)
        k = cv2.waitKey(delay) & 0xff
        if k == 27:
            break
        nt += 1
