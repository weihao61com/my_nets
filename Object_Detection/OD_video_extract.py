import pickle
import numpy as np
import cv2
import shutil
import os
import glob
from sortedcontainers import SortedDict
from CV_text_detection import decode


# text detector
model_path = '/media/weihao/DISK0/Object_detection'
# model ="C:\\GITHUB\\learnopencv-master\\TextDetectionEAST\\frozen_east_text_detection.pb"
model = os.path.join(model_path, "frozen_east_text_detection.pb")
outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")
confThreshold = 0.5  # args.thr
nmsThreshold = 0.4  # args.nms


def text_detection(frame):
    max_width = 200
    scale = 1.0

    # Get frame height and width
    height_ = frame.shape[0]
    width_ = frame.shape[1]

    if width_ > max_width:
        scale = max_width / width_

    inpWidth = int(width_ / 32 * scale) * 32
    inpHeight = int(height_ / 32 * scale) * 32

    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)

    # Create a 4D blob from frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

    # Load network
    net = cv2.dnn.readNet(model)
    # Run the model
    net.setInput(blob)
    output = net.forward(outputLayers)

    # Get scores and geometry
    scores = output[0]
    geometry = output[1]
    [boxes, confidences] = decode(scores, geometry, confThreshold)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

    out_boxes = []

    # Apply NMS
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        xmin = 1e6
        ymin = 1e6
        xmax = -1
        ymax = -1
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH
            if vertices[j][0] < xmin:
                xmin = vertices[j][0]
            if vertices[j][0] > xmax:
                xmax = vertices[j][0]
            if vertices[j][1] < ymin:
                ymin = vertices[j][1]
            if vertices[j][1] > ymax:
                ymax = vertices[j][1]
        xmin = int(xmin) - 10
        ymin = int(ymin) - 10
        xmax = int(xmax) + 10
        ymax = int(ymax) + 10
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0

        ig = frame[ymin:ymax, xmin:xmax, :]
        if ig.shape[0] == 0 or ig.shape[1] == 0:
            continue
        out_boxes.append(ig)
    return out_boxes



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
    def __init__(self, f, id, b, mask):
        self.frame = f
        self.box = b
        self.detection_id = id
        self.person_id = None
        self.mask = mask
        self.text_boxes = None


if __name__=='__main__':
    import sys

    filename = '/media/weihao/DISK0/flickr_images/BOLDERBoulder_10K_livestream_at_finish_line.mp4'# '/home/weihao/tmp/out.mp4'
    out_dir = '/media/weihao/DISK0/flickr_images/ps'
    cnt = True

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

            for n1 in range(len(rs)):
                if md['class_ids'][n1] != 1:
                    continue
                p = rs[n1]
                mask = md['masks'][n1]
                val = Detection(a, n1, p, mask)
                w = p[3]-p[1]
                if w < 50:
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

    if not cnt:
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
                pass
                # print(min_w, max_w)

    n = 0
    for r in rm:
        n += 1
        persons.remove(r)

    print("total person", len(persons))

    frame_ids = {}
    pid = 0
    for ps in persons:
        if not cnt:
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
                outname = '{}/{}/I{}_{}.jpg'.format(out_dir, pid, nt, m.detection_id)
                if not os.path.exists(outname):
                    m.text_boxes = text_detection(ig)
                    cv2.imwrite(outname, ig)
                    outname = '{}/{}/I{}_{}.p'.format(out_dir, pid, nt, m.detection_id)
                    with open(outname, 'wb') as fp:
                        pickle.dump(m, fp)
                r = b
                cv2.line(frame, (r[1], r[0]), (r[3], r[0]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(frame, (r[3], r[0]), (r[3], r[2]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(frame, (r[3], r[2]), (r[1], r[2]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(frame, (r[1], r[2]), (r[1], r[0]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "{}".format(pid), (r[1], r[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
            delay = 1
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
