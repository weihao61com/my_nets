import glob
import os
import cv2


class Record:
    def __init__(self, eles):
        self.id = eles[1]
        self.folder = eles[0]
        self.file = eles[2]


def read_records(filenaem):
    records = {}
    with open(filenaem, 'r') as fp:
        for line in fp.readlines():
            strs = line[:-1].split(' ')
            key = strs[1]
            if key != "None":
                if key in records:
                    records[key].append(Record(strs))
                else:
                    records[strs[1]] = [Record(strs)]

    print('Total records', len(records))
    return records

def play(rs, vid):
    files = []
    for r in rs:
        files = files + glob.glob(os.path.join(r.folder, '*.jpg'))
    ids = []
    for f in files:
        basename = os.path.basename(f).split('_')
        ids.append(int(basename[0][1:]))
    v1 = min(ids)
    v2 = max(ids)
    print(v1, v2)
    dv = v1-v2
    screen_capture = cv2.VideoCapture(filename) #init videocapture
    while v1>dv:
        v1 -= 1
        if v1>0:
            ret = screen_capture.grab()  # grab frame but dont process it
        else:
            hasFrame, frame = screen_capture.read()
            cv2.imshow('t',frame)
            k = cv2.waitKey(200) & 0xff
            if k == 27:
                break

if __name__=='__main__':
    import sys

    filename = '/media/weihao/DISK0/flickr_images/BOLDERBoulder_10K_livestream_at_finish_line.mp4'# '/home/weihao/tmp/out.mp4'
    out_dir = '/media/weihao/DISK0/flickr_images/ps'

    rst_file = out_dir + ".txt"
    records = read_records(rst_file)
    keys = list(records.keys())

    while True:
        text = input("ID=")  # Python 3
        if text.isnumeric():
            id = int(text)
            if id < len(keys):
                text = keys[id]
        if text not in records:
            print(text, 'not found')
            break
        print(text)
        play(records[text], filename)


