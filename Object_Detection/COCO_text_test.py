# matplotlib inline
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import shutil
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
# from blob_detection import mser_detecter
import cv2
import pickle

from pascal_voc_writer import Writer

import sys
sys.path.append('/media/weihao/DISK0/Object_detection/COCO/coco-text-master')
import coco_text

def has_digits(s):
    for a in range(len(s)):
        if s[a].isnumeric():
            return True
    return False

string_tag = 'utf8_string'

ct = coco_text.COCO_Text('/media/weihao/DISK0/Object_detection/COCO/COCO_Text.json')
ct.info()

# imgs = ct.getImgIds(imgIds=ct.train,
#                     catIds=[('legibility','legible'),('class','machine printed')])
# anns = ct.getAnnIds(imgIds=ct.val,
#                     catIds=[('legibility','legible'),('class','machine printed')],
#                     areaRng=[1000,200000])
dataDir='/home/weihao/Downloads'
dataType='train2017'
output_dir = '/home/weihao/tmp/data/'
if output_dir is not None:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

# get all images containing at least one instance of legible text
# imgIds = ct.getImgIds(imgIds=ct.train,
#                     catIds=[('legibility','legible')])
imgIds = ct.getImgIds(imgIds=ct.train,
                      catIds=[('legibility','legible'),('class','machine printed')])

print('Total images', len(imgIds))

# detector = mser_detecter(min_=100)

mini_area = 1000
tags = {}
nids = set()
nt = 0
for id in  imgIds:
    annIds = ct.getAnnIds(imgIds=id)
    anns = ct.loadAnns(annIds)
    for a in anns:
        if string_tag in a:
            s = a[string_tag].strip()
            if len(s)==1 and a['area']>mini_area and a['class']=='machine printed':
                # if has_digits(a[string_tag]):
                nids.add(id)
                if a[string_tag] not in tags:
                    tags[a[string_tag]] = nt
                    nt += 1

imgIds = nids
print('Total digits images', len(imgIds))
print('Total labels', len(tags))

with open(os.path.join(output_dir, 'classes.txt'), 'w') as fp:
    for t in tags:
        fp.write('{}\n'.format(t))

nt = 0
for id in imgIds:
    # imgIds = [131703]
    # pick one at random
    img = ct.loadImgs(id)[0]
    metadata = []
    file_name = img['file_name'].split('_')[-1]
    image_name = '%s/%s/%s'%(dataDir,dataType,file_name)
    print(img['id'], image_name)
    I = io.imread(image_name)

    # plt.figure()
    # plt.imshow(I)
    sz = I.shape[:2]
    annIds = ct.getAnnIds(imgIds=img['id'])
    anns = ct.loadAnns(annIds)
    # ans = []
    # nn = []

    basename = os.path.basename(image_name)[:-4]
    writer = Writer(image_name, sz[0], sz[1])
    for a in anns:
        if string_tag in a:
            s = a[string_tag].strip()
            if len(s) == 1 and a['area']>mini_area:
                # if has_digits(a[string_tag]):
                print(a[string_tag], int(a['area']), np.array(a['bbox']).astype(int))
                b = a['bbox']
                d = (tags[s], (b[0]+b[2]/2)/sz[1], (b[1]+b[3]/2)/sz[0], b[2]/sz[1], b[3]/sz[0])
                metadata.append(d)
                # nn.append(a)
                writer.addObject('tag', b[0], b[1], b[0]+b[2], b[1]+b[3])

    if len(metadata)>0:
        if output_dir is not None:
            image_name = '{}/{}.jpg'.format(output_dir, basename)
            txt_name = '{}/{}.txt'.format(output_dir, basename)
            writer.save('{}/{}.xml'.format(output_dir, basename))

            # sz = I.shape
            # s0 = int(round(sz[0]/16.0))*16
            # s1 = int(round(sz[1]/16.0))*16
            # I = cv2.resize(I, (s1, s0))
            # mask = cv2.resize(mask, (s0, s1))
            cv2.imwrite(image_name, cv2.cvtColor(I, cv2.COLOR_RGB2BGR))
            with open(txt_name, 'w') as fp:
                for m in metadata:
                    fp.write("{} {} {} {} {}\n".format(m[0], m[1], m[2], m[3], m[4]))
            nt += 1

    # print('Boxes:', len(ans))
    # ct.showAnns(nn)
    # plt.show()