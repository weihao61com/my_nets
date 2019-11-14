import pickle
import numpy as np

filename = '/home/weihao/tmp/out.mp4'

with open(filename + '.p0', 'rb') as fp:
    dm = pickle.load(fp)

for b in dm:
    m = dm[b]
    msk = []
    regions = m['rois']
    for a in range(len(regions)):
        box = regions[a]
        mask = m['masks'][:, :, a]
        submask = mask[box[0]:box[2], box[1]:box[3]]
        msk.append(submask)
    dm[b]['masks'] = msk

with open(filename + '.p0', 'wb') as fp:
    pickle.dump(dm, fp)