import numpy as np
import sys
from random import randint, random

from cv_location import VisualOdometry2
from imagery_utils import SiftFeature, pose_realign
from pose_ana import *
import datetime
import pickle
import matplotlib.pyplot as plt

from scipy.optimize import minimize


def errors(para, X, Y):
    e1 = X[:, :3].dot(para) - Y[:,0]
    e2 = X[:, 3:6].dot(para) - Y[:,1]
    e3 = X[:, 6:].dot(para) - Y[:,2]
    return e1, e2, e3


def objective_function(para, X, Y):
    e1, e2, e3 = errors(para, X, Y)
    rp = np.linalg.norm(para)
    error = e1*e1+ e2*e2 + e3*e3 + 100*(rp-1)*(rp-1)
    return sum(error)


this_file_path = os.path.dirname(os.path.realpath(__file__))
HOME = '{}/../../'.format(this_file_path)

#HOME = '/home/weihao/Projects'

range2 = 1
range3 = 0
range1 = -range2
#key = 'heads'
#mode = 'Train'
key = '02'
mode = 'Test'

if len(sys.argv)>1:
    key = sys.argv[1]
if len(sys.argv)>2:
    mode = sys.argv[2]

print key, mode

#location = '/home/weihao/Projects/cambridge/OldHospital'
#pose_file = 'dataset_train.txt'
#poses_dic, cam = load_cambridge_poses(location, pose_file)

if key.startswith('0'):
    location = '{}/datasets/kitty'.format(HOME)
    poses_dic, cam = load_kitty_poses(location, key)
    key = 'kitty_{}'.format(key)
else:
    location = "{}/datasets/indoors/{}".format(HOME, key) #office" #heads
    poses_dic, cam = load_indoor_7_poses(location, "{}Split.txt".format(mode))

filename = '{}p_files/{}_{}_cv_s{}_2.p'.format(HOME, key, mode, range2)
output_file = '{}tmp/{}_{}.csv'.format(HOME, key, mode)
print location
print filename
print output_file

x = []
y = []
r0 = []
direction = [0,0,1]

for p in poses_dic:
    poses = poses_dic[p]
    print p, len(poses)
    pre_pose = None
    for id in poses:
        pose = poses[id]
        if pre_pose is not None:
            dt = pose.tran - pre_pose.tran
            rr = np.linalg.norm(dt)
            if rr>0.001:
                m = pre_pose.m3x3
                r0.append(rr)
                y.append(dt/rr)
                x.append(m)
            #print pose

        pre_pose = pose

X = []
with open(output_file, 'w') as fp:
    for a in range(len(x)):
        dt = y[a]
        dx = x[a].dot(direction)
        X.append(x[a].reshape(9))

        fp.write('{},{},{},{},{},{}\n'.format(
            dt[0],dt[1],dt[2],dx[0],dx[1],dx[2]))


X = np.array(X)
y = np.array(y)
beta_init = np.array([.3,.3,.4])
result = minimize(objective_function, beta_init, args=(X,y),
                  method='BFGS', options={'maxiter': 500})

x = result.x
print x, result.fun, np.linalg.norm(x)

error = objective_function(x, X, y)
print y.shape, np.sqrt(error/y.shape[0]), np.mean(r0)

bins = (np.array(range(100))-50)/500.0
e1, e2, e3 = errors(x, X, y)
plt.subplot(2, 2, 1)
plt.hist(e1, bins)
plt.subplot(2, 2, 2)
plt.hist(e2, bins)
plt.subplot(2, 2, 3)
plt.hist(e3, bins)
plt.subplot(2, 2, 4)
plt.hist(r0, 100)
plt.show()
# plt.plot(data[st:nb, 1], data[st:nb, 2], '.')
