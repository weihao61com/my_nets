import pickle
from LANL_Utils import l_utils

fnm = '/home/weihao/Projects/p_files/L_0.p'
with open(fnm, 'r') as fp:
    d = pickle.load(fp)

nt = 0
for a in d:
    if nt%100==0:
        str = l_utils.csv_line(a[1])
        print '{},{}'.format(a[0], str)
    nt += 1

