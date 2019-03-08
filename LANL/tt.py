import pickle
from LANL_Utils import l_utils

fnm = '/home/weihao/Projects/p_files/L_0.p'
with open(fnm, 'r') as fp:
    d = pickle.load(fp)

d0 = d[0]
d1 = d[1]
for a in range(len(d0)):
    str = l_utils.csv_line(d1[a, :])
    print '{},{},{}'.format(a, d0[a], str)


