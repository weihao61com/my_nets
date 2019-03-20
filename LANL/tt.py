import pickle
from LANL_Utils import l_utils
#import matplotlib as plt
import matplotlib.pyplot as plt


#
# fnm = '/home/weihao/Projects/p_files/L_0.p'
# with open(fnm, 'r') as fp:
#     d = pickle.load(fp)
#
# nt = 0
# for a in d:
#     if nt%100==0:
#         str = l_utils.csv_line(a[1])
#         print '{},{}'.format(a[0], str)
#     nt += 1


filename = '/home/weihao/Projects/tmp/rst.csv'
x = []
y = []
with open(filename, 'r') as fp:
    for line in fp.readlines()[1:]:
        str = line.split(',')
        x.append(float(str[1]))
        #y.append(float(str[3]))


n, bins, patches = plt.hist(x, 100)
#print n, bins
plt.show()
