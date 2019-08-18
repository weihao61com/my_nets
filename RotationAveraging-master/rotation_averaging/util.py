
import scipy
import scipy.linalg
import numpy as np

def fix_matrix(m):
	m0 = m.transpose().dot(m)
	sq = scipy.linalg.inv(scipy.linalg.sqrtm(m0))
	if sq.dtype != m.dtype:
		#print(m0)
		#print(sq)
		sq = np.eye(3)
		#print(sq)
		#raise Exception()
	return m.dot(sq)