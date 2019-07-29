import numpy
import numpy.random
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import unittest
# import compare
import logging
import rotation_averaging
import rotation_averaging.util
import rotation_averaging.so3
import rotation_averaging.compare
import rotation_averaging.graph
import rotation_averaging.algorithms
import scipy.io
import pickle


def load_data(filename):
	if filename.endswith("mat"):
		return scipy.io.loadmat(filename)
	else:
		with open(filename, 'rb') as fp:
			return pickle.load(fp)


def main():

	import sys
	logging.basicConfig(level=logging.INFO)
	filename = "/home/weihao/tmp/test.p" #data/notredame/Notredame.mat"

	if len(sys.argv) > 1:
		filename = sys.argv[1]

	a = load_data(filename)

	I = a['I'].transpose()
	Rgt = a['Rgt']
	RR = a['RR']
	# print a
	# I0 = I[0]
	# print(RR[:,:,0])
	# print(Rgt[:,:,I0[0]])
	# print(Rgt[:,:,I0[1]])

	testi = rotation_averaging.util.fix_matrix(numpy.array([[1.0, 0.1, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
	# print testi
	# print scipy.linalg.norm(numpy.identity(3) - testi.dot(testi.transpose()))

	mv = numpy.min(I)
	indices = [(I[i, 0]-mv, I[i, 1]-mv) for i in range(I.shape[0])]
	global_rotations = [rotation_averaging.util.fix_matrix(Rgt[:,:,i]) for i in range(Rgt.shape[2])]
	relative_rotations = [rotation_averaging.util.fix_matrix(RR[:,:,i]) for i in range(RR.shape[2])]

	max_err = 0.0
	for mat in global_rotations:
		max_err = max(max_err, scipy.linalg.norm(numpy.identity(3) - mat.dot(mat.transpose())))

	print(max_err)

	max_err = 0.0
	for mat in relative_rotations:
		max_err = max(max_err, scipy.linalg.norm(numpy.identity(3) - mat.dot(mat.transpose())))

	# print(max_err)
	# print(global_rotations[0])
	# print(rotation_averaging.so3.matrix_to_axis_angle(global_rotations[0]))
	# print(rotation_averaging.so3.matrix_to_axis_angle(relative_rotations[0]))
	# graphi = graph.generate_random_so3_graph(200, completeness=0.5, noise=0.2)
	# global_rotations = graphi[0]
	# relative_rotations = graphi[1]
	# indices = graphi[2]

	rt = rotation_averaging.compare.compare_global_rotation_to_graph(global_rotations, relative_rotations, indices, plot=True)
	print(rt)
	initial_guess = rotation_averaging.graph.compute_initial_guess(len(global_rotations), relative_rotations, indices)

	max_err = 0.0
	for mat in initial_guess:
		max_err = max(max_err, scipy.linalg.norm(numpy.identity(3) - mat.dot(mat.transpose())))

	print(max_err)

	rt = rotation_averaging.compare.compare_global_rotation_to_graph(initial_guess, relative_rotations, indices, plot=True)
	print(rt)

	# print initial_guess[0].shape, len(initial_guess)
	# solution = rotation_averaging.algorithms.L1RA(len(global_rotations), relative_rotations, indices, initial_guess)
	# rotation_averaging.compare.compare_global_rotation_to_graph(solution, relative_rotations, indices, plot=True)
	solution = rotation_averaging.algorithms.IRLS(len(global_rotations), relative_rotations, indices, initial_guess)
	rt=rotation_averaging.compare.compare_global_rotation_to_graph(solution, relative_rotations, indices, plot=True)
	print(rt)

if __name__ == '__main__':
	main()