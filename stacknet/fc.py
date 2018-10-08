#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from time import time
from o2_load import *

# Download the dataset
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

dim_intput = 100
dim_output = 2
nl1 = 1200
nl2 = 100
lr = 0.0001
batch_size = 500

# correct labels
y_ = tf.placeholder(tf.float32, [None, dim_output])

# input data
x = tf.placeholder(tf.float32, [None, dim_intput])

# build the network
#keep_prob_input = tf.placeholder(tf.float32)
#x_drop = tf.nn.dropout(x, keep_prob=keep_prob_input)

W_fc1 = weight_variable([dim_intput, nl1])
b_fc1 = bias_variable([nl1])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([nl1, nl2])
b_fc2 = bias_variable([nl2])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

#h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([nl2, dim_output])
b_fc3 = bias_variable([dim_output])
y = tf.matmul(h_fc2, W_fc3) + b_fc3

# define the loss function
cross_entropy = tf.reduce_sum(tf.square(tf.subtract(y, y_)))

# define training step and accuracy
train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).\
    minimize(cross_entropy)

# train_step = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).
# minimize(cross_entropy)

# create a saver
saver = tf.train.Saver()

# initialize the graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# train
print("Startin Burn-In...")
saver.save(sess, '/home/weihao/posenet/my_nets/stacknet/fc_best')
import datetime
t0 = datetime.datetime.now()
tr_data = '/home/weihao/posenet/my_nets/stacknet/c2_tr.p'
tr = load_data(tr_data, 500)
data_gen = gen_data_batch(tr, batch_size)

for i in range(700000):
    input_a, output_a = next(data_gen)

    if i % 6000 == 0:
        train_accuracy = \
            sess.run(cross_entropy,
                     feed_dict={x: input_a, y_: output_a})
        t1 = datetime.datetime.now()
        print("step {}, training accuracy {} {}" .format(i, train_accuracy, t1-t0))
        t0 = t1

    sess.run(train_step,
             feed_dict={x: input_a, y_: output_a})
