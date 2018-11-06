#!/usr/bin/env python

import tensorflow
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

num_inputs = 784
num_examples = 10000
true_w = [2, -3.4]
true_b = 4.2
X = np.random.randn(num_examples, num_inputs)
#X = np.random.randn(num_examples, num_examples)
Y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
Y += .01 * np.random.randn(len(Y))
#y += 0.01 * np.random.randn(y.shape[0])


# Construct data iterator.
def data_iter(batch_size):
    idx = list(range(num_examples))
    np.random.shuffle(idx)
    for batch_i, i in enumerate(range(0, num_examples, batch_size)):
        j = np.array(idx[i: min(i + batch_size, num_examples)])
        yield batch_i, X[j], Y[j]


def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))


if __name__ == '__main__':

    import datetime
    a_0 = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    middle = 30
    w_1 = tf.Variable(tf.truncated_normal([784, middle]))
    b_1 = tf.Variable(tf.truncated_normal([1, middle]))
    w_2 = tf.Variable(tf.truncated_normal([middle, 10]))
    b_2 = tf.Variable(tf.truncated_normal([1, 10]))

    z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
    a_1 = sigma(z_1)
    z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
    a_2 = sigma(z_2)

    diff = tf.subtract(a_2, y)

    d_z_2 = tf.multiply(diff, sigmaprime(z_2))
    d_b_2 = d_z_2
    d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

    d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
    d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
    d_b_1 = d_z_1
    d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)

    eta = tf.constant(0.001)
    # step = [
    #     tf.assign(w_1,
    #               tf.subtract(w_1, tf.multiply(eta, d_w_1)))
    #     , tf.assign(b_1,
    #                 tf.subtract(b_1, tf.multiply(eta,
    #                                              tf.reduce_mean(d_b_1, axis=[0]))))
    #     , tf.assign(w_2,
    #                 tf.subtract(w_2, tf.multiply(eta, d_w_2)))
    #     , tf.assign(b_2,
    #                 tf.subtract(b_2, tf.multiply(eta,
    #                                              tf.reduce_mean(d_b_2, axis=[0]))))
    # ]
    cost = tf.multiply(diff, diff)
    step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    
    acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
    acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    dt = 0
    for i in xrange(500000):
        #batch_i, batch_xs, batch_ys = next(di)
        batch_xs, batch_ys = mnist.train.next_batch(5000)
        t0 = datetime.datetime.now()
        sess.run(step, feed_dict={a_0: batch_xs,
                                  y: batch_ys})
        dt += (datetime.datetime.now() - t0).microseconds
        if i % 100 == 0:
            res = sess.run(acct_res, feed_dict=
            {a_0: mnist.test.images[:1000],
             y: mnist.test.labels[:1000]})
            print i, res, dt*1e-6
            dt = 0
