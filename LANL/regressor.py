#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Tensorflow DNNRegressor in Python
# CC-BY-2.0 Paul Balzer
# see: http://www.cbcity.de/deep-learning-tensorflow-dnnregressor-einfach-erklaert
#
TRAINING = True
WITHPLOT = False

# Import Stuff
import tensorflow.contrib.learn as skflow
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np

np.set_printoptions(precision=2)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)
logging.info('Tensorflow %s' % tf.__version__)  # 1.4.1

# This is the magic function which the Deep Neural Network
# has to 'learn' (see http://neuralnetworksanddeeplearning.com/chap4.html)
f = lambda x: 0.2 + 0.4 * x ** 2 + 0.3 * x * np.sin(15 * x) + 0.05 * np.cos(50 * x)

# Generate the 'features'
X = np.linspace(0, 1, 1000001).reshape((-1, 1))

# Generate the 'labels'
y = f(X)

# Train/Test Dataset for Validation
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=2)


# Defining the Tensorflow input functions
# for training
def training_input_fn(batch_size=1):
    return tf.estimator.inputs.numpy_input_fn(
        x={'X': X_train.astype(np.float32)},
        y=y_train.astype(np.float32),
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)


# for test
def test_input_fn():
    return tf.estimator.inputs.numpy_input_fn(
        x={'X': X_test.astype(np.float32)},
        y=y_test.astype(np.float32),
        num_epochs=1,
        shuffle=False)


# Network Design
# --------------
feature_columns = [tf.feature_column.numeric_column('X', shape=(1,))]

STEPS_PER_EPOCH = 100
EPOCHS = 1000
BATCH_SIZE = 100

hidden_layers = [16, 16, 16, 16, 16]
dropout = 0.0

MODEL_PATH = './DNNRegressors/'
for hl in hidden_layers:
    MODEL_PATH += '%s_' % hl
MODEL_PATH += 'D0%s' % (int(dropout * 10))
logging.info('Saving to %s' % MODEL_PATH)

# Validation and Test Configuration
validation_metrics = {"MSE": tf.contrib.metrics.streaming_mean_squared_error}
test_config = skflow.RunConfig(save_checkpoints_steps=100,
                               save_checkpoints_secs=None)

# Building the Network
regressor = skflow.DNNRegressor(feature_columns=feature_columns,
                                label_dimension=1,
                                hidden_units=hidden_layers,
                                model_dir=MODEL_PATH,
                                dropout=dropout,
                                config=test_config)

# Train it
if TRAINING:
    logging.info('Train the DNN Regressor...\n')
    MSEs = []  # for plotting
    STEPS = []  # for plotting

    for epoch in range(EPOCHS + 1):

        # Fit the DNNRegressor (This is where the magic happens!!!)
        regressor.fit(input_fn=training_input_fn(batch_size=BATCH_SIZE),
                      steps=STEPS_PER_EPOCH)
        # Thats it -----------------------------
        # Start Tensorboard in Terminal:
        # 	tensorboard --logdir='./DNNRegressors/'
        # Now open Browser and visit localhost:6006\

        # This is just for fun and educational purpose:
        # Evaluate the DNNRegressor every 10th epoch
        if epoch % 10 == 0:
            eval_dict = regressor.evaluate(input_fn=test_input_fn(),
                                           metrics=validation_metrics)

            print('Epoch %i: %.5f MSE' % (epoch + 1, eval_dict['MSE']))

            if WITHPLOT:
                # Generate a plot for this epoch to see the Network learning
                y_pred = regressor.predict(x={'X': X}, as_iterable=False)

                E = (y.reshape((1, -1)) - y_pred)
                MSE = np.mean(E ** 2.0)
                step = (epoch + 1) * STEPS_PER_EPOCH
                title_string = '%s DNNRegressor after %06d steps (MSE=%.5f)' % \
                               (MODEL_PATH.split('/')[-1], step, MSE)

                MSEs.append(MSE)
                STEPS.append(step)

                fig = plt.figure(figsize=(9, 4))
                ax1 = fig.add_subplot(1, 4, (1, 3))
                ax1.plot(X, y, label='function to predict')
                ax1.plot(X, y_pred, label='DNNRegressor prediction')
                ax1.legend(loc=2)
                ax1.set_title(title_string)
                ax1.set_ylim([0, 1])

                ax2 = fig.add_subplot(1, 4, 4)
                ax2.plot(STEPS, MSEs)
                ax2.set_xlabel('Step')
                ax2.set_xlim([0, EPOCHS * STEPS_PER_EPOCH])
                ax2.set_ylabel('Mean Square Error')
                ax2.set_ylim([0, 0.01])

                plt.tight_layout()
                plt.savefig(MODEL_PATH + '_%05d.png' % (epoch + 1), dpi=72)
                logging.info('Saved %s' % MODEL_PATH + '_%05d.png' % (epoch + 1))

                plt.close()

# Now it's trained. We can try to predict some values.
else:
    logging.info('No training today, just prediction')
    try:
        # Prediction
        X_pred = np.linspace(0, 1, 11)
        y_pred = regressor.predict(x={'X': X_pred}, as_iterable=False)
        print(y_pred)

        # Get trained values out of the Network
        for variable_name in regressor.get_variable_names():
            if str(variable_name).startswith('dnn/hiddenlayer') and \
                    (str(variable_name).endswith('weights') or \
                     str(variable_name).endswith('biases')):
                print('\n%s:' % variable_name)
                weights = regressor.get_variable_value(variable_name)
                print(weights)
                print('size: %i' % weights.size)

        # Final Plot
        if WITHPLOT:
            plt.plot(X, y, label='function to predict')
            plt.plot(X, regressor.predict(x={'X': X}, as_iterable=False), \
                     label='DNNRegressor prediction')
            plt.legend(loc=2)
            plt.ylim([0, 1])
            plt.title('%s DNNRegressor' % MODEL_PATH.split('/')[-1])
            plt.tight_layout()
            plt.savefig(MODEL_PATH + '.png', dpi=72)
            plt.close()
    except:
        logging.Error('Prediction failed! Maybe first train a model?')
