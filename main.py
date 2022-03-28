import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp


# stuff for importing MINST

import requests
requests.packages.urllib3.disable_warnings()
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class Model(tf.keras.Model) :

    def __init__(self):

        super(Model, self).__init__()

        self.MLP  = keras.Sequential(
            [
                layers.Dense(200,
                             activation="relu"),
                layers.Dense(200,
                             activation="relu"),
                layers.Dense(200,
                             activation="relu"),
                layers.Dense(200,
                             activation="relu"),
                layers.Dense(1),
            ]
        )

    # call function for the model
    def call(self, inputs): return self.MLP(inputs)

    # training step to run on each minibatch
    def train_step(self, data):

        # train the model
        data_train, y_train = data

        with tf.GradientTape() as tape:

            # train the model and get loss
            tape.watch(data_train)
            y_pred = self.call(data_train)

            # get losses
            total_loss = self.loss(y_train, y_pred)

        # update parameters
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss" : total_loss
        }

# make linear model
n_samples = 10000
def noise(n_samples) : return np.random.normal(0.0, 0.1, size=n_samples)

Z = noise(n_samples)
X = np.exp(-0.5 * Z * Z) * np.sin(2 * Z) + noise(n_samples)
Y = 0.5 * np.exp(-0.5 * Z * Z) * np.sin(2 * Z) + 0.1 * X + 0.02 * noise(n_samples)

# define variables
data_for_A = (Z - np.mean(Z))/np.var(Z)
data_for_X = (X - np.mean(X))/np.var(X)
data_for_Y = (Y - np.mean(Y))/np.var(Y)

# training and testing: 80% of the sample are used for training, and 20% for testing
rnd = np.random.uniform(0,1,n_samples)
train_idx = np.array(np.where(rnd <  0.8)).flatten()
test_idx  = np.array(np.where(rnd >= 0.8)).flatten()

train_A = data_for_A[train_idx]
train_X = data_for_X[train_idx]
train_Y = data_for_Y[train_idx]

test_A = data_for_A[test_idx]
test_X = data_for_X[test_idx]
test_Y = data_for_Y[test_idx]

data_train = np.vstack((data_for_X[train_idx], data_for_A[train_idx])).T
data_test  = np.vstack((data_for_X[test_idx], data_for_A[test_idx])).T

# define optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_function  = tf.keras.losses.MeanSquaredError()

# fit the model over training examples
autoencoder = Model()
autoencoder.compile(optimizer=optimizer,
                    loss=loss_function,
                    run_eagerly=True)
autoencoder.fit(x = data_train,
                y = train_Y,
                batch_size=125,
                epochs = 5)
y_predict = autoencoder.predict(data_test)

# ----------- plot stuff

import matplotlib.pyplot as plt

plt.figure(1)
plt.scatter(test_A, test_X, c = test_Y, alpha = 1.0, cmap = 'plasma')
plt.figure(2)
plt.scatter(test_A, test_X, c = y_predict, alpha = 1.0, cmap = 'plasma')

plt.show()
