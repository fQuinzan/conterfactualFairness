import numpy as np
import tensorflow as tf
from tensorflow import keras
import model

# -------------------- cerate dataset:
print("Created dataset: 10000 samples")

# sample variables
n_samples = 10000
def noise(n_samples) : return np.random.normal(0.0, 0.1, size=n_samples)

A = noise(n_samples)
X = np.exp(-0.5 * A * A) * np.sin(2 * A) + noise(n_samples)
Y = 0.5 * np.exp(-0.5 * A * A) * np.sin(2 * A) + 0.1 * X + 0.02 * noise(n_samples)

data_for_A = (A - np.mean(A))/np.var(A)
data_for_X = (X - np.mean(X))/np.var(X)
data_for_Y = (Y - np.mean(Y))/np.var(Y)

# 80% of the sample are used for training, and 20% for testing
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

# -------------------- cerate first model:
print("Train model without HSCIC:")

# define optimizer, loss function, and weights for losses
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.MeanSquaredError()

# fit the first model over training examples (no HSCIC loss)
autoencoder = model.Model(model_loss_weight=1.0,
                          kl_loss_weight=1.0,
                          hscic_loss_weight=0.0)
autoencoder.compile(optimizer=optimizer,
                    loss=loss_function,
                    run_eagerly=True)
autoencoder.fit(x = data_train,
                y = train_Y,
                batch_size=125,
                epochs = 5)
y_predict = autoencoder.predict(data_test)

# fit the second model over training examples (with HSCIC loss)
print("Train model with HSCIC:")

autoencoder = model.Model(model_loss_weight=0.0001,
                          kl_loss_weight=0.0001,
                          hscic_loss_weight=1.0)
autoencoder.compile(optimizer=optimizer,
                    loss=loss_function,
                    run_eagerly=True)
autoencoder.fit(x = data_train,
                y = train_Y,
                batch_size=125,
                epochs = 5)
y_predict_hscic = autoencoder.predict(data_test)

# ----------- plot stuff
import matplotlib.pyplot as plt

plt.figure(1)
plt.scatter(test_A, test_X, c = test_Y, alpha = 1.0, cmap = 'Spectral')
plt.xlabel('A')
plt.ylabel('X')
plt.suptitle('Original Dataset')
plt.figure(2)
plt.scatter(test_A, test_X, c = y_predict, alpha = 1.0, cmap = 'Spectral')
plt.xlabel('A')
plt.ylabel('X')
plt.suptitle('Model Predictions without HSCIC')
plt.figure(3)
plt.scatter(test_A, test_X, c = y_predict_hscic, alpha = 1.0, cmap = 'Spectral')
plt.xlabel('A')
plt.ylabel('X')
plt.suptitle('Model Predictions with HSCIC')
plt.show()
