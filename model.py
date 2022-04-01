import HSCIC as hs
import tensorflow as tf
from tensorflow import keras
from keras import layers

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

    def __init__(self,
                 model_loss_weight = 1.0,
                 kl_loss_weight = 1.0,
                 hscic_loss_weight = 1.0):

        super(Model, self).__init__()

        # define the encoder
        encoder_inputs = keras.Input(shape=(2,))

        x = layers.Dense(200, activation="relu")(encoder_inputs)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(1000, activation="relu")(x)
        x = layers.Dense(200, activation="relu")(x)

        latent_mean    = layers.Dense(2, name="mean")(x)
        latent_log_var = layers.Dense(2, name="log_var")(x)
        latent_sample  = self.gaussian_sample([latent_mean, latent_log_var])

        self.encoder = keras.Model(encoder_inputs,
                                   [latent_mean, latent_log_var, latent_sample],
                                   name="encoder")

        # define the decoder
        self.decoder = keras.Sequential(
            [
                layers.Dense(200,activation="relu"),
                layers.Dense(200,activation="relu"),
                layers.Dense(200,activation="relu"),
                layers.Dense(200,activation="relu"),
                layers.Dense(1),
            ]
        )

        # define additional losses
        self.HSCIC = hs.HSCIC()

        # weights for loss functions
        self.model_loss_weight = model_loss_weight
        self.kl_loss_weight    = kl_loss_weight
        self.hscic_loss_weight = hscic_loss_weight

        # define loss trackers
        self.model_loss_tracker = keras.metrics.Mean(name="model_loss")
        self.kl_loss_tracker    = keras.metrics.Mean(name="kl_loss")
        self.hscic_loss_tracker = keras.metrics.Mean(name="hscic_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.model_loss_tracker,
            self.kl_loss_tracker,
            self.hscic_loss_tracker,
            self.total_loss_tracker
        ]

    # samle from Gaussian distribution
    def gaussian_sample(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim   = tf.shape(mean)[1]
        eps   = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * eps

    # call function for the model
    def call(self, inputs):
        _, _, latent_sample = self.encoder(inputs)
        output = self.decoder(latent_sample)
        return output

    # training step to run on each minibatch
    def train_step(self, data):

        # train the model
        inputs, y_train = data

        with tf.GradientTape() as tape:

            # train the model and get loss
            latent_mean, latent_log_var, latent_sample = self.encoder(inputs)
            y_pred = self.decoder(latent_sample)

            # get model losse
            model_loss = self.loss(y_train, y_pred)

            # get HSCIC loss
            hscic_loss = self.HSCIC(Y = y_pred,
                                    A = inputs[:, 1],
                                    X = inputs[:, 0],
                                    H = latent_sample)

            # get KL loss
            kl_loss = -0.5 * (1 + latent_log_var - tf.math.square(latent_mean) - tf.math.exp(latent_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # get weighted total loss
            total_loss = self.model_loss_weight * model_loss
            total_loss += self.kl_loss_weight * kl_loss
            total_loss += self.hscic_loss_weight * hscic_loss

        # update parameters
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # update trakcer
        self.model_loss_tracker.update_state(model_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.hscic_loss_tracker.update_state(hscic_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "model_loss" : self.model_loss_tracker.result(),
            "kl_loss"    : self.kl_loss_tracker.result(),
            "hscic_loss" : self.hscic_loss_tracker.result(),
            "total_loss" : self.total_loss_tracker.result()
        }