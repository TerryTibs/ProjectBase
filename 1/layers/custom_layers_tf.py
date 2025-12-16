import tensorflow as tf
from tensorflow.keras import layers, initializers
import numpy as np

# ---------------- NoisyLinear Class (Factorized Gaussian Noise - TF) ---------------- #
class NoisyLinear(layers.Layer):
    """
    Noisy linear layer implementation using Factorized Gaussian Noise. (TensorFlow)
    """
    def __init__(self, units, std_init=0.5, **kwargs):
        super(NoisyLinear, self).__init__(**kwargs)
        self.units = units
        self.std_init = std_init

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)
        self.in_features = input_shape[-1]
        # Learnable parameters (means)
        self.weight_mu = self.add_weight(shape=(self.in_features, self.units), initializer='glorot_uniform', trainable=True, name='weight_mu')
        self.bias_mu = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True, name='bias_mu')
        # Learnable parameters (standard deviations)
        self.weight_sigma = self.add_weight(shape=(self.in_features, self.units), initializer=initializers.Constant(self.std_init / np.sqrt(self.in_features)), trainable=True, name='weight_sigma')
        self.bias_sigma = self.add_weight(shape=(self.units,), initializer=initializers.Constant(self.std_init / np.sqrt(self.units)), trainable=True, name='bias_sigma')
        # Non-learnable noise variables
        self.weight_epsilon = tf.Variable(self._scale_noise((self.in_features, self.units)), trainable=False, name='weight_epsilon')
        self.bias_epsilon = tf.Variable(self._scale_noise((self.units,)), trainable=False, name='bias_epsilon')
        super(NoisyLinear, self).build(input_shape)

    def _scale_noise(self, shape):
        noise = tf.random.normal(shape)
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self):
        epsilon_in = self._scale_noise((self.in_features, 1))
        epsilon_out = self._scale_noise((1, self.units))
        self.weight_epsilon.assign(epsilon_in * epsilon_out)
        self.bias_epsilon.assign(tf.squeeze(epsilon_out, axis=0)) # Squeeze fix

    def call(self, x, training=None):
        if training:
            self.reset_noise()
        noisy_weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        noisy_bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return tf.matmul(x, noisy_weight) + noisy_bias

    def get_config(self):
        config = super(NoisyLinear, self).get_config()
        config.update({'units': self.units, 'std_init': self.std_init})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ---------------- Mean Reducer Class (TF) ---------------- #
class MeanReducer(layers.Layer):
    """
    Simple layer to compute the mean along a specified axis. (TensorFlow)
    """
    def __init__(self, axis=1, keepdims=True, **kwargs):
        super(MeanReducer, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return tf.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super(MeanReducer, self).get_config()
        config.update({'axis': self.axis, 'keepdims': self.keepdims})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)