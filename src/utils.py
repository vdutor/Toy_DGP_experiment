import numpy as np
import tensorflow as tf
from gpflow import settings
from scipy.stats import norm


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def xavier_weights(input_dim: int, output_dim: int) -> np.ndarray:
    """
    Xavier initialization for the weights of NN layer
    :return: np.array
        weight matrix of shape `input_dim` x `output_dim`,
        where each element is drawn i.i.d. from N(0, sqrt(2. / (in + out)))

    See:
       Xavier Glorot and Yoshua Bengio (2010):
       Understanding the difficulty of training deep feedforward neural networks.
       International conference on artificial intelligence and statistics.
    """

    xavier_std = (2.0 / (input_dim + output_dim)) ** 0.5
    return np.random.randn(input_dim, output_dim) * xavier_std


def phi(d, iterations=20):
    # http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    x = 1.0
    for i in range(iterations):
        x = x - (pow(x, d + 1) - x - 1) / ((d + 1) * pow(x, d) - 1)
    return x


def prand_np(shape):
    # modified from http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

    # Number of dimensions.
    d = shape[-1]

    # number of required points
    n = np.prod(shape[:-1])

    g = phi(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = pow(1 / g, j + 1) % 1
    z = np.zeros((n, d))

    # This number can be any real number.
    # Default setting typically 0 or 0.5
    seed = np.random.uniform()
    for i in range(n):
        z[i] = (seed + alpha * (i + 1)) % 1

    return norm.ppf(z.reshape(shape))


def prand(shape):
    # TODO native tensorflow code
    return tf.py_func(prand_np, [shape], settings.float_type)


def random_normal(shape, use_prand=False):
    if (
        False
    ):  # use_prand: # TODO just use vanilla random numbers now, but investigate this
        return prand(shape)
    else:
        return tf.random_normal(shape, dtype=settings.float_type)
