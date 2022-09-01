from typing import Optional

import numpy as np
import tensorflow as tf
from gpflow import Param, params_as_tensors, settings

from ..utils import xavier_weights
from .layers import BaseLayer


class LinearLayer(BaseLayer):
    """
    Performs a deterministic linear transformation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
    ):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if weight is None:
            weight = xavier_weights(input_dim, output_dim)

        if bias is None:
            bias = np.zeros((output_dim,))

        self.weight = Param(weight)
        self.bias = Param(bias)

    @params_as_tensors
    def propagate(self, X, sampling=True, **kwargs):
        if not sampling:
            raise ValueError(
                "We can only sample from a single " "layer multi-perceptron."
            )
        else:
            # matmul doesn't broadcast
            extra_dims = tf.shape(X)[:-2]
            shape = tf.concat([extra_dims, [1, 1]], 0)
            zeros = tf.zeros(shape, dtype=settings.float_type)
            return tf.matmul(X, zeros + self.weight) + self.bias

    def KL(self):
        return tf.cast(0.0, settings.float_type)

    def __str__(self):
        return "LinearLayer: input_dim {}, output_dim {}".format(
            self.input_dim, self.output_dim
        )
