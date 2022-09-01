from gpflow import params_as_tensors

from .linear_layer import LinearLayer


class PerceptronLayer(LinearLayer):
    """
    Performs a linear transformation and can potentially
    pass the output through a non-linear activation function.
    """

    def __init__(self, input_dim: int, output_dim: int, activation=None):
        super().__init__(input_dim, output_dim)
        self.activation = activation

    @params_as_tensors
    def propagate(self, X, sampling=True, **kwargs):
        linear_transformation = super().propagate(X, sampling)

        if self.activation is not None:
            return self.activation(linear_transformation)
        else:
            return linear_transformation
