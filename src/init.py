import gpflow
import numpy as np
from gpflow.kernels import RBF
from sklearn.feature_extraction.image import extract_patches_2d


class Initializer(object):
    """
    Base class for parameter initializers.
    It should be subclassed when implementing new types.
    Can be used for weights of a neural network, inducing patches, points, etc.
    """

    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError()  # pragma: no cover


class PatchSamplerInitializer(Initializer):
    def __init__(self, X, width=None, height=None):
        """
        :param X: np.array
            N x W x H
        """
        if width is None and height is None:
            if X.ndim <= 2:
                raise ValueError("Impossible to infer image width and height")
            else:
                width, height = X.shape[1], X.shape[2]

        self.X = np.reshape(X, [-1, width, height])

    def sample(self, shape):
        """
        :param shape: tuple
            M x w x h, number of patches x patch width x patch height
        :return: np.array
            returns M patches of size w x h, specified by the `shape` param.
        """
        num = shape[0]  # M
        patch_size = shape[1:]  # w x h

        patches = np.array([extract_patches_2d(im, patch_size) for im in self.X])
        patches = np.reshape(patches, [-1, *patch_size])  # (N * P) x w x h
        idx = np.random.permutation(range(len(patches)))[:num]  # M
        return patches[idx, ...]  # M x w x h


class NormalInitializer(Initializer):
    """
    Sample initial weights from the Normal distribution.
    :param std: float
        Std of initial parameters.
    :param mean: float
        Mean of initial parameters.
    """

    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def sample(self, shape):
        return np.random.normal(loc=self.mean, scale=self.std, size=shape)


class KernelStructureMixingMatrixInitializer(Initializer):
    """
    Initialization routine for the Mixing Matrix P,
    used in f(x) = P g(x).
    """

    def __init__(self, kern=None):
        self.kern = RBF(2, variance=2.0) if kern is None else kern

    def sample(self, shape):
        """
        :param shape: tuple, P x L.
        Note that P is both used for the dimension and the matrix.
        """
        im_width, num_latent = shape  # P x L
        IJ = np.vstack(
            [x.flatten() for x in np.meshgrid(np.arange(im_width), np.arange(im_width))]
        ).T
        K_IJ = self.kern.compute_K_symm(IJ) + np.eye(im_width ** 2) * 1e-6
        u, s, v = np.linalg.svd(K_IJ)
        P = u[:, :num_latent] * s[None, :num_latent] ** 0.5  # P x L
        return P
