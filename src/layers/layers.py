from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import (
    Param,
    Parameterized,
    features,
    params_as_tensors,
    params_as_tensors_for,
    settings,
)
from gpflow.conditionals import _sample_mvn, conditional, sample_conditional
from gpflow.features import InducingFeature, InducingPoints, Kuf, Kuu
from gpflow.kernels import Kernel
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Linear, Zero
from gpflow.multioutput import MixedKernelSharedMof, Mok
from gpflow.params import Parameter, Parameterized


def independent_multisample_sample_conditional(
    Xnew: tf.Tensor,
    feat: InducingPoints,
    kern: Kernel,
    f: tf.Tensor,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False
):
    """
    Multisample, single-output GP conditional.
    NB if full_cov=False is required, this functionality can be achieved by reshaping Xnew to SN x D
    nd using conditional. The purpose of this function is to compute full covariances in batch over S samples.
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: M x M
    - Kuf: S x M x N
    - Kff: S x N or S x N x N
    ----------
    :param Xnew: data matrix, size S x N x D.
    :param f: data matrix, M x R
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs. Must be False
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x R or R x M x M.
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     S x N x R
        - variance: S x N x R, S x R x N x N
    """
    if full_output_cov:
        raise NotImplementedError

    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # M x M

    S, N, D = tf.shape(Xnew)[0], tf.shape(Xnew)[1], tf.shape(Xnew)[2]
    M = tf.shape(Kmm)[0]

    Kmn_M_SN = Kuf(feat, kern, tf.reshape(Xnew, [S * N, D]))  # M x SN
    Knn = kern.K(Xnew) if full_cov else kern.Kdiag(Xnew)  # S x N or S x N x N

    num_func = tf.shape(f)[1]  # (=R)
    Lm = tf.cholesky(Kmm)  # M x M

    # Compute the projection matrix A
    A_M_SN = tf.matrix_triangular_solve(Lm, Kmn_M_SN, lower=True)
    A = tf.transpose(tf.reshape(A_M_SN, [M, S, N]), [1, 0, 2])  # S x M x N

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)  # S x N x N
        fvar = tf.tile(fvar[:, None, :, :], [1, num_func, 1, 1])  # S x R x N x N
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # S x N
        fvar = tf.tile(fvar[:, None, :], [1, num_func, 1])  # S x R x N

    # another backsubstitution in the unwhitened case
    if not white:
        A_M_SN = tf.matrix_triangular_solve(tf.transpose(Lm), A_M_SN, lower=False)
        A = tf.transpose(tf.reshape(A_M_SN, [M, S, N]), [1, 0, 2])  # S x M x N

    # construct the conditional mean
    fmean = tf.matmul(
        A, tf.tile(f[None, :, :], [S, 1, 1]), transpose_a=True
    )  # S x N x R
    # fmean = tf.einsum('snm,nr->smr', A, f)  # S x N x R

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = (
                A[:, None, :, :] * tf.transpose(q_sqrt)[None, :, :, None]
            )  # S x R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            # L = tf.tile(tf.matrix_band_part(q_sqrt, -1, 0)[None, :, :, :], [S, 1, 1, 1])  # S x R x M x M
            # A_tiled = tf.tile(tf.expand_dims(A, 1), tf.stack([1, num_func, 1, 1]))  # S x R x M x N
            # LTA = tf.matmul(L, A_tiled, transpose_a=True)  # S x R x M x N
            LTA = tf.einsum("rMm,sMn->srmn", tf.matrix_band_part(q_sqrt, -1, 0), A)
        else:  # pragma: no cover
            raise ValueError(
                "Bad dimension for q_sqrt: %s" % str(q_sqrt.get_shape().ndims)
            )
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # S x R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 2)  # S x R x N

    if not full_cov:
        z = tf.random_normal(tf.shape(fmean), dtype=settings.float_type)
        fvar = tf.matrix_transpose(fvar)  # S x N x R
        sample = fmean + z * fvar ** 0.5
    else:
        fmean_SRN1 = tf.transpose(fmean, [0, 2, 1])[:, :, :, None]
        z = tf.random_normal(tf.shape(fmean_SRN1), dtype=settings.float_type)
        sample_SRN1 = fmean + tf.matmul(tf.cholesky(fvar), z)
        sample = tf.transpose(sample_SRN1[:, :, :, 0], [0, 2, 1])

    return sample, fmean, fvar  # fmean is S x N x R, fvar is S x R x N x N or S x N x R


def multisample_sample_conditional(
    Xnew: tf.Tensor,
    feat: InducingPoints,
    kern: Kernel,
    f: tf.Tensor,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False
):
    if isinstance(kern, SharedMixedMok) and isinstance(feat, MixedKernelSharedMof):
        if Xnew.get_shape().ndims == 3:
            sample, gmean, gvar = independent_multisample_sample_conditional(
                Xnew,
                feat.feat,
                kern.kernel,
                f,
                white=white,
                q_sqrt=q_sqrt,
                full_output_cov=False,
                full_cov=False,
            )  # N x L, N x L

            o = tf.ones(([tf.shape(Xnew)[0], 1, 1]), dtype=settings.float_type)

        else:
            sample, gmean, gvar = sample_conditional(
                Xnew,
                feat.feat,
                kern.kernel,
                f,
                white=white,
                q_sqrt=q_sqrt,
                full_output_cov=False,
                full_cov=False,
            )  # N x L, N x L

            o = 1.0

        with params_as_tensors_for(kern):
            f_sample = tf.matmul(sample, o * kern.W, transpose_b=True)
            f_mu = tf.matmul(gmean, o * kern.W, transpose_b=True)
            f_var = tf.matmul(gvar, o * kern.W ** 2, transpose_b=True)

        return f_sample, f_mu, f_var
    else:
        assert not isinstance(kern, Mok)
        if Xnew.get_shape().ndims == 3:
            return independent_multisample_sample_conditional(
                Xnew,
                feat,
                kern,
                f,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                q_sqrt=q_sqrt,
                white=white,
            )
        else:
            return sample_conditional(
                Xnew,
                feat,
                kern,
                f,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                q_sqrt=q_sqrt,
                white=white,
            )


class BaseLayer(Parameterized):
    def propagate(self, X, sampling=True, **kwargs):
        """
        :param X: tf.Tensor
            N x D
        :param sampling: bool
           If `True` returns a sample from the predictive distribution
           If `False` returns the mean and variance of the predictive distribution
        :return: If `sampling` is True, then the function returns a tf.Tensor
            of shape N x P, else N x P for the mean and N x P for the variance
            where P is of size W x H x C (in the case of images)
        """
        raise NotImplementedError()  # pragma: no cover

    def KL(self):
        """ returns KL[q(U) || p(U)] """
        raise NotImplementedError()  # pragma: no cover

    def __str__(self):
        """ describes the key properties of a layer """
        raise NotImplementedError()  # pragma: no cover


class GPLayer(BaseLayer):
    def __init__(
        self,
        kern: gpflow.kernels.Kernel,
        feature: gpflow.features.InducingFeature,
        num_latents: int,
        q_mu: Optional[np.ndarray] = None,
        q_sqrt: Optional[np.ndarray] = None,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        input_prop=False,
    ):
        r"""
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = L v + mean_function(X), where v ~ N(0, I) and L Lᵀ = kern.K(X)

        The variational distribution over the whitened inducing function values is
        q(v) = N(q_mu, q_sqrt q_sqrtᵀ)

        The layer holds num_latents independent GPs, potentially with different kernels or
        different inducing inputs.

        :param kern: The kernel for the layer (input_dim = D_in)
        :param feature: inducing features
        :param num_latents: number of latent GPs in the layer
        :param q_mu: Variational mean initialization (M x num_latents)
        :param q_sqrt: Variational Cholesky factor of variance initialization (num_latents x M x M)
        :param mean_function: The mean function that links inputs to outputs
                (e.g., linear mean function)
        """
        super().__init__()
        assert isinstance(feature, InducingFeature)  # otherwise breaks conditional
        self.feature = feature
        self.kern = kern
        self.mean_function = Zero() if mean_function is None else mean_function
        self.input_prop = input_prop

        M = len(self.feature)
        self.num_latents = num_latents
        q_mu = np.zeros((M, num_latents)) if q_mu is None else q_mu
        q_sqrt = np.tile(np.eye(M), (num_latents, 1, 1)) if q_sqrt is None else q_sqrt
        self.q_mu = Param(q_mu, dtype=settings.float_type)
        self.q_sqrt = Param(q_sqrt, dtype=settings.float_type)

    @params_as_tensors
    def propagate(
        self, X, *, sampling=True, full_output_cov=False, full_cov=False, **kwargs
    ):
        """
        :param X: N x P
        """
        if X.get_shape().ndims == 3:
            mean, var = multisample_conditional(
                X,
                self.feature,
                self.kern,
                self.q_mu,
                q_sqrt=self.q_sqrt,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                white=True,
            )
        else:
            mean, var = conditional(
                X,
                self.feature,
                self.kern,
                self.q_mu,
                q_sqrt=self.q_sqrt,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                white=True,
            )

        if sampling:
            if full_cov:
                # mean = [S, N, D], cov = [S, D, N, N], and we want a full cov sample over N
                # The _sample_mvn function assumes mean = [..., D] and var = [..., D, D], so we need to transpose
                sample = tf.matrix_transpose(
                    _sample_mvn(tf.matrix_transpose(mean), var, "full")
                )
            else:
                # mean = [S, N, D], cov = [S, N, D]. No need to transpose here as _sample_mvn is elementwise
                sample = _sample_mvn(mean, var, "diag")

            sample = sample + self.mean_function(X)  # S x N x P

            if self.input_prop:
                sample = tf.concat([X, sample], -1)

            return sample
        else:

            if self.input_prop:
                raise NotImplementedError

            return mean + self.mean_function(X), var  # N x P, variance depends on args

        # else:
        #     if full_cov:
        #         # note to self...
        #         raise NotImplementedError('Not actually invalid, but are you sure you want this...?')
        #     return mean + self.mean_function(X), var

        # else:
        #     return
        #
        # if sampling:
        #         mean, var = conditional(X, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt,
        #                                 full_cov=full_cov,
        #                                 full_output_cov=full_output_cov,
        #                                 white=True)
        #
        #         # assert full_cov is False
        #         # sample = sample_conditional(X, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt,
        #         #                             full_cov=full_cov,
        #         #                             # full_output_cov=full_output_cov,
        #         #                             white=True)
        #
        #         sample = sample + self.mean_function(X)  # S x N x P
        #
        #         if self.input_prop:
        #             sample = tf.concat([X, sample], -1)
        #         return sample
        #
        #     else:
        #         if self.input_prop:
        #             raise NotImplementedError
        #
        #         mean, var = conditional(X, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt,
        #                                 full_cov=full_cov,
        #                                 # full_output_cov=full_output_cov,
        #                                 white=True)
        #
        #     return mean + self.mean_function(X), var  # N x P, variance depends on args

    @params_as_tensors
    def KL(self):
        """
        The KL divergence from the variational distribution to the prior
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        return gauss_kl(self.q_mu, self.q_sqrt)

    def __str__(self):
        """ returns a string with the key properties of a GPlayer """
        return "GPLayer: kern {}, features {}, mean {}, L {}".format(
            self.kern.__class__.__name__,
            self.feature.__class__.__name__,
            self.mean_function.__class__.__name__,
            self.num_latents,
        )
