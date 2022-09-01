from collections import Iterable
from enum import Enum
from functools import reduce
from typing import List, Optional

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import settings
from gpflow.decors import autoflow, params_as_tensors
from gpflow.likelihoods import Gaussian
from gpflow.models.model import Model
from gpflow.params.dataholders import DataHolder, Minibatch
from gpflow.quadrature import mvhermgauss
from scipy.stats import norm
from tensorflow.contrib.distributions import Normal

from ..layers.latent_variable_layer import LatentVariableLayer, LatentVarMode
from ..utils import random_normal


class DeepGP(Model):
    """
    Implementation of a Deep Gaussian process, following the specification of:

    @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
    }
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        layers: List,
        *,
        likelihood: Optional[gpflow.likelihoods.Likelihood] = None,
        batch_size: Optional[int] = None,
        name: Optional[str] = None,
        num_samples=1
    ):
        """
        :param X: np.ndarray, N x Dx
        :param Y: np.ndarray, N x Dy
        :param layers: list
            List of `layers.BaseLayer` instances, e.g. PerceptronLayer, ConvLayer, GPLayer, ...
        :param likelihood: gpflow.likelihoods.Likelihood object
            Analytic expressions exists for the Gaussian case.
        :param batch_size: int
        """
        Model.__init__(self, name=name)

        assert X.ndim == 2
        assert Y.ndim == 2

        self.num_samples = num_samples

        self.num_data = X.shape[0]
        self.layers = gpflow.ParamList(layers)
        self.likelihood = likelihood or Gaussian()

        if (batch_size is not None) and (batch_size > 0) and (batch_size < X.shape[0]):
            self.X = Minibatch(X, batch_size=batch_size, seed=0)
            self.Y = Minibatch(Y, batch_size=batch_size, seed=0)
            # self.scale = self.num_data / batch_size
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)
            # self.scale = 1.0

    def _get_Ws_iter(self, latent_var_mode: LatentVarMode, Ws=None) -> iter:
        i = 0
        for layer in self.layers:
            if latent_var_mode == LatentVarMode.GIVEN and isinstance(
                layer, LatentVariableLayer
            ):

                # passing some fixed Ws, which are packed to a single tensor for ease of use with autoflow
                assert isinstance(Ws, tf.Tensor)
                d = layer.latent_variables_dim
                yield Ws[:, i : (i + d)]
                i += d
            else:
                yield None

    @params_as_tensors
    def _build_decoder(
        self,
        Z,
        full_cov=False,
        full_output_cov=False,
        Ws=None,
        latent_var_mode=LatentVarMode.POSTERIOR,
    ):
        """
        :param Z: N x W
        """
        Z = tf.cast(Z, dtype=settings.float_type)
        Ws_iter = self._get_Ws_iter(
            latent_var_mode, Ws
        )  # iter, returning either None or slices from Ws

        for layer, W in zip(self.layers[:-1], Ws_iter):
            Z = layer.propagate(
                Z,
                sampling=True,
                W=W,
                latent_var_mode=latent_var_mode,
                full_output_cov=full_output_cov,
                full_cov=full_cov,
            )

        return self.layers[-1].propagate(
            Z,
            sampling=False,
            W=next(Ws_iter),
            latent_var_mode=latent_var_mode,
            full_output_cov=full_output_cov,
            full_cov=full_cov,
        )  # f_mean, f_var

    @params_as_tensors
    def _build_likelihood(self):
        X, Y = tf.tile(self.X, [self.num_samples, 1]), tf.tile(
            self.Y, [self.num_samples, 1]
        )
        # self.XX, self.YY = X, Y  # for the latent var layers
        f_mean, f_var = self._build_decoder(X)  # N x P, N x P
        self.E_log_prob = tf.reduce_sum(
            self.likelihood.variational_expectations(f_mean, f_var, Y)
        )

        self.KL_all = [l.KL() for l in self.layers]
        self.KL_U_layers = reduce(tf.add, self.KL_all)

        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(
            tf.shape(self.X)[0], settings.float_type
        )
        ELBO = self.E_log_prob * scale / float(self.num_samples) - self.KL_U_layers
        return tf.cast(ELBO, settings.float_type)

    def _predict_f(self, X):
        mean, variance = self._build_decoder(
            X, latent_var_mode=LatentVarMode.PRIOR
        )  # N x P, N x P
        return mean, variance

    @params_as_tensors
    @autoflow([settings.float_type, [None, None]])
    def predict_y(self, X):
        mean, var = self._predict_f(X)
        return self.likelihood.predict_mean_and_var(mean, var)

    @autoflow([settings.float_type, [None, None]])
    def predict_f(self, X):
        return self._predict_f(X)

    @autoflow([settings.float_type, [None, None]])
    def predict_f_full_cov(self, X):
        return self._build_decoder(
            X, latent_var_mode=LatentVarMode.PRIOR, full_cov=True
        )

    @autoflow([settings.float_type, [None, None]], [settings.float_type, [None, None]])
    def predict_f_with_Ws(self, X, Ws):
        return self._build_decoder(X, Ws=Ws, latent_var_mode=LatentVarMode.GIVEN)

    @autoflow([settings.float_type, [None, None]], [settings.float_type, [None, None]])
    def predict_f_with_Ws_full_output_cov(self, X, Ws):
        return self._build_decoder(
            X, Ws=Ws, full_output_cov=True, latent_var_mode=LatentVarMode.GIVEN
        )

    @autoflow([settings.float_type, [None, None]], [settings.float_type, [None, None]])
    def predict_f_with_Ws_full_cov(self, X, Ws):
        return self._build_decoder(
            X, Ws=Ws, full_cov=True, latent_var_mode=LatentVarMode.GIVEN
        )

    @autoflow()
    def compute_KL_U(self):
        return self.KL_U_layers

    @autoflow()
    def compute_data_fit(self):
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(
            tf.shape(self.X)[0], settings.float_type
        )
        return self.E_log_prob * scale

    def log_pdf(self, X, Y):
        m, v = self.predict_y(X)
        l = norm.logpdf(Y, loc=m, scale=v ** 0.5)  # TODO only Gaussian..!
        return np.average(l)

    def __str__(self):
        """ High-level description of the model """
        desc = self.__class__.__name__
        desc += "\nLayers"
        desc += "\n------\n"
        desc += "\n".join(str(l) for l in self.layers)
        desc += "\nlikelihood: " + self.likelihood.__class__.__name__
        return desc


class QuadratureMode(Enum):
    """
    Options for performing quadrature
    """

    GAUSS_HERMITE = 1  # NB scales exponentially in the quad_dim
    PRIOR_SAMPLES = 2  # NB stochastic
    IWAE = 3  # NB stochastic
    # TODO: ode solver
    # TODO: scipy.integrate.quad


class DeepGPQuad(DeepGP):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        layers: List,
        *,
        H: Optional[int] = 200,
        quad_layers: Optional[List] = [],
        likelihood: Optional[gpflow.likelihoods.Likelihood] = None,
        batch_size: Optional[int] = None,
        name: Optional[str] = None,
        quadrature_mode: Optional[QuadratureMode] = QuadratureMode.GAUSS_HERMITE,
        inner_sample_full_cov=True
    ):
        DeepGP.__init__(
            self, X, Y, layers, likelihood=likelihood, batch_size=batch_size, name=name
        )
        self.inner_sample_full_cov = inner_sample_full_cov
        self.quad_layers = quad_layers
        assert (
            len(self.quad_layers) > 0
        ), "must do quadrature over at least one layer for this code to work"

        self.Ws_quad_dim = sum([layer.latent_variables_dim for layer in quad_layers])
        self.H = H
        self.quadrature_mode = quadrature_mode

    def _get_Ws_iter(
        self, latent_var_mode: LatentVarMode, Ws_quad=None, Ws=None
    ) -> iter:
        i = 0
        j = 0
        for layer in self.layers:
            if (
                (layer in self.quad_layers)
                and (latent_var_mode == LatentVarMode.POSTERIOR)
                and (Ws_quad is not None)
            ):
                d = layer.latent_variables_dim
                yield Ws_quad[..., j : (j + d)]
                j += d

            elif latent_var_mode == LatentVarMode.GIVEN and isinstance(
                layer, LatentVariableLayer
            ):
                d = layer.latent_variables_dim
                yield Ws[..., i : (i + d)]
                i += d

            else:
                yield None

    # def _get_eps_iter(self, eps):
    #     i = 0
    #     for layer in self.layers:
    #         if hasattr(layer, 'q_mu') and (eps is not None): # A GP layer
    #             d = tf.shape(layer.q_mu)[1]
    #             yield eps[:, i:(i+d)]
    #             i += d
    #         else:
    #             yield None

    @params_as_tensors
    def _build_decoder(
        self,
        Z,
        full_cov=False,
        full_output_cov=False,
        Ws=None,
        Ws_quad=None,
        eps=None,
        latent_var_mode=LatentVarMode.POSTERIOR,
        inner_sample_full_cov=True,
    ):
        """
        :param Z: N x W
        """
        Z = tf.cast(Z, dtype=settings.float_type)

        Ws_iter = self._get_Ws_iter(
            latent_var_mode, Ws=Ws, Ws_quad=Ws_quad
        )  # iter, returning either None or slices from Ws
        # eps_iter = self._get_eps_iter(eps)

        inner_sample_full_cov = full_cov or (
            inner_sample_full_cov and Z.get_shape().ndims == 3
        )

        for layer, W in zip(self.layers[:-1], Ws_iter):
            Z = layer.propagate(
                Z,
                sampling=True,
                W=W,
                latent_var_mode=latent_var_mode,
                full_output_cov=full_output_cov,
                full_cov=inner_sample_full_cov,
            )

        return self.layers[-1].propagate(
            Z,
            sampling=False,
            W=next(Ws_iter),
            latent_var_mode=latent_var_mode,
            full_output_cov=full_output_cov,
            full_cov=full_cov,
        )

    @params_as_tensors
    def _build_likelihood(self, inner_sample_full_cov=None):
        inner_sample_full_cov = inner_sample_full_cov or self.inner_sample_full_cov
        N = tf.shape(self.X)[0]

        quad_dim = np.sum([layer.latent_variables_dim for layer in self.quad_layers])
        l = lambda layer: layer.prior_std * tf.ones(
            layer.latent_variables_dim, dtype=settings.float_type
        )
        prior_stds = tf.concat(
            [l(layer) for layer in self.quad_layers], 0
        )  # [quad_dim, ]
        eps = None

        if self.quadrature_mode == QuadratureMode.GAUSS_HERMITE:
            xn, wn = mvhermgauss(
                self.H, quad_dim
            )  # NB this is H**quad_dim, so infeasible if quad_dim is large
            xn *= np.sqrt(2.0)
            wn *= np.pi ** (-0.5 * quad_dim)
            log_wn = np.log(wn)

            gh_x = tf.reshape(xn, (1, self.H ** quad_dim, quad_dim))
            log_gh_w = tf.reshape(log_wn, (1, self.H ** quad_dim, 1))

            W = tf.cast(
                gh_x * tf.reshape(prior_stds, [1, 1, quad_dim]),
                dtype=settings.float_type,
            )
            W = tf.tile(W, [N, 1, 1])  # [N, H**quad_dim, quad_dim]

        elif self.quadrature_mode == QuadratureMode.PRIOR_SAMPLES:
            xn = random_normal((N, self.H, quad_dim))
            W = tf.cast(
                xn * tf.reshape(prior_stds, [1, 1, quad_dim]), dtype=settings.float_type
            )
            log_gh_w = -np.ones((1, self.H, 1)) * np.log(self.H)

        elif self.quadrature_mode == QuadratureMode.IWAE:
            log_wn = []
            xn = []
            for layer in self.quad_layers:
                layer.encode_once()

                z = random_normal([N, self.H, layer.latent_variables_dim])

                x = layer.q_mu[:, None, :] + z * (
                    layer.q_sqrt[:, None, :]
                )  # [N, H**quad_dim, Dw]
                logp = Normal(
                    tf.cast(0.0, settings.float_type), layer.prior_std
                ).log_prob(
                    x
                )  # [N, H**quad_dim, Dw]
                logq = Normal(
                    layer.q_mu[:, None, :], layer.q_sqrt[:, None, :]
                ).log_prob(
                    x
                )  # [N, H**quad_dim, Dw]
                xn.append(x)

                log_wn.append(logp - logq)  # [N, H**quad_dim, Dw]

            W = tf.concat(xn, 2)  # N, H**quad_dim, Dw

            log_gh_w = tf.reduce_sum(tf.concat(log_wn, 2), 2) - np.log(self.H)  # N, H

            gp_dims = 0
            for layer in self.layers:
                if hasattr(layer, "q_mu"):
                    gp_dims += tf.shape(layer.q_mu)[1]
        else:
            raise NotImplementedError

        X = tf.tile(
            self.X[:, None, :], [1, tf.shape(log_gh_w)[1], 1]
        )  # [N, H**quad_dim, Dx]
        Y = tf.tile(
            self.Y[:, None, :], [1, tf.shape(log_gh_w)[1], 1]
        )  # [N, H**quad_dim, Dy]

        # self.XX, self.YY = X, Y  # for the latent var layers

        f_mean, f_var = self._build_decoder(
            X, Ws_quad=W, inner_sample_full_cov=inner_sample_full_cov
        )  # [N, H, Dy]

        log_f = tf.reduce_sum(
            self.likelihood.variational_expectations(f_mean, f_var, Y), -1
        )  # [N, H]

        self.E_log_prob = tf.reduce_sum(tf.reduce_logsumexp(log_f + log_gh_w, axis=1))

        self.KL_all = [l.KL() for l in self.layers]
        self.KL_U_layers = reduce(tf.add, self.KL_all)

        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(
            tf.shape(self.X)[0], settings.float_type
        )
        ELBO = self.E_log_prob * scale - self.KL_U_layers
        return tf.cast(ELBO, settings.float_type)

    @gpflow.autoflow()
    @gpflow.params_as_tensors
    def compute_log_likelihood(self):
        # always use the full cov sampling for the maginal likelihood, but not necessarily for training
        return self._build_likelihood(inner_sample_full_cov=True)


class InterpolatingDeepGPQuad(DeepGP):
    pass


#     def __init__(self,
#                  X: np.ndarray,
#                  Y: np.ndarray,
#                  layers: List, *,
#                  H: Optional[int] = 200,
#                  quad_layers: Optional[List] = [],
#                  likelihood: Optional[gpflow.likelihoods.Likelihood] = None,
#                  batch_size: Optional[int] = None,
#                  name: Optional[str] = None,
#                  quadrature_mode: Optional[QuadratureMode] = QuadratureMode.GAUSS_HERMITE,
#                  alpha=1.):  # alpha=1 is correlated samples, alpha=0 is iid samples
#         DeepGP.__init__(self, X, Y, layers,
#                         likelihood=likelihood,
#                         batch_size=batch_size,
#                         name=name)
#         self.quad_layers = quad_layers
#         assert len(self.quad_layers) > 0, 'must do quadrature over at least one layer for this code to work'
#
#         d = 0
#         for layer in quad_layers:
#             d +=  layer.latent_variables_dim
#         self.Ws_quad_dim = d
#         self.H = H
#         self.quadrature_mode = quadrature_mode
#         self.alpha = gpflow.Param(alpha)
#         self.alpha.set_trainable(False)
#
#     def _get_Ws_iter(self, latent_var_mode : LatentVarMode, Ws_quad=None, Ws=None) -> iter:
#         i = 0
#         j = 0
#         for layer in self.layers:
#             if (layer in self.quad_layers) and (latent_var_mode == LatentVarMode.POSTERIOR) and (Ws_quad is not None):
#                 d = layer.latent_variables_dim
#                 yield Ws_quad[..., j:(j + d)]
#                 j += d
#
#             elif (latent_var_mode == LatentVarMode.GIVEN and isinstance(layer, LatentVariableLayer)):
#                 d = layer.latent_variables_dim
#                 yield Ws[..., i:(i+d)]
#                 i += d
#
#             else:
#                 yield None
#
#     @params_as_tensors
#     def _build_decoder(self, Z, full_cov=False, full_output_cov=False,
#                        Ws=None, Ws_quad=None, eps=None, latent_var_mode=LatentVarMode.POSTERIOR,
#                        inner_sample_full_cov=True):
#         """
#         :param Z: N x W
#         """
#         Z = tf.cast(Z, dtype=settings.float_type)
#
#         Ws_iter = self._get_Ws_iter(latent_var_mode, Ws=Ws, Ws_quad=Ws_quad)  # iter, returning either None or slices from Ws
#
#         inner_sample_full_cov = full_cov or (inner_sample_full_cov and Z.get_shape().ndims==3)
#
#         for layer, W in zip(self.layers[:-1], Ws_iter):
#             Z = layer.propagate(Z,
#                                 sampling=True,
#                                 W=W,
#                                 latent_var_mode=latent_var_mode,
#                                 full_output_cov=full_output_cov,
#                                 full_cov=inner_sample_full_cov)
#
#         return self.layers[-1].propagate(Z,
#                                          sampling=False,
#                                          W=next(Ws_iter),
#                                          latent_var_mode=latent_var_mode,
#                                          full_output_cov=full_output_cov,
#                                          full_cov=full_cov)
#     @params_as_tensors
#     def _build_likelihood(self):
#         alpha = self.alpha
#         return self._bl(True) * alpha + self._bl(False) * (1. - alpha)
#
#     @gpflow.autoflow()
#     @gpflow.params_as_tensors
#     def compute_log_likelihood_alpha1(self):
#         return self._bl(True)
#
#     @gpflow.autoflow()
#     @gpflow.params_as_tensors
#     def compute_log_likelihood_alpha0(self):
#         return self._bl(False)
#
#     @params_as_tensors
#     def _bl(self, inner_sample_full_cov):
#
#         N = tf.shape(self.X)[0]
#
#         quad_dim = np.sum([layer.latent_variables_dim for layer in self.quad_layers])
#         l = lambda layer: layer.prior_std * tf.ones(layer.latent_variables_dim, dtype=settings.float_type)
#         prior_stds = tf.concat([l(layer) for layer in self.quad_layers], 0)  # [quad_dim, ]
#         eps = None
#
#         if self.quadrature_mode == QuadratureMode.GAUSS_HERMITE:
#             xn, wn = mvhermgauss(self.H, quad_dim)  # NB this is H**quad_dim, so infeasible if quad_dim is large
#             xn *= np.sqrt(2.0)
#             wn *= np.pi ** (-0.5 * quad_dim)
#             log_wn = np.log(wn)
#
#             gh_x = tf.reshape(xn, (1, self.H ** quad_dim, quad_dim))
#             log_gh_w = tf.reshape(log_wn, (1, self.H ** quad_dim, 1))
#
#             W = tf.cast(gh_x * tf.reshape(prior_stds, [1, 1, quad_dim]), dtype=settings.float_type)
#             W = tf.tile(W, [N, 1, 1])  # [N, H**quad_dim, quad_dim]
#
#
#         elif self.quadrature_mode == QuadratureMode.PRIOR_SAMPLES:
#             xn = random_normal((N, self.H, quad_dim))
#             W = tf.cast(xn * tf.reshape(prior_stds, [1, 1, quad_dim]), dtype=settings.float_type)
#             log_gh_w = -np.ones((1, self.H, 1)) * np.log(self.H)
#
#         elif self.quadrature_mode == QuadratureMode.IWAE:
#             log_wn = []
#             xn = []
#             for layer in self.quad_layers:
#                 layer.encode_once()
#
#                 z = random_normal([N, self.H, layer.latent_variables_dim])
#
#                 x = layer.q_mu[:, None, :] + z * (layer.q_sqrt[:, None, :])  # [N, H**quad_dim, Dw]
#                 logp = Normal(tf.cast(0., settings.float_type), layer.prior_std).log_prob(x)  # [N, H**quad_dim, Dw]
#                 logq = Normal(layer.q_mu[:, None, :], layer.q_sqrt[:, None, :]).log_prob(x) # [N, H**quad_dim, Dw]
#                 xn.append(x)
#
#                 log_wn.append(logp - logq)  # [N, H**quad_dim, Dw]
#
#             W = tf.concat(xn, 2)  # N, H**quad_dim, Dw
#
#             log_gh_w = tf.reduce_sum(tf.concat(log_wn, 2), 2, keepdims=True) - np.log(self.H)  # N, H, quad_dim
#
#             gp_dims = 0
#             for layer in self.layers:
#                 if hasattr(layer, 'q_mu'):
#                     gp_dims += tf.shape(layer.q_mu)[1]
#         else:
#             raise NotImplementedError
#
#         X = tf.tile(self.X[:, None, :], [1, tf.shape(log_gh_w)[1], 1])  # [N, H**quad_dim, Dx]
#         Y = tf.tile(self.Y[:, None, :], [1, tf.shape(log_gh_w)[1], 1])  # [N, H**quad_dim, Dy]
#
#     # self.XX, self.YY = X, Y  # for the latent var layers
#
#         f_mean, f_var = self._build_decoder(X, Ws_quad=W, inner_sample_full_cov=inner_sample_full_cov)  # [N, H, Dy]
#
#         log_f = self.likelihood.variational_expectations(f_mean, f_var, Y)
#
#         E_log_prob = tf.reduce_sum(tf.reduce_logsumexp(log_f + log_gh_w, axis=1))
#
#         KL_all = [l.KL() for l in self.layers]
#         KL_U_layers = reduce(tf.add, KL_all)
#
#         scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(self.X)[0], settings.float_type)
#         ELBO = E_log_prob * scale - KL_U_layers
#
#         return tf.cast(ELBO, settings.float_type)
#
#         # self.E_log_prob = tf.reduce_sum(tf.reduce_logsumexp(log_f + log_gh_w, axis=1))
#         # self.KL_all = [l.KL() for l in self.layers]
#         # self.KL_U_layers = reduce(tf.add, self.KL_all)
#         # ELBO = self.E_log_prob * self.scale - self.KL_U_layers
