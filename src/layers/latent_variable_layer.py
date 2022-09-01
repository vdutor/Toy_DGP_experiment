from enum import Enum

import numpy as np
import tensorflow as tf
from gpflow import Param, params_as_tensors, settings, transforms
from gpflow.kullback_leiblers import gauss_kl
from tensorflow.contrib.distributions import Normal, kl_divergence

from ..encoders import RecognitionNetwork
from ..utils import random_normal
from .layers import BaseLayer


class LatentVarMode(Enum):
    """
    We need to distinguish between training and test points
    when propagating with latent variables. We have
    a parameterized variational posterior for the N data,
    but at test points we might want to do one of three things:
    """

    # sample from N(0, 1)
    PRIOR = 1

    # we are dealing with the N observed data points,
    # so we use the vatiational posterior
    POSTERIOR = 2

    # for plotting purposes, it is useful to have a mechanism
    # for setting W to fixed values, e.g. on a grid
    GIVEN = 3


class LatentVariableLayer(BaseLayer):
    """
    A latent variable layer, with amortized mean-field VI

    The prior is N(0, 1), and inference is factorised N(a, b), where a, b come from
    an encoder network.

    When propagating there are two possibilities:
    1) We're doing inference, so we use the variational distribution
    2) We're looking at test points, so we use the prior
    """

    def __init__(self, latent_variables_dim, prior_std=1.0, XY_dim=None, encoder=None):
        BaseLayer.__init__(self)
        self.latent_variables_dim = latent_variables_dim

        if (encoder is None) and XY_dim:
            encoder = RecognitionNetwork(latent_variables_dim, XY_dim, [10, 10])

        self.encoder = encoder
        self.prior_std = Param(prior_std, transform=transforms.positive)
        self.is_encoded = False

    def encode_once(self):
        if not self.is_encoded:
            XY = tf.concat([self.root.X, self.root.Y], 1)
            q_mu, log_q_sqrt = self.encoder(XY)
            self.q_mu = tf.tile(q_mu, [self.root.num_samples, 1])
            self.q_sqrt = tf.tile(tf.exp(log_q_sqrt), [self.root.num_samples, 1])
            self.is_encoded = True

    def KL(self):
        self.encode_once()
        p = Normal(tf.cast(0.0, dtype=settings.float_type), self.prior_std)
        q = Normal(self.q_mu, self.q_sqrt)
        kl = tf.reduce_sum(kl_divergence(q, p))
        scale = tf.cast(self.root.num_data, settings.float_type) / tf.cast(
            tf.shape(self.root.X)[0], settings.float_type
        )
        kl *= scale
        # kl *= self.root.scale
        kl /= float(self.root.num_samples)
        return kl
        # return gauss_kl(self.q_mu, self.q_sqrt) could whiten and just use this

    def propagate(self, X, sampling=True, W=None, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return "{} with latent dim {}".format(
            self.__class__.__name__, self.latent_variables_dim
        )


class LatentVariableConcatLayer(LatentVariableLayer):
    """
    A latent variable layer where the latents are concatenated with the input
    """

    @params_as_tensors
    def propagate(
        self,
        X,
        sampling=True,
        latent_var_mode=LatentVarMode.POSTERIOR,
        W=None,
        full_cov=False,
        full_output_cov=False,
        eps=None,
    ):

        if sampling:
            if W is not None:
                assert isinstance(W, tf.Tensor)

            elif latent_var_mode == LatentVarMode.POSTERIOR:
                self.encode_once()
                z = (
                    random_normal(tf.shape(self.q_mu)) if eps is None else eps
                )  #  *self.prior_std if whitening
                W = self.q_mu + z * self.q_sqrt

            elif latent_var_mode == LatentVarMode.PRIOR:
                W = self.prior_std * random_normal(
                    [tf.shape(X)[-2], self.latent_variables_dim], use_prand=True
                )

            return tf.concat([X, W], -1)

        else:
            raise NotImplementedError
            # if latent_var_mode == LatentVarMode.POSTERIOR:
            #     self.encode_once()
            #     XW_mean = tf.concat([X, self.q_mu], -1)
            #     XW_var = tf.concat([tf.zeros_like(X), self.q_sqrt ** 2], -1)
            #     return XW_mean, XW_var
            #
            # elif latent_var_mode == LatentVarMode.PRIOR:
            #     z = tf.zeros([tf.shape(X)[0], self.latent_variables_dim], dtype=settings.float_type)
            #     o = tf.ones([tf.shape(X)[0], self.latent_variables_dim], dtype=settings.float_type)
            #     XW_mean = tf.concat([X, z], -2)
            #     XW_var = tf.concat([tf.zeros_like(X), o])
            #     return XW_mean, XW_var
            #
            # else:
            #     raise NotImplementedError

    def describe(self):
        return "LatentVarConcatLayer: with dim={}, using amortized VI".format(
            self.latent_variables_dim
        )


class QuadratureLatentVariableConcatLayer(LatentVariableConcatLayer):
    """
    A quadrature version of the VI latent variable layer. In this case the KL term is set to zero.
    This layer cannot be called with latent_var_mode=LatentVarMode.POSTERIOR, or it will result in an error
    """

    def KL(self):
        return tf.cast(0.0, dtype=settings.float_type)

    def describe(self):
        return "LatentVarConcatLayer: with dim={}, using quadrature".format(
            self.latent_variables_dim
        )


class LatentVariableAdditiveLayer(LatentVariableLayer):
    """
    A latent variable layer where the latents are added to the input
    """

    @params_as_tensors
    def propagate(
        self,
        X,
        sampling=True,
        latent_var_mode=LatentVarMode.POSTERIOR,
        W=None,
        full_cov=False,
        full_output_cov=False,
        eps=None,
    ):

        if sampling:
            if W is not None:
                assert isinstance(W, tf.Tensor)

            elif latent_var_mode == LatentVarMode.POSTERIOR:
                self.encode_once()
                eps = (
                    random_normal(tf.shape(self.q_mu)) if eps is None else eps
                )  #  *self.prior_std if whitening
                z = self.prior_std * eps
                W = self.q_mu + z * self.q_sqrt

            elif latent_var_mode == LatentVarMode.PRIOR:
                W = self.prior_std * random_normal(
                    [tf.shape(X)[-2], self.latent_variables_dim], use_prand=True
                )

            return X + W

        else:

            if latent_var_mode == LatentVarMode.POSTERIOR:
                self.encode_once()
                return X + self.q_mu, self.q_sqrt ** 2

            elif latent_var_mode == LatentVarMode.PRIOR:
                return X, tf.ones_like(X) * self.prior_std

            else:
                raise NotImplementedError

    def describe(self):
        return "LatentVarConcatLayer: with dim={}, using amortized VI".format(
            self.latent_variables_dim
        )


class QuadratureLatentVariableAdditiveLayer(LatentVariableAdditiveLayer):
    """
    A quadrature version of the VI latent variable layer. In this case the KL term is set to zero.
    This layer cannot be called with latent_var_mode=LatentVarMode.POSTERIOR, or it will result in an error
    """

    def KL(self):
        return tf.cast(0.0, dtype=settings.float_type)

    def describe(self):
        return "LatentVarAdditiveLayer: with dim={}, using quadrature".format(
            self.latent_variables_dim
        )
