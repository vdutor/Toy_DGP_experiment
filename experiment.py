# Copyright (2018). Hugh Salimbeni.

import os

import gpflow.training.monitor as mon
import numpy as np
import tensorflow as tf
from gpflow import defer_build
from gpflow.features import InducingPoints
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Identity
from gpflow.training import AdamOptimizer, NatGradOptimizer
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from scipy.stats import norm

from src.layers import GPLayer, LatentVariableConcatLayer, LinearLayer
from src.layers import QuadratureLatentVariableConcatLayer
from src.models import DeepGP, DeepGPQuad, QuadratureMode

if not os.path.isdir("results"):
    os.mkdir("results")

figs_base_path = os.path.join("results", "figs")
tensorboard_base_path = os.path.join("results", "tensorboard")
checkpoints_base_path = os.path.join("results", "checkpoints")

if not os.path.isdir(figs_base_path):
    os.mkdir(figs_base_path)

if not os.path.isdir(tensorboard_base_path):
    os.mkdir(tensorboard_base_path)

if not os.path.isdir(checkpoints_base_path):
    os.mkdir(checkpoints_base_path)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--configuration", default="L1_G1_G1", nargs="?", type=str)
parser.add_argument("--mode", default="VI", nargs="?", type=str)
parser.add_argument("--H", default=1, nargs="?", type=int)
input_args = parser.parse_args()


class ARGS:
    likelihood_variance = 1e-1
    gamma = 0.1
    lr = 0.01
    prior_std = 1.0
    inner_var = 2.0
    M = 100
    mb = 1000
    lengthscale = 1.0
    full_natgrads = False

    train_linear_projections = False

    configuration = input_args.configuration
    H = input_args.H
    mode = input_args.mode

    file_name = "italics_H{}_DGP_{}_{}".format(H, configuration, mode)
    tensorboard_path = os.path.join(tensorboard_base_path, file_name)
    checkpoint_path = os.path.join(checkpoints_base_path, file_name)
    figs_path = os.path.join(figs_base_path, file_name)

    print_freq = 1000
    saving_freq = 50
    tensorboard_freq = 50
    plotting_freq = 5000

    num_samples = 500

    iterations = 62000


print(ARGS.file_name)


def make_data():
    from PIL import Image

    image = np.array(Image.open("DGP.png"))
    image[np.where(image != 0.0)] = 255.0
    image = image[..., 0]
    X, y = [], []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 0.0:
                X_n, y_n = j, image.shape[1] - i
                X.append(X_n)
                y.append(y_n)
    X = np.array(X)
    y = np.array(y)
    y = 6 * ((y - y.min()) / (y.max() - y.min()) - 0.5)
    X = 6 * ((X - X.min()) / (X.max() - X.min()) - 0.5)
    X = X[..., None]
    Y = y[..., None]
    return X, Y


def make_model(X, Y):
    with defer_build():
        lik = Gaussian()
        lik.variance = ARGS.likelihood_variance

        layers = []
        quad_layers = []

        D = 1
        DX = 1
        PX = np.eye(1)
        Z = np.linspace(min(X), max(X), ARGS.M).reshape(-1, 1)

        if len(ARGS.configuration) > 0:
            for c, d in ARGS.configuration.split("_"):
                if c == "G":
                    kern = RBF(
                        D,
                        lengthscales=ARGS.lengthscale * float(D) ** 0.5,
                        variance=ARGS.inner_var,
                        ARD=True,
                    )

                    if d == "X":
                        l = GPLayer(
                            kern, InducingPoints(Z), D, mean_function=Identity()
                        )
                        layers.append(l)
                    else:
                        d = min(
                            int(d), DX
                        )
                        l = GPLayer(kern, InducingPoints(Z), d, input_prop=True)
                        layers.append(l)
                        P = np.concatenate([PX[:, :d], np.eye(D)], 1).T
                        l = LinearLayer(d + D, D, weight=P)
                        l.set_trainable(ARGS.train_linear_projections)
                        layers.append(l)

                elif c == "L":
                    d = int(d)
                    D += d
                    Z = np.concatenate([Z, np.random.randn(Z.shape[0], d)], 1)
                    PX = np.concatenate([PX, np.zeros((d, DX))], 0)
                    encoder = (
                        None
                    )
                    if ARGS.mode == "VI":
                        layer = LatentVariableConcatLayer(
                            d, XY_dim=DX + 1, prior_std=ARGS.prior_std, encoder=encoder
                        )
                        layers.append(layer)
                    else:
                        layer = QuadratureLatentVariableConcatLayer(
                            d, XY_dim=DX + 1, prior_std=ARGS.prior_std, encoder=encoder
                        )
                        quad_layers.append(layer)
                        layers.append(layer)

        kern = RBF(
            D, lengthscales=ARGS.lengthscale * float(D) ** 0.5, variance=1.0, ARD=True
        )
        layers.append(GPLayer(kern, InducingPoints(Z), 1))

        ####################################

        if ARGS.mode == "VI":
            model = DeepGP(
                X,
                Y,
                layers,
                likelihood=lik,
                num_samples=ARGS.H,
                batch_size=ARGS.mb,
                name="Model",
            )

        else:
            model = DeepGPQuad(
                X,
                Y,
                layers,
                likelihood=lik,
                batch_size=ARGS.mb,
                quadrature_mode=QuadratureMode.IWAE,
                quad_layers=quad_layers,
                H=ARGS.H,
                name="Model",
            )

    model.model_name = ARGS.file_name
    model.compile()

    return model


def draw_from_prior(model, ax):
    XX = np.random.uniform(low=-3, high=3, size=(3000, 1))
    m, v = model.predict_f_full_cov(XX)
    s = m + np.linalg.cholesky(
        v[0, :, :] + 1e-6 * np.eye(XX.shape[0])
    ) @ np.random.randn(m.shape[0], 1)
    ax.scatter(XX, s, marker=".", alpha=0.2)


def plot_prior(model):
    for i in range(5):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        draw_from_prior(model, ax)
        plt.savefig(
            os.path.join(ARGS.figs_path, "prior_{}_{}.png".format(model.model_name, i)),
            bbox_inches="tight",
        )
        plt.close()


def plot_posterior(model, X, Y):
    print("plot posterior")
    sess = model.enquire_session()
    Ns = 200
    Xs = np.linspace(min(X) - 1.5, max(X) + 1.5, Ns).reshape(-1, 1)
    ls = []

    S = ARGS.num_samples
    My = 200
    levels = np.linspace(min(Y) - 1.5, max(Y) + 1.5, My)

    ms, vs = [], []
    lik_var = model.likelihood.variance.read_value(session=sess)
    for i in range(My * S):
        print(i, end=" ")
        m, vfull = model.predict_f_full_cov(Xs, session=sess)
        v = np.diag(vfull[0, :, :]) + lik_var
        ms.append(m)
        vs.append(v)
    print("done")

    m = np.reshape(np.array(ms), [S, My, Ns])
    v = np.reshape(np.array(vs), [S, My, Ns])
    l = logsumexp(norm.logpdf(levels[None, :], m, v ** 0.5), axis=0) - np.log(S)

    for colormap in ["viridis", "cool", "gray", "gray_r", "bwr", "seismic"]:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        ax1.axis("off")
        ax1.pcolor(Xs.flatten(), levels, np.exp(l), cmap=colormap)

        plt.savefig(
            os.path.join(
                ARGS.figs_path, f"posterior_{model.model_name}_{colormap}.png"
            ),
            bbox_inches="tight",
        )
        ax1.scatter(X, Y, marker=".", color="C1", alpha=0.2, s=1, zorder=1)
        plt.savefig(
            os.path.join(
                ARGS.figs_path, f"posterior_data_{model.model_name}_{colormap}.png"
            ),
            bbox_inches="tight",
        )
    print("plot posterior -- END")
    return fig


def train(model):
    sess = model.enquire_session()

    for layer in model.layers[:-1]:
        if isinstance(layer, GPLayer):
            layer.q_sqrt = layer.q_sqrt.read_value() * 1e-5

    op_ngs = []
    gamma_inners = []

    if ARGS.full_natgrads == True:
        for layer in model.layers[:-1]:
            if isinstance(layer, GPLayer):
                var_list = [[layer.q_mu, layer.q_sqrt]]

                layer.q_mu.set_trainable(False)
                layer.q_sqrt.set_trainable(False)

                gamma = tf.Variable(ARGS.gamma, dtype=tf.float64)
                op_ng = NatGradOptimizer(gamma=gamma).make_optimize_tensor(
                    model, var_list=var_list
                )
                op_ngs.append(op_ng)
                gamma_inners.append(gamma)

        sess.run(tf.variables_initializer(gamma_inners))

    op_increase_gamma_inners = [
        tf.assign(g, tf.where(g * 1.1 < ARGS.gamma, g * 1.1, g)) for g in gamma_inners
    ]
    op_decrease_gamma_inners = [tf.assign(g, g / (1.1 ** 25)) for g in gamma_inners]

    var_list = [[model.layers[-1].q_mu, model.layers[-1].q_sqrt]]

    model.layers[-1].q_mu.set_trainable(False)
    model.layers[-1].q_sqrt.set_trainable(False)

    op_ng = NatGradOptimizer(gamma=ARGS.gamma).make_optimize_tensor(
        model, var_list=var_list
    )
    op_adam = AdamOptimizer(ARGS.lr).make_optimize_tensor(model)

    model.L = []

    ############################ monitor stuff
    global_step = mon.create_global_step(sess)
    op_increment = tf.assign_add(global_step, 1)

    print_task = (
        mon.PrintTimingsTask()
        .with_name("print")
        .with_condition(mon.PeriodicIterationCondition(ARGS.print_freq))
    )

    checkpoint_task = (
        mon.CheckpointTask(checkpoint_dir=ARGS.checkpoint_path)
        .with_name("checkpoint")
        .with_condition(mon.PeriodicIterationCondition(ARGS.saving_freq))
        .with_exit_condition(True)
    )

    writer = mon.LogdirWriter(ARGS.tensorboard_path)
    tensorboard_task = (
        mon.ModelToTensorBoardTask(writer, model)
        .with_name("tensorboard")
        .with_condition(mon.PeriodicIterationCondition(ARGS.tensorboard_freq))
    )

    plot_task = (
        mon.ImageToTensorBoardTask(
            writer, lambda: plot_posterior(model, X, Y), "posterior"
        )
        .with_name("posteriors")
        .with_condition(mon.PeriodicIterationCondition(ARGS.plotting_freq))
        .with_exit_condition(True)
    )

    monitor_tasks = [print_task]

    ###############################

    with mon.Monitor(monitor_tasks, sess, global_step, print_summary=True) as monitor:
        try:
            mon.restore_session(sess, ARGS.checkpoint_path)
        except ValueError:
            pass

        print(sess.run(global_step))

        for it in range(max([ARGS.iterations - sess.run(global_step), 0])):

            sess.run(op_increment)
            monitor()

            try:
                sess.run(op_ngs)
                sess.run(op_increase_gamma_inners)

            except tf.errors.InvalidArgumentError:
                g_old = sess.run(gamma_inners[0])
                sess.run(op_decrease_gamma_inners)
                g_new = sess.run(gamma_inners[0])
                s = "gamma = {} on iteration {} is too big! Falling back to {}"
                print(s.format(it, g_old, g_new))

            sess.run(op_ng)
            sess.run(op_adam)

    model.anchor(sess)


if not os.path.isdir(ARGS.figs_path):
    os.mkdir(ARGS.figs_path)


X, Y = make_data()
print("len", len(X))

model = make_model(X, Y)
try:
    train(model)
except:
    model.anchor(model.enquire_session())
plot_posterior(model, X, Y)
