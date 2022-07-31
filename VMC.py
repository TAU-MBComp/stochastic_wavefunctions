#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib

matplotlib.use('WebAgg')
from matplotlib import pyplot as plt
import argparse
import time
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
import datetime

parser = argparse.ArgumentParser(description='Run a VMC simulation')
parser.add_argument('-N',
                    '--n_sample',
                    type=int,
                    help='Sampling number',
                    required=True)
parser.add_argument('-max',
                    '--x_max',
                    type=float,
                    help='Border of the coordinates',
                    required=True)
parser.add_argument('-m',
                    '--mode',
                    type=str,
                    help='System: Ha: harmonic, He: helium, H: hydrogen',
                    required=True)
parser.add_argument('-min',
                    '--x_min',
                    type=float,
                    help='Border of the coordinates',
                    required=True)
parser.add_argument('-int',
                    '--interactions',
                    type=float,
                    help='1-yes, 0-no',
                    required=False,
                    default=0.)
parser.add_argument('-up',
                    '--spin_up',
                    type=int,
                    help='number of particles with spin up',
                    required=True)
parser.add_argument('-down',
                    '--spin_down',
                    type=int,
                    help='number of particles with spin down',
                    required=True)

# Setting variables
args = parser.parse_args()
n_samples = args.n_sample
x_max = args.x_max
x_min = args.x_min
m = args.mode
interactions = args.interactions

epsilon = 1.e-6
n_layers = 4
layer_size = 128
activation_function = "elu"

if m[0:3] == "2Ha":
    d = int(m[3:]) * 2
    p = int(m[3:])
    spin_up = args.spin_up
    spin_down = args.spin_down
    nuclei = []
    nuclei_charge = []
    energy_list = [1, 3, 5, 8, 11, 14, 18, 22, 26, 30]
    real_energy = energy_list[spin_up - 1]
    if spin_down > 0:
        real_energy += energy_list[spin_down - 1]

    @tf.function(jit_compile=True, experimental_relax_shapes=True)
    def potential(x):
        return 0.5 * tf.reduce_sum(x**2, axis=1, keepdims=True)

elif m[0:3] == "1Ha":
    d = int(m[3:])
    p = int(m[3:])
    spin_up = args.spin_up
    spin_down = args.spin_down
    nuclei = []
    real_energy = 0.5 * spin_up**2 + 0.5 * spin_down**2

    @tf.function(jit_compile=True)
    def potential(x):
        return 0.5 * tf.reduce_sum(x**2, axis=1, keepdims=True)

elif m[0:3] == "H":
    d = 3
    p = 1
    spin_up = 1
    spin_down = 0
    nuclei = []
    real_energy = -0.5

    @tf.function(jit_compile=True)
    def potential(x):
        return -1. / dis(x)


space_d = d // p


@tf.function(jit_compile=True, experimental_relax_shapes=True)
def dis(r1, r2=tf.zeros(d // p)):
    return tf.sqrt(tf.reduce_sum(
        (r1 - r2)**2, axis=1, keepdims=True)) + epsilon


@tf.function(jit_compile=True, experimental_relax_shapes=True)
def body_potential(x):
    value = 0.
    for i in range(p):  # e-e
        for j in range(p):
            if j > i:
                value += 1. / dis(x[:, i * d // p:(i + 1) * d // p],
                                  x[:, j * d // p:(j + 1) * d // p])
    for i in range(len(nuclei)):  # p-p
        for j in range(len(nuclei)):
            if j > i:
                value += nuclei_charge[i] * nuclei_charge[j] / dis(
                    nuclei[i], nuclei[j])
    for i in range(p):  # e-p
        for j in range(len(nuclei)):
            value += -nuclei_charge[j] / dis(x[:, i * d // p:(i + 1) * d // p],
                                             nuclei[j])
    return value


@tf.function(jit_compile=True)
def full_potential(x):
    return potential(x) + tf.math.multiply_no_nan(body_potential(x),
                                                  interactions)


@tf.function(experimental_relax_shapes=True, jit_compile=True)
def parity(x, spin):
    result = 1.
    range_for = tf.range(spin, dtype=tf.int32)
    for i in range(spin):
        for j in range(spin):
            if i < j:
                x_i = x[:, i:i + 1]
                x_j = x[:, j:j + 1]
                in_i = tf.gather(range_for, x_i)
                in_j = tf.gather(range_for, x_j)
                result = result * ((tf.gather(x, in_i, batch_dims=1) -
                                    tf.gather(x, in_j, batch_dims=1)) /
                                   (x_i - x_j))
    return result


@tf.function(experimental_relax_shapes=True)
def reorder_more_d(x):
    n = tf.shape(x)[0]
    if spin_up > 0:
        x_up = x[:, 0 * (d // p):(d // p) * spin_up]
        new_shaped_x_up = tf.reshape(x_up, [n, spin_up, space_d])
        s_up = tf.reduce_sum(
            new_shaped_x_up *
            (1000.**tf.reverse(tf.range(space_d, dtype=tf.float32), [0])),
            axis=2)
        val, idx_up = tf.nn.top_k(s_up, spin_up)
        new_x_up = tf.reshape(
            tf.gather(new_shaped_x_up,
                      tf.expand_dims(idx_up, axis=-1),
                      batch_dims=1), [n, spin_up * space_d])
        new_parity_up = tf.cast(parity(idx_up, spin_up), tf.float32)

    if spin_down > 0:
        x_down = x[:, spin_up * (d // p):]
        new_shaped_x_down = tf.reshape(x_down, [n, spin_down, space_d])
        s_down = tf.reduce_sum(
            new_shaped_x_down *
            (1000.**tf.reverse(tf.range(space_d, dtype=tf.float32), [0])),
            axis=2)
        val, idx_down = tf.nn.top_k(s_down, spin_down)
        new_x_down = tf.reshape(
            tf.gather(new_shaped_x_down,
                      tf.expand_dims(idx_down, axis=-1),
                      batch_dims=1), [n, spin_down * space_d])
        new_parity_down = tf.cast(parity(idx_down, spin_down), tf.float32)

        new_x = tf.concat([new_x_up, new_x_down], axis=1)
        new_parity = new_parity_down * new_parity_up
    else:
        new_x = new_x_up
        new_parity = new_parity_up
    return new_x, new_parity


@tf.keras.utils.register_keras_serializable()
class boundary(keras.layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(boundary, self).__init__(name=name)
        super(boundary, self).__init__(**kwargs)
        self.alpha = tf.Variable(initial_value=2. / (x_max - x_min),
                                 trainable=False,
                                 name="a")
        self.beta = tf.Variable(initial_value=1., trainable=True)
        self.gamma = tf.Variable(initial_value=0., trainable=True, name="c")
        self.delta = tf.Variable(initial_value=2., trainable=True, name="d")
        self.sigma = tf.Variable(initial_value=2., trainable=True)
        self.a = tf.Variable(initial_value=1., trainable=True)
        self.b = tf.Variable(initial_value=1., trainable=True)
        self.listup1 = tf.Variable(initial_value=tf.random.uniform(
            [spin_up, 1], minval=-1, maxval=1),
                                   trainable=True,
                                   name="detup1",
                                   shape=[spin_up, 1])
        self.listup2 = tf.Variable(initial_value=tf.random.uniform([1]),
                                   trainable=True,
                                   name="detup2",
                                   shape=[1])
        self.listdown1 = tf.Variable(initial_value=tf.random.uniform(
            [spin_down, 1], minval=-1, maxval=1),
                                     trainable=True,
                                     name="detdown1",
                                     shape=[spin_down, 1])
        self.listdown2 = tf.Variable(initial_value=tf.random.uniform([1]),
                                     trainable=True,
                                     name="detdown2",
                                     shape=[1])
        # if symmetry == -1:
        #     self.detav = tf.Variable(initial_value=tf.random.uniform([(p-p//2)]), trainable=True, name="detav",
        #                                 shape=[(p-p//2)])
        #     self.detaw = tf.Variable(initial_value=tf.random.uniform([(p-p//2)]), trainable=True, name="detaw",
        #                                 shape=[(p-p//2)])
        #     self.deta_n = tf.Variable(initial_value=tf.zeros([3*(p-p//2)]), trainable=True, name="deta_n",
        #                                 shape=[3*(p-p//2)])
        #     self.detbv = tf.Variable(initial_value=tf.random.uniform([(p//2)]), trainable=True, name="detbv",
        #                                 shape=[(p//2)])
        #     self.detbw = tf.Variable(initial_value=tf.random.uniform([(p//2)]), trainable=True, name="detbw",
        #                                 shape=[(p//2)])
        #     self.detb_n = tf.Variable(initial_value=tf.zeros([3*(p//2)]), trainable=True, name="detb_n",
        #                                 shape=[3*(p//2)])

    def get_config(self):
        config = super().get_config()
        return config

    if m == "box2d":

        def call(self, x):
            x1 = x[:, 0:1]
            x2 = x[:, 1:2]
            return bound_func(x1) * bound_func(
                x2
            )  # * tf.sqrt((x1 - 3. * tf.sin(np.pi *x1 / 5.) - x2)**2. + 0.05)#tf.sqrt((bound_func(x1) * bound_func(x2) * (x1 + 0.7*tf.sin(2.*x1*np.pi) / np.pi -x2))**2 + epsilon**2.)

    if d // p == 3:

        @tf.function(jit_compile=True)
        def call(self, x):
            a = self.alpha
            # tf.print(a)
            b = self.b
            # tf.print(b)
            # c = self.gamma
            # delt = self.delta
            value = 1.
            for i in range(p):
                r = dis(x[:, i * d // p:(i + 1) * d // p])
                value = value * tf.exp(-a**2 * r)
            return value * jastrow(x, b**2)

    elif False:

        @tf.function(experimental_relax_shapes=True, jit_compile=True)
        def call(self, x):
            symdetup = 1.
            symdetdown = 1.
            list_up = self.listup1
            a_up = self.listup2
            list_down = self.listdown1
            a_down = self.listdown2
            if spin_up > 0:
                x_up = x[:, 0 * (d // p):(d // p) * spin_up]
                symdetup = detn(symmetry_detplus(x_up, spin_up, list_up, a_up),
                                spin_up)
            if spin_down > 0:
                x_down = x[:, spin_up * (d // p):(d // p) * p]
                symdetdown = detn(
                    symmetry_detplus(x_down, spin_down, list_down, a_down),
                    spin_down)
            return symdetup * symdetdown

    else:

        @tf.function(jit_compile=True)
        def call(self, x):
            # a = self.alpha
            # # tf.print(a)
            # b = self.b
            # # tf.print(b)
            # # c = self.gamma
            # # delt = self.delta
            # value = 1.
            # for i in range(p):
            #     r = dis(x[:, i * d // p:(i + 1) * d // p])
            #     value = value * tf.exp(- a ** 2. * r ** 2.)
            # return b * tf.reduce_prod(tf.exp(- a *(x) ** 2), axis=1, keepdims=True)# * boundary()(inputs)#tf.reduce_prod(tf.exp(- (2. /((x_max - x_min)*float(d)))* (inputs) ** 2), axis=1, keepdims=True)
            return self.a * tf.exp(-(tf.reduce_sum(
                (x)**2., axis=1, keepdims=True)) / self.sigma)


def build_model(layers_num, size_num):
    inputs = keras.Input(shape=([
        d,
    ]), name="input")
    hidden = layers.Dense(size_num,
                          activation='elu',
                          kernel_initializer='uniform')(inputs)
    for i in range(layers_num - 1):
        hidden = layers.Dense(size_num,
                              activation='elu',
                              kernel_initializer='uniform')(hidden)
    output = layers.Dense(1, activation='linear')(hidden)
    outputs = layers.multiply([output, boundary()(inputs)])
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


@tf.function(jit_compile=True, experimental_relax_shapes=True)
def body_potential(x):
    value = 0.
    for i in range(p):  # e-e
        for j in range(p):
            if j > i:
                value += 1. / dis(x[:, i * d // p:(i + 1) * d // p],
                                  x[:, j * d // p:(j + 1) * d // p])
    for i in range(len(nuclei)):  # p-p
        for j in range(len(nuclei)):
            if j > i:
                value += nuclei_charge[i] * nuclei_charge[j] / dis(
                    nuclei[i], nuclei[j])
    for i in range(p):  # e-p
        for j in range(len(nuclei)):
            value += -nuclei_charge[j] / dis(x[:, i * d // p:(i + 1) * d // p],
                                             nuclei[j])
    return value


@tf.function(experimental_relax_shapes=True)
def reorder_more_d(x):
    n = tf.shape(x)[0]
    if spin_up > 0:
        x_up = x[:, 0 * (d // p):(d // p) * spin_up]
        new_shaped_x_up = tf.reshape(x_up, [n, spin_up, space_d])
        s_up = tf.reduce_sum(
            new_shaped_x_up *
            (1000.**tf.reverse(tf.range(space_d, dtype=tf.float32), [0])),
            axis=2)
        val, idx_up = tf.nn.top_k(s_up, spin_up)
        new_x_up = tf.reshape(
            tf.gather(new_shaped_x_up,
                      tf.expand_dims(idx_up, axis=-1),
                      batch_dims=1), [n, spin_up * space_d])
        new_parity_up = tf.cast(parity(idx_up, spin_up), tf.float32)

    if spin_down > 0:
        x_down = x[:, spin_up * (d // p):]
        new_shaped_x_down = tf.reshape(x_down, [n, spin_down, space_d])
        s_down = tf.reduce_sum(
            new_shaped_x_down *
            (1000.**tf.reverse(tf.range(space_d, dtype=tf.float32), [0])),
            axis=2)
        val, idx_down = tf.nn.top_k(s_down, spin_down)
        new_x_down = tf.reshape(
            tf.gather(new_shaped_x_down,
                      tf.expand_dims(idx_down, axis=-1),
                      batch_dims=1), [n, spin_down * space_d])
        new_parity_down = tf.cast(parity(idx_down, spin_down), tf.float32)

        new_x = tf.concat([new_x_up, new_x_down], axis=1)
        new_parity = new_parity_down * new_parity_up
    else:
        new_x = new_x_up
        new_parity = new_parity_up
    return new_x, new_parity


def run():
    model = build_model(n_layers, layer_size)

    @tf.function(experimental_relax_shapes=True)
    def d2(x):
        _x = tf.unstack(x, axis=1)
        _x_ = [tf.expand_dims(i, axis=1) for i in _x]
        _x2 = tf.transpose(tf.stack(_x_))[0]
        y = model(_x2)
        _grady = [tf.squeeze(tf.gradients(y, i)) for i in _x]
        grad2y = tf.stack(tf.gradients(_grady[0], _x_[0]))[0]
        for i in range(1, d):
            grad2y = tf.concat(
                (grad2y, tf.stack(tf.gradients(_grady[i], _x_[i]))[0]), axis=1)
        return tf.reduce_sum(grad2y, axis=1, keepdims=True)

    def d2(x):
        with tf.GradientTape() as g:
            g.watch(x)
            der1 = d1(x)
        laplace = g.batch_jacobian(der1, x)
        return tf.reduce_sum(tf.linalg.diag_part(laplace),
                             axis=1,
                             keepdims=True)

    def d1(x):
        with tf.GradientTape() as g:
            g.watch(x)
            der1 = model(x)
        return g.gradient(der1, x)

    @tf.function(experimental_relax_shapes=True)
    def d22(x):  # maybe it's good for larger systems?
        _x = tf.unstack(x, axis=1)
        _x_ = [tf.expand_dims(i, axis=1) for i in _x]
        _x2 = tf.transpose(tf.stack(_x_))[0]
        y = model(_x2)
        _grady = [tf.squeeze(tf.gradients(y, i)) for i in _x]
        grad2y = tf.stack(tf.gradients(_grady[0], _x_[0]))[0]
        for i in range(1, d):
            grad2y = tf.concat(
                (grad2y, tf.stack(tf.gradients(_grady[i], _x_[i]))[0]), axis=1)
        return tf.reduce_sum(grad2y, axis=1, keepdims=True)

    @tf.function(experimental_relax_shapes=True)
    def energy_loss(x, y):
        V = interactions * body_potential(x) + potential(x)
        T = (-1 / 2) * d22(x)
        O = V * y**2 + T * y
        N = y**2
        weight = y**2 + epsilon
        return tf.reduce_mean(
            O / weight) / (tf.reduce_mean(N / weight) + epsilon)

    @tf.function(experimental_relax_shapes=True)
    def symmetry_loss(x, y):
        new_x, sign = reorder_more_d(x)
        new_y = model(new_x)
        return tf.reduce_mean(
            (y - sign * new_y)**2.) / tf.reduce_mean(y**2. + epsilon)

    @tf.function(experimental_relax_shapes=True)
    def total_loss(x, y):
        return symmetry_loss(x, y) * 1.e3 + energy_loss(x, y)

    @tf.function(jit_compile=True)
    def target(x):
        psi_sqrd = model(x)**2
        return tf.math.log(psi_sqrd)

    @tf.function(experimental_relax_shapes=True)
    def exact_metropolis(n):
        x = tf.random.uniform([n_samples, d], minval=x_min, maxval=x_max)
        sample = tfp.mcmc.sample_chain(
            num_results=1,
            current_state=x,
            kernel=tfp.mcmc.RandomWalkMetropolis(target),
            num_burnin_steps=n,
            trace_fn=None,
            parallel_iterations=1)
        return sample[0]

    epochs = 1000
    batches = 10
    initial_lr = 0.0001
    # opt = tf.optimizers.Adam()
    # opt = tfp.optimizer.VariationalSGD(
    #     n_samples // batches, n_samples, max_learning_rate=0.01,
    #     preconditioner_decay_rate=0.95, burnin=25, burnin_max_learning_rate=1.e-6,
    #     use_single_learning_rate=False, name=None
    # )
    opt = keras.optimizers.Adam(0.01)
    x = tf.random.uniform([n_samples, d], maxval=x_max, minval=x_min)
    loss_historysym = []

    #metropolis(n_samples, 500)
    batches = 10
    # model.compile(loss=symmetry_loss, optimizer=opt, metrics=['mae', 'mean_squared_error'])
    # try:  # If it's too long, one can stop the training with ctrl+c and still get a graph
    #     aa = 0
    #     for i in range(epochs):
    #         print(i)
    #         aa += 1
    #         x = exact_metropolis(100)
    #         history_callback = model.fit(x, x, epochs=1, batch_size=n_samples // batches, shuffle=False)
    #         loss_historysym.append(history_callback.history["loss"])
    #         # energyi = energy_loss(x, model(x)).numpy()
    #         # print(energyi)
    #         # loss_history.append(energyi)
    #         if loss_historysym[-1][0] < 1.e-4:
    #             break
    #         # lr = initial_lr * tf.exp(-i / 100)
    #         # K.set_value(model.optimizer.learning_rate, lr)
    # except KeyboardInterrupt:
    #     pass
    # weights = model.get_weights()
    # # opt = tfp.optimizer.VariationalSGD(
    # #     n_samples // batches, n_samples, max_learning_rate=1.e-2,
    # #     preconditioner_decay_rate=0.95, burnin=1, burnin_max_learning_rate=1.e-6,
    # #     use_single_learning_rate=False, name=None
    # # )
    model.compile(loss=total_loss,
                  optimizer=opt,
                  metrics=['mae', 'mean_squared_error'])
    # model.set_weights(weights)
    batches = 10
    loss_history = [energy_loss(x, model(x)).numpy()]
    try:  # If it's too long, one can stop the training with ctrl+c and still get a graph
        for i in range(epochs):
            print(i)
            # K.set_value(model.optimizer.learning_rate, 1.e-3)
            x = exact_metropolis(500)
            history_callback = model.fit(x,
                                         x,
                                         epochs=1,
                                         batch_size=n_samples // batches,
                                         shuffle=False)
            energyi = energy_loss(x, model(x)).numpy()
            print(energyi)
            loss_history.append(energyi)
            loss_historysym.append(
                1.e-3 * (history_callback.history["loss"] - energyi))
            # lr = initial_lr * tf.exp(-i / 100)
            # K.set_value(model.optimizer.learning_rate, lr)
    except KeyboardInterrupt:
        pass
    plt.semilogy(loss_historysym)
    plt.axhline(y=0., color='r', linestyle='--', alpha=0.5)
    # plt.axvline(x=aa, color='b', linestyle='--', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel("Symmetry")
    plt.savefig('symmetry.pdf')
    plt.show()
    plt.clf()

    plt.plot(loss_history, label='Energy')
    plt.axhline(y=real_energy, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel(r'$<E>$')
    plt.savefig('energy.pdf')
    plt.show()
    plt.clf()
    if d == 1:
        a = np.linspace(x_min, x_max, n_samples)
        x = tf.reshape(tf.Variable(a, dtype=float), [n_samples, 1])
        y = (model(x)).numpy()[:, 0]
        plt.plot(x.numpy(), y)
        plt.savefig('plot.pdf')


run()
