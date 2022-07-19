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
import pickle as pkl

parser = argparse.ArgumentParser(description='Run a VMC simulation')
parser.add_argument(
    '-N',
    '--n_sample',
    type=int,
    help='Number of Monte Carlo samples',
    required=True)
parser.add_argument(
    '-max',
    '--x_max',
    type=float,
    help='Upper Border of the coordinates (all dimensions)',
    required=True)
parser.add_argument(
    '-min',
    '--x_min',
    type=float,
    help='Lower Border of the coordinates (all dimensions)',
    required=True)
parser.add_argument(
    '-m',
    '--mode',
    type=str,
    help='The mode is 1HaX where X is the number of fermions',
    required=True)
parser.add_argument(
    '-int',
    '--interactions',
    type=float,
    help='1-yes, 0-no',
    required=False,
    default=0.)
parser.add_argument(
    '-up',
    '--spin_up',
    type=int,
    help='number of particles with spin up',
    required=True)
parser.add_argument(
    '-down',
    '--spin_down',
    type=int,
    help='number of particles with spin down',
    required=True
)
parser.add_argument(
    '-ls',
    '--layer_size',
    type=int,
    help='Number of neurons in the deep layer',
    required=False,
    default=128)
parser.add_argument(
    '-ln',
    '--layer_number',
    type=int,
    help='Number of deep layers in the deep layer',
    required=False,
    default=3)
parser.add_argument(
    '-af',
    '--activation_function',
    type=str,
    help='Activation function used',
    required=False,
    default='elu')
parser.add_argument(
    '-epochs',
    '--number_of_epochs',
    type=int,
    help='Training epochs',
    required=False,
    default=50)

# Setting variables
args = parser.parse_args()
n_samples = args.n_sample
x_max = args.x_max
x_min = args.x_min
m = args.mode
interactions = args.interactions
layer_size = args.layer_size
n_layers = args.layer_number
activation_function = args.activation_function
epochs = args.number_of_epochs

epsilon = 1.e-6 #for interactions



if m[0:3] == "1Ha":
    d = int(m[3:])
    p = int(m[3:])
    spin_up = args.spin_up
    spin_down = args.spin_down
    if p != spin_up+spin_down:
        raise ValueError("Number of fermions doesn't match the number of spin up particles and spin down particles")

    nuclei = []
    real_energy = 0.5*spin_up**2 + 0.5*spin_down**2

    @tf.function(jit_compile=True)
    def potential(x):
        return 0.5 * tf.reduce_sum(x ** 2, axis=1, keepdims=True)


space_d = d//p


@tf.function(jit_compile=True, experimental_relax_shapes=True)
def dis(r1, r2=tf.zeros(d // p)):
    return tf.sqrt(tf.reduce_sum((r1 - r2) ** 2, axis=1, keepdims=True)) + epsilon


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
                result = result * ((tf.gather(x, in_i, batch_dims=1) - tf.gather(x, in_j, batch_dims=1)) / (x_i - x_j))
    return result


@tf.keras.utils.register_keras_serializable()
class boundary(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(boundary, self).__init__(name=name)
        super(boundary, self).__init__(**kwargs)
        self.sigma = tf.Variable(initial_value=2., trainable=True)
        self.a = tf.Variable(initial_value=1., trainable=True)


    def get_config(self):
        config = super().get_config()
        return config

    @tf.function(jit_compile=True)
    def call(self, x):
        return self.a * tf.exp(-(tf.reduce_sum(x**2., axis=1, keepdims=True)) / self.sigma)


def build_model(layers_num, size_num):
    inputs = keras.Input(shape=([d, ]), name="input")
    hidden = layers.Dense(size_num, activation='elu', kernel_initializer='uniform')(inputs)
    for i in range(layers_num-1):
        hidden = layers.Dense(size_num, activation='elu', kernel_initializer='uniform')(hidden)
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
                value += 1. / dis(x[:, i * d // p:(i + 1) * d // p], x[:, j * d // p:(j + 1) * d // p])
    return value


@tf.function(experimental_relax_shapes=True)
def reorder_more_d(x):
    n = tf.shape(x)[0]
    if spin_up > 0:
        x_up = x[:, 0 * (d // p):(d // p) * spin_up]
        new_shaped_x_up = tf.reshape(x_up, [n, spin_up, space_d])
        s_up = tf.reduce_sum(new_shaped_x_up * (1000.**tf.reverse(tf.range(space_d, dtype=tf.float32), [0])), axis=2)
        val, idx_up = tf.nn.top_k(s_up, spin_up)
        new_x_up = tf.reshape(tf.gather(new_shaped_x_up, tf.expand_dims(idx_up, axis=-1), batch_dims=1), [n, spin_up*space_d])
        new_parity_up = tf.cast(parity(idx_up, spin_up), tf.float32)

    if spin_down > 0:
        x_down = x[:, spin_up * (d // p):]
        new_shaped_x_down = tf.reshape(x_down, [n, spin_down, space_d])
        s_down = tf.reduce_sum(new_shaped_x_down * (1000.**tf.reverse(tf.range(space_d, dtype=tf.float32), [0])), axis=2)
        val, idx_down = tf.nn.top_k(s_down, spin_down)
        new_x_down = tf.reshape(tf.gather(new_shaped_x_down, tf.expand_dims(idx_down, axis=-1), batch_dims=1), [n, spin_down*space_d])
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
    def d22(x):  # We thank Zhi-Qin John Xu's implementation of the laplacian that can be found in
        # https://github.com/xuzhiqin1990/laplacian
        _x = tf.unstack(x, axis=1)
        _x_ = [tf.expand_dims(i, axis=1) for i in _x]
        _x2 = tf.transpose(tf.stack(_x_))[0]
        y = model(_x2)
        _grady = [tf.squeeze(tf.gradients(y, i)) for i in _x]
        grad2y = tf.stack(tf.gradients(_grady[0], _x_[0]))[0]
        for i in range(1, d):
            grad2y = tf.concat((grad2y, tf.stack(tf.gradients(_grady[i], _x_[i]))[0]), axis=1)
        return tf.reduce_sum(grad2y, axis=1, keepdims=True)

    @tf.function(experimental_relax_shapes=True)
    def energy_loss(x, y):
        V = interactions * body_potential(x) + potential(x)
        T = (-1 / 2) * d22(x)
        O = V * y ** 2 + T * y
        N = y ** 2
        weight = y ** 2 + epsilon
        return tf.reduce_mean(O / weight) / (tf.reduce_mean(N / weight) + epsilon)

    @tf.function(experimental_relax_shapes=True)
    def symmetry_loss(x, y):
        new_x, sign = reorder_more_d(x)
        new_y = sign * model(new_x)
        return tf.reduce_mean((y - new_y)**2.) /tf.reduce_mean(y**2. + epsilon)

    @tf.function(experimental_relax_shapes=True)
    def total_loss(x, y):
            return symmetry_loss(x, y)*1.e3 + energy_loss(x, y)

    @tf.function(jit_compile=True)
    def target(x):
        psi_sqrd = model(x) ** 2
        return tf.math.log(psi_sqrd)

    @tf.function(experimental_relax_shapes=True)
    def exact_metropolis(N, n):
        x = tf.random.uniform([N, d], minval=x_min, maxval=x_max)
        sample = tfp.mcmc.sample_chain(
            num_results=1,
            current_state=x,
            kernel=tfp.mcmc.RandomWalkMetropolis(target),
            num_burnin_steps=n,
            trace_fn=None, parallel_iterations=1)
        return sample[0]

    opt = keras.optimizers.Adam(0.001)
    x = tf.random.uniform([n_samples, d], maxval=x_max, minval=x_min)
    loss_historysym = []

    model.compile(loss=total_loss, optimizer=opt, metrics=['mae', 'mean_squared_error'])
    batches = 1
    loss_history = [energy_loss(x, model(x)).numpy()]
    try:  # If it's too long, one can stop the training with ctrl+c and still get a graph
        for i in range(epochs):
            print(i)
            x = exact_metropolis(n_samples, 500)
            history_callback = model.fit(x, x, epochs=1, batch_size=n_samples // batches, shuffle=False)
            energyi = energy_loss(x, model(x)).numpy()
            print(energyi)
            loss_history.append(energyi)
            loss_historysym.append(1.e-3*(history_callback.history["loss"]-energyi))
    except KeyboardInterrupt:
        pass

    def energy(x, y, num):
        V = interactions * body_potential(x) + potential(x)
        d2_list = []
        for i in range(int(tf.math.ceil(num / 10000))):
            d2_list.append(d22(x[i * 10000:(i + 1) * 10000, :]))
        dd2 = tf.concat(d2_list, axis=0)
        T = -0.5 * dd2
        O = V * y ** 2 + T * y
        N = y ** 2 + epsilon
        weight = y ** 2 + epsilon
        part1 = O / weight
        part2 = N / weight
        en = tf.reduce_mean(part1) / (tf.reduce_mean(part2))
        err = (1. / num) ** 0.5 * tf.math.reduce_std(part1)
        return en, err

    x = exact_metropolis(100000, 500)
    energy_mean, err = energy(x, model(x), 100000)
    print(energy_mean, err)

    plt.semilogy(loss_historysym)
    plt.axhline(y=0., color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel("Symmetry")
    plt.savefig('symmetry'+m+'.pdf')
    plt.clf()

    plt.plot(loss_history, label='Energy')
    plt.axhline(y=real_energy, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel(r'$<E>$')
    plt.savefig('energy'+m+'.pdf')
    plt.clf()
    data = [loss_historysym, loss_history, energy_mean, err]
    with open('data' + m + '.pkl', 'wb') as f:
        pkl.dump(data, f)
run()