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
import tensorflow_probability as tfp
import pickle as pkl
from sympy.combinatorics import Permutation
import itertools

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

epsilon = 1.e-6  # for interactions

if m[0:3] == "1Ha":  # relevant system properties
    d = int(m[3:])
    p = int(m[3:])
    spin_up = args.spin_up
    spin_down = args.spin_down
    if p != spin_up + spin_down:
        raise ValueError("Number of fermions doesn't match the number of spin up particles and spin down particles")

    nuclei = []
    real_energy = 0.5 * spin_up ** 2 + 0.5 * spin_down ** 2


    @tf.function(jit_compile=True)
    def potential(x):
        return 0.5 * tf.reduce_sum(x ** 2, axis=1, keepdims=True)

space_d = d // p


@tf.function(jit_compile=True, experimental_relax_shapes=True)
def dis(r1, r2=tf.zeros(d // p)):  # normal of a vector
    return tf.sqrt(tf.reduce_sum((r1 - r2) ** 2, axis=1, keepdims=True)) + epsilon


# symmetry functions
perm_up = list(itertools.permutations(range(spin_up)))
pairity_up = []
for i in perm_up:
    pairity_up.append([Permutation(i).parity()])
pairity_up = tf.constant((-1.) ** np.array(pairity_up))
perm_up = tf.constant(perm_up, dtype=tf.int32)

perm_down = list(itertools.permutations(range(spin_down)))
pairity_down = []
for i in perm_down:
    pairity_down.append([Permutation(i).parity()])
pairity_down = tf.constant((-1.) ** np.array(pairity_down))
perm_down = tf.constant(perm_down, dtype=tf.int32)

blank_perm = np.array([range(d // p)])
blank_perm = tf.cast(tf.tile(blank_perm, [1, p]), tf.int32)
pairity = tf.tile(pairity_up, [perm_down.shape[0], 1]) * tf.repeat(pairity_down, perm_up.shape[0], axis=0)
perm = tf.concat([tf.tile(perm_up, [perm_down.shape[0], 1]), tf.repeat(perm_down, perm_up.shape[0], axis=0) + spin_up], axis=1)


@tf.function(jit_compile=True, experimental_relax_shapes=True)
def permute(x): # returns random permuted state
    sample_perms = tf.random.uniform([tf.shape(x)[0], ], maxval=tf.shape(perm)[0], dtype=tf.int32)
    perms = tf.gather(perm, sample_perms)
    axis2 = d // p * tf.repeat(perms, d // p, axis=1)
    x_perm = tf.gather(x, blank_perm + axis2, axis=1, batch_dims=1)
    y_sign = tf.gather(pairity, sample_perms)
    return x_perm, tf.cast(y_sign, tf.float32)


@tf.keras.utils.register_keras_serializable()
class boundary(keras.layers.Layer):  # boundary layer
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
        return self.a * tf.exp(-(tf.reduce_sum(x ** 2., axis=1, keepdims=True)) / self.sigma)


def build_model(layers_num, size_num):  # building the model
    inputs = keras.Input(shape=([d, ]), name="input")
    hidden = layers.Dense(size_num, activation='elu', kernel_initializer='uniform')(inputs)
    for i in range(layers_num - 1):
        hidden = layers.Dense(size_num, activation='elu', kernel_initializer='uniform')(hidden)
    output = layers.Dense(1, activation='linear')(hidden)
    outputs = layers.multiply([output, boundary()(inputs)])
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


@tf.function(jit_compile=True, experimental_relax_shapes=True)
def body_potential(x):  # coulomb interactions
    value = 0.
    for i in range(p):  # e-e
        for j in range(p):
            if j > i:
                value += 1. / dis(x[:, i * d // p:(i + 1) * d // p], x[:, j * d // p:(j + 1) * d // p])
    return value



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
    def energy_loss(x, y): # monte carlo integration
        V = interactions * body_potential(x) + potential(x)
        T = (-1 / 2) * d22(x)
        O = V * y ** 2 + T * y
        N = y ** 2
        weight = y ** 2 + epsilon
        return tf.reduce_mean(O / weight) / (tf.reduce_mean(N / weight) + epsilon)

    @tf.function(experimental_relax_shapes=True)
    def symmetry_loss(x, y): # loss function to punish for un-symmetry
        new_x, sign = permute(x)
        new_y = sign * model(new_x)
        return tf.reduce_mean((y - new_y) ** 2.) / tf.reduce_mean(y ** 2. + epsilon)

    @tf.function(experimental_relax_shapes=True)
    def total_loss(x, y):
        return symmetry_loss(x, y) * 1.e3 + energy_loss(x, y)

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
            loss_historysym.append(1.e-3 * (history_callback.history["loss"] - energyi))
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
    plt.savefig('symmetry' + m + '.pdf')
    plt.clf()

    plt.plot(loss_history, label='Energy')
    plt.axhline(y=real_energy, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel(r'$<E>$')
    plt.savefig('energy' + m + '.pdf')
    plt.clf()
    data = [loss_historysym, loss_history, energy_mean, err]
    with open('data' + m + '.pkl', 'wb') as f:
        pkl.dump(data, f)


run()
