#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
import datetime

parser = argparse.ArgumentParser(description='Run a VMC simulation')
parser.add_argument(
    '-N',
    '--n_sample',
    type=int,
    help='Sampling number',
    required=True)
parser.add_argument(
    '-max',
    '--x_max',
    type=float,
    help='Border of the coordinates',
    required=True)
parser.add_argument(
    '-m',
    '--mode',
    type=str,
    help='System: Ha: harmonic, He: helium, H: hydrogen',
    required=True)
parser.add_argument(
    '-min',
    '--x_min',
    type=float,
    help='Border of the coordinates',
    required=True)
parser.add_argument(
    '-int',
    '--interactions',
    type=float,
    help='1-yes, 0-no',
    required=False,
    default=1.)
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

# Setting variables
args = parser.parse_args()
n_samples = args.n_sample
x_max = args.x_max
x_min = args.x_min
m = args.mode
interactions = args.interactions

epsilon = 1.e-6
n_layers = 5
layer_size = 128
activation_function = "tanh"

if m[0:3] == "2Ha":
    d = int(m[3:])*2
    p = int(m[3:])
    spin_up = args.spin_up
    spin_down = args.spin_down
    nuclei = []
    nuclei_charge = []
    energy_list = [1, 3, 5, 8, 11, 14, 18, 22, 26, 30]
    real_energy = energy_list[spin_up - 1]
    if spin_down > 0:
        real_energy += energy_list[spin_down-1]

    @tf.function(jit_compile=True, experimental_relax_shapes=True)
    def potential(x):
        return 0.5 * tf.reduce_sum(x ** 2, axis=1, keepdims=True)

elif m[0:3] == "1Ha":
    d = int(m[3:])
    p = int(m[3:])
    spin_up = args.spin_up
    spin_down = args.spin_down
    nuclei = []
    real_energy = 0.5*spin_up**2 + 0.5*spin_down**2

    @tf.function(jit_compile=True)
    def potential(x):
        return 0.5 * tf.reduce_sum(x * x, axis=1, keepdims=True)



elif m[0:3] == "H":
    d = 3
    p = 1
    spin_up = 1
    spin_down = 0
    nuclei = []
    real_energy = -0.5

    @tf.function(jit_compile=True)
    def potential(x):
        return - 1. / dis(x)

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



@tf.keras.utils.register_keras_serializable()
class boundary111(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(boundary111, self).__init__(name=name)
        super(boundary111, self).__init__(**kwargs)
        self.sigma = tf.Variable(initial_value=4., trainable=True)
        self.a = tf.Variable(initial_value=1., trainable=True)

    def get_config(self):
        config = super().get_config()
        return config

    @tf.function(experimental_relax_shapes=True, jit_compile=True)
    def call(self, x):
        # new_x, new_parity = reorder_more_d(x)
        return self.a * tf.exp(-(tf.reduce_sum((x)**2., axis=1, keepdims=True)) / self.sigma)


def build_model(layers_num, size_num):
    inputs = keras.Input(shape=([d, ]), name="input")
    bound = boundary111()(inputs)
    hidden = layers.Dense(size_num, activation=activation_function)(inputs)
    for i in range(layers_num-1):
        hidden = layers.Dense(size_num, activation=activation_function)(hidden)
    output = layers.Dense(1, activation='linear')(hidden) * bound
    model = keras.Model(inputs=[inputs], outputs=[output/tf.reduce_max(tf.abs(output))])
    return model



@tf.function(jit_compile=True, experimental_relax_shapes=True)
def body_potential(x):
    value = 0.
    for i in range(p):  # e-e
        for j in range(p):
            if j > i:
                value += 1. / dis(x[:, i * d // p:(i + 1) * d // p], x[:, j * d // p:(j + 1) * d // p])
    for i in range(len(nuclei)):  # p-p
        for j in range(len(nuclei)):
            if j > i:
                value += nuclei_charge[i] * nuclei_charge[j] / dis(nuclei[i], nuclei[j])
    for i in range(p):  # e-p
        for j in range(len(nuclei)):
            value += -nuclei_charge[j] / dis(x[:, i * d // p:(i + 1) * d // p], nuclei[j])
    return value


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
            grad2y = tf.concat((grad2y, tf.stack(tf.gradients(_grady[i], _x_[i]))[0]), axis=1)
        return tf.reduce_sum(grad2y, axis=1, keepdims=True)


    @tf.function(experimental_relax_shapes=True)
    def energy_loss(x, y):
        V = interactions * body_potential(x) + potential(x)
        T = (-1 / 2) * d2(x)
        O = V * y ** 2 + T * y
        weight = 1.e-10 + y ** 2.
        return tf.reduce_mean(O / weight) + 1.e-10

    @tf.function(experimental_relax_shapes=True, jit_compile=True)
    def target(x):
        psi_sqrd = model(x) ** 2.
        return tf.math.log(psi_sqrd)

    @tf.function(experimental_relax_shapes=True)
    def metropolis(samples, n):
        x = tf.random.uniform([samples, d], minval=x_min, maxval=x_max)
        # x2 = tf.random.normal([samples // 2, d])
        # x = tf.concat([x1, x2], axis=0)
        sample = tfp.mcmc.sample_chain(
            num_results=1,
            current_state=x,
            kernel=tfp.mcmc.RandomWalkMetropolis(target, tfp.mcmc.random_walk_normal_fn(
                    name=None
                    )),
            num_burnin_steps=n,
            trace_fn=None, parallel_iterations=1)
        return sample[0]

    epochs = 1000
    batches = 1
    initial_lr = 0.0001
    opt = tf.optimizers.Adam()
    # opt = tfp.optimizer.VariationalSGD(
    #                                    n_samples // batches, n_samples, max_learning_rate=initial_lr,
    #                                                    preconditioner_decay_rate=0.95, burnin=25, burnin_max_learning_rate=1.e-6,
    #                                                                    use_single_learning_rate=False, name=None)
    # opt = keras.optimizers.SGD(1.e-4, decay=1.e-5, momentum=0.9)
    x = metropolis(n_samples, 200)
    loss_history = [energy_loss(x, model(x)).numpy()]
    model.compile(loss=energy_loss, optimizer=opt, metrics=['mae', 'mean_squared_error'])
    try:  # If it's too long, one can stop the training with ctrl+c and still get a graph
        for i in range(epochs):
            print(i)
            x = metropolis(n_samples, 200)
            history_callback = model.fit(x, x, epochs=1, batch_size=n_samples // batches, shuffle=False)
            loss_history.append(history_callback.history["loss"][0]) #fixme add symmetry?
            # lr = initial_lr * tf.exp(-i / 100)
            # K.set_value(model.optimizer.learning_rate, lr)
    except KeyboardInterrupt:
        pass

    plt.plot(loss_history, label='Energy')
    plt.axhline(y=real_energy, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel(r'$<E>$')
    plt.savefig('energy.pdf')
    plt.clf()

    a = np.linspace(x_min, x_max, n_samples)
    x = tf.reshape(tf.Variable(a, dtype=float), [n_samples, 1])
    y = (model(x)).numpy()[:, 0]
    plt.plot(x.numpy(), y)
    plt.savefig('plot.pdf')
run()