#!/usr/bin/env python
import numpy as np
import math
import random
from math import pi 
import matplotlib.pyplot as plt
import itertools
from sympy.combinatorics import Permutation
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, losses, regularizers 
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Multiply, Reshape
from tensorflow.keras.optimizers import SGD
from datetime import datetime
import os
from os.path import join as pjoin


@tf.function(jit_compile=True)
def target(x):
    psi_sqrd = model(x) ** 2
    return tf.math.log(psi_sqrd)


@tf.function(experimental_relax_shapes=True)
def exact_metropolis(n):
    x = tf.random.uniform([n_samples, d], minval=x_min, maxval=x_max)
    sample = tfp.mcmc.sample_chain(
        num_results=1,
        current_state=x,
        kernel=tfp.mcmc.RandomWalkMetropolis(target),
        num_burnin_steps=n,
        trace_fn=None, parallel_iterations=1)
    return sample[0]

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

class symmetry_sampler(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        init_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        init_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_x, sign_y = reorder_more_d(init_x)
        batch_y = init_y * sign_y
        return np.array(batch_x), np.array(batch_y)

# dataset = tf.data.Dataset.from_generator(symmetry_sampler,
#                                          output_signature=(
#                                              tf.TensorSpec(shape=(None, d), dtype=tf.float32),
#                                              tf.TensorSpec(shape=(None, 1), dtype=tf.float32)), args=(x_fit, y_fit, batch_size))


# history_callback = model.fit(symmetry_sampler(x, y, batch_size), epochs=epochs, batch_size=batch_size, shuffle=True)


class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, n_particles, dim_physical, sigma_init):
        super(GaussianLayer, self).__init__()
        self.sigma = tf.Variable(initial_value=sigma_init, trainable=True)
        self.n_particles = tf.Variable(initial_value=n_particles, trainable=False)
        self.dim_physical = tf.Variable(initial_value=dim_physical, trainable=False)
    def build(self, input_shape):
        pass

    def call(self, input, bosonic):
        r1 = tf.slice(input, [0, 0], [-1, self.dim_physical])
        r2 = tf.slice(input, [0, self.dim_physical], [-1, self.dim_physical])
        def psi(x, n):
            # return (x**n) * K.exp(-(x * x) / (2.0 * tf.math.square(self.sigma))) / tf.math.sqrt(2.0 * tf.constant(pi)) / self.sigma
            return (x**n) * K.exp(-(x * x) / (tf.constant(2.0) * self.sigma))
        def wavefunc_0(x):
            return tf.reduce_prod(psi(x,0), axis=1, keepdims=True)
        return wavefunc_0(r1)*wavefunc_0(r2)


def neural_fit(x, psi, n_samples, perm_sub, perm, parity, analysis_data, iteration, load_weights, bosonic, U, n_particles, dim_physical, n_layers, layer_size, epochs, batch_size, reg, normalize=True):
    tf.keras.backend.clear_session()    
    activation = 'gelu' 
    # activation = 'tanh' 
    dim = x.shape[1]
    
    def create_model():
        input = tf.keras.Input(shape=(dim,), name='inputs')
        print('input :' , input)
        hidden = layers.Dense(layer_size, input_dim=dim, activation=activation, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(reg))(input)
        for i in range(n_layers):
            hidden = layers.Dense(layer_size, activation=activation, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(reg))(hidden)
        boundary = GaussianLayer(n_particles, dim_physical, sigma_init=2.0)(input, bosonic)
        output_initial = layers.Dense(units=1, activation='linear', name='outputs')(hidden)
        output = keras.layers.multiply([output_initial, boundary])

        model = keras.Model(inputs=[input], outputs=[output])
        opt = keras.optimizers.SGD(lr=0.2, decay=1e-5, momentum=0.9, nesterov=False)
        
        print("Compiling model...")
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mae', 'accuracy'])
        return model 

    model = create_model()
    
    dir = os.getcwd()
    if bosonic == 1:
        dir_to_load_and_save = pjoin(dir, "{}Nboson_checkpoints_U{}/my_checkpoint_{}".format(dim, U, iteration))
    else: 
        dir_to_load_and_save = pjoin(dir, "{}Nfermion_checkpoints_U{}/my_checkpoint_{}".format(dim, U, iteration))
    if load_weights == 1:
        model.load_weights(dir_to_load_and_save)
    else:
        print("Fitting...")
        history = model.fit(x, psi, epochs=epochs, batch_size=batch_size, validation_split=0.33)
        analysis_data['history'] = history
        model.save_weights(dir_to_load_and_save)
        
        # Plot learning curves
        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.yscale('log')
        plt.savefig('learning_curves.pdf')
        plt.close()
    
    def fitfunc(x, batch_size=128):
        return model.predict(x, batch_size)

    @tf.function(experimental_relax_shapes=True)
    def d22(x):
        _x = tf.unstack(x, axis=1)
        _x_ = [tf.expand_dims(i, axis=1) for i in _x]
        _x2 = tf.transpose(tf.stack(_x_))[0]
        y = model(_x2)
        _grady = [tf.squeeze(tf.gradients(y, i)) for i in _x]
        grad2y = tf.stack(tf.gradients(_grady[0], _x_[0]))[0]
        for i in range(1, dim):
            grad2y = tf.concat((grad2y, tf.stack(tf.gradients(_grady[i], _x_[i]))[0]), axis=1)
        grad2 = tf.reduce_sum(grad2y, axis=1, keepdims=True)
        return grad2

    return fitfunc, d22

if __name__ == '__main__':
    # Parameters:
    nsamples = 100
    epochs = 200
    batch_size = 500 # 128
    reg = 1e-8
    h = 1e-2
    n_layers = 1
    layer_size = 1*128
    bosonic = 0
    iteration = 0
    n_particles = 2
    dim_physical = 1
    perm_sub = 1
    load_weights = 0     

    from sample_distribution_Nd import sample_mixed
    import stochastic_propagation_Nd
    import sample_wavefunction_Nd
    import harmonic_oscillator_Nd as ho

    U = 0.0 
    nu = 0.001
    hbar = 1.0
    m = 1.0
    omega = 1.0
    tmax = 1.0
    n_t = 100
    offset = [0.0, 0.0]
 
    t = -1j * np.linspace(0, tmax, n_t)
    dt = t[1] - t[0]

    def wavefunction(x):
        return ho.eigenfunction_samples(0, x, n_particles, dim_physical, offset, hbar, m, omega, bosonic=bosonic)
    def wavefunction_d2(x, h):
        psi_ph = np.zeros(x.shape, dtype=complex)
        psi_mh = np.zeros(x.shape, dtype=complex)
        for i in range(0, dim_physical*n_particles):
            x_ph = x.copy()
            x_mh = x.copy()
            x_ph[:,i] += h 
            x_mh[:,i] -= h
            psi_ph[:,i] = wavefunction(x_ph)
            psi_mh[:,i] = wavefunction(x_mh)
        return (psi_ph.sum(axis=-1) + psi_mh.sum(axis=-1) - 2 * (dim_physical*n_particles) * wavefunction(x)) / (h**2)
    def P0(x):
        return np.real(np.conj(wavefunction(x) * wavefunction(x)))
    def eval_V(x):
        return ho.potential(x, m, omega)
    def eval_I(x):
        return ho.coulomb(x, U, nu)
    def Hpsi(x):
            return -(hbar ** 2) / (2 * m) * wavefunction_d2(x, h) + eval_V(x) * wavefunction(x)  + eval_I(x) * wavefunction(x)

    x_max = 5.0
    step = np.full(dim_physical*n_particles, x_max) 
    x0 = np.zeros(dim_physical*n_particles)
    decorrelation_steps = 10
    uniform_ratio = 0.1
    
    print("Generating samples...")
    start = datetime.now()
    samples = sample_mixed(P0, x0, step, x_max, nsamples, decorrelation_steps, uniform_ratio)
    #samples = sample_from_distribution(P0, x0, step, nsamples, decorrelation_steps)
    print("sampling time = ", datetime.now()-start)
    energy_exact, mse_exact = sample_wavefunction_Nd.vec_sample_energy(wavefunction, wavefunction_d2, Hpsi, x0, step, nsamples, decorrelation_steps, x_max)
    print("exact energy: ", energy_exact, "mse_exact: ", mse_exact)
    print("sampling time = ", datetime.now()-start)
    
    d = dim_physical * n_particles
    perm = list(itertools.permutations(range(n_particles)))
    parity = []
    for i in perm:
        parity.append([Permutation(i).parity()])

    psi = wavefunction(samples).real
    average_value = np.max(np.abs(psi))
    psi = psi.reshape(psi.shape[0],1) / average_value
    
    # Fit a neural network to it:
    start = datetime.now()
    analysis_data = {}
    nn_fitfunc, d2_nn_fitfunc= neural_fit(samples, psi, nsamples, perm_sub, perm, parity, analysis_data, iteration, load_weights, bosonic, U, n_particles, dim_physical, n_layers, layer_size, epochs, batch_size, reg)
    print("training_duration = ", datetime.now()-start)
    start = datetime.now()

    def eval_psi(x):
        return nn_fitfunc(x)[:,0]
    def eval_d2psi(x):
        f_d2 = (d2_nn_fitfunc(x).numpy())[:,0]
        return f_d2
    def Hpsi(x):
            return -(hbar ** 2) / (2 * m) * eval_d2psi(x) + eval_V(x) * eval_psi(x)  + eval_I(x) * eval_psi(x)
    decorrelation_steps = 10
    nsamples = 10*nsamples
    energy, mse = sample_wavefunction_Nd.vec_sample_energy(eval_psi, eval_d2psi, Hpsi, x0, step, nsamples, decorrelation_steps, x_max)
    print("exact energy: ", energy_exact, "exact mse: ", mse_exact)
    print("NN energy: ", energy, "NN mse: ", mse)
    print("calculation_time_energy = ", datetime.now()-start)

