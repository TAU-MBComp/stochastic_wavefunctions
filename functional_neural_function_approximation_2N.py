#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
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
            return (x**n) * K.exp(-(x * x) / self.sigma)
        def wavefunc_00(x): 
            return tf.reduce_prod(psi(x,0), axis=1, keepdims=True)
        def wavefunc_01(x):
            y = tf.constant(1.0)
            for dim in range(dim_physical):
                if dim == (dim_physical-1):
                    y = y * psi(tf.slice(x, [0, dim], [-1, 1]), 1) 
                else:
                    y = y * psi(tf.slice(x, [0, dim], [-1, 1]), 0) 
            return y 
        if bosonic == 1:
            return wavefunc_00(r1)*wavefunc_00(r2)
        else: 
            return wavefunc_00(r1)*wavefunc_00(r2)
            # return (wavefunc_00(r1)*wavefunc_01(r2) - wavefunc_00(r2)*wavefunc_01(r1)) / 2.0 
 

def neural_fit(x, psi, analysis_data, iteration, load_weights, bosonic, U, n_particles, dim_physical, n_layers, layer_size, epochs, learning_batch_size, reg, normalize=True):
    tf.keras.backend.clear_session()    
    activation = 'tanh' 
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
        # output = output_initial

        model = keras.Model(inputs=[input], outputs=[output])
        opt = keras.optimizers.SGD(lr=1.0, decay=1e-5, momentum=0.8, nesterov=False)
        # opt = keras.optimizers.SGD(lr=0.2, decay=1e-5, momentum=0.9, nesterov=False)
        # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.0, decay_steps=100, decay_rate=2.0, staircase=True)
        # opt = keras.optimizers.SGD(learning_rate=lr_schedule)
        
        print("Compiling model...")
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mae', 'accuracy'])
        return model 

    model = create_model()
    
    dir = os.getcwd()
    if bosonic == 1:
        dir_to_load_and_save = pjoin(dir, "{}Nboson_checkpoints_U{}/my_checkpoint_{}".format(dim, U, iteration))
    else: 
        dir_to_load_and_save = pjoin(dir, "{}Nfermion_checkpoints_U{}/my_checkpoint_{}".format(dim, U, iteration))
    print(dir_to_load_and_save)    
    if load_weights == 1:
        model.load_weights(dir_to_load_and_save)
    else: 
        print("Fitting...")
        history = model.fit(x, psi, epochs=epochs, batch_size=learning_batch_size, validation_split=0.33)
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
    return fitfunc

if __name__ == '__main__':
    # Parameters:
    nsamples = 10000
    epochs = 100
    batch_size = 128
    reg = 1e-8
    h = 1e-2
    n_layers = 3
    layer_size = 128 
    bosonic = 0
    iteration = 0
    n_particles = 2
    dim_physical = 3
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
    # offset = [1.0, -1.0]
    offset = [0.0, -0.0]
 
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

    x_max = 10.0
    step = np.full(dim_physical*n_particles, x_max) 
    x0 = np.zeros(dim_physical*n_particles)
    decorrelation_steps = 5
    uniform_ratio = 0.2
    
    print("Generating samples...")
    start = datetime.now()
    samples = sample_mixed(P0, x0, step, x_max, nsamples, decorrelation_steps, uniform_ratio)
    #samples = sample_from_distribution(P0, x0, step, nsamples, decorrelation_steps)
    print("sampling time = ", datetime.now()-start)
    energy_exact, mse_exact = sample_wavefunction_Nd.vec_sample_energy(wavefunction, wavefunction_d2, Hpsi, x0, step, nsamples, decorrelation_steps, x_max)
    print("exact energy: ", energy_exact, "mse_exact: ", mse_exact)
    print("sampling time = ", datetime.now()-start)
    
    psi = wavefunction(samples).real
    average_value = np.max(np.abs(psi))
    # print(average_value)
    # psi = psi / average_value
    
    # Fit a neural network to it:
    start = datetime.now()
    analysis_data = {}
    nn_fitfunc = neural_fit(samples, np.real(psi).ravel(), analysis_data, iteration, load_weights, bosonic, U, n_particles, dim_physical, n_layers, layer_size, epochs, batch_size, reg)
    print("training_duration = ", datetime.now()-start)
    start = datetime.now()

    def eval_psi(x):
        # return nn_fitfunc(x)[:,0]*average_value
        return nn_fitfunc(x)[:,0]
    def eval_d2psi(x, h):
        psi_ph = np.zeros(x.shape, dtype=complex)
        psi_mh = np.zeros(x.shape, dtype=complex)
        for i in range(0,dim_physical*n_particles):
             x_ph = x.copy()
             x_mh = x.copy()
             x_ph[:,i] += h 
             x_mh[:,i] -= h 
             psi_ph[:,i] = eval_psi(x_ph)
             psi_mh[:,i] = eval_psi(x_mh)
        return (psi_ph.sum(axis=-1) + psi_mh.sum(axis=-1) - 2 * (dim_physical*n_particles) * eval_psi(x)) / (h**2)
    def Hpsi(x):
            return -(hbar ** 2) / (2 * m) * eval_d2psi(x, h) + eval_V(x) * eval_psi(x)  + eval_I(x) * eval_psi(x)
    energy, mse = sample_wavefunction_Nd.vec_sample_energy(eval_psi, eval_d2psi, Hpsi, x0, step, nsamples, decorrelation_steps, x_max)
    print("exact energy: ", energy_exact, "NN energy: ", energy)
    print("exact mse: ", mse_exact, "NN mse: ", mse)
    print("calculation_time_energy = ", datetime.now()-start)

    plt.scatter(samples[:,0], samples[:,1], c=psi/1000)
    plt.savefig('samples.pdf')

    #plt.imshow(pred[:,:,1])
    #plt.show()
    #plt.imshow(psi[:,:,1])
    #plt.show()
