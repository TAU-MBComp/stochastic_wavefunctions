#!/usr/bin/env python
import numpy as np
from math import factorial
from itertools import permutations
import sample_distribution_Nd
import sample_wavefunction_Nd
#import svm_function_approximation
import functional_neural_function_approximation_2N as nn_fit
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import math
import os
import matplotlib.pyplot as plt
import pickle 
import sys
"""
Given functions for evaluating the wavefunction psi, its laplacian d2psi and the
potential V at time t, calculate the wavefunction at each of the sample points x
for time t + dt using the Euler method.
"""
def propagate_samples(psi, d2psi, V, I, dt, x, m, hbar, n_particles, dim):
    psi_x = psi(x)
    d2psi_x = d2psi(x)
    return psi_x - (1j * dt / hbar) * (-(hbar ** 2) / (2 * m) * d2psi_x + (V(x) + I(x)) * psi_x)


"""
Perform time propagation using Runeg-Kutter method. This is not recommended here
because higher derivatives are not smooth.
"""
def RK4(psi, d2psi, V, I, dt,x, m, hbar):
    psi_x = psi(x)
    d2psi_x = np.gradient(np.gradient(psi_x))
    k1 = -(1j * dt / hbar) * (-(hbar ** 2) / (2 * m) * d2psi_x + (V(x) + I(x)) * psi_x)
    psi2_x = psi_x + k1/2
    d2psi2_x = d2psi_x + np.gradient(np.gradient(k1/2))
    k2 = -(1j * dt / hbar) * (-(hbar ** 2) / (2 * m) * (d2psi2_x) + (V(x) + I(x)) * psi2_x)
    psi3_x = psi_x + k2/2
    d2psi3_x = d2psi_x + np.gradient(np.gradient(k2/2))
    k3 = -(1j * dt / hbar) * (-(hbar ** 2) / (2 * m) * (d2psi3_x) + (V(x) + I(x)) * psi3_x)
    psi4_x = psi_x + k3
    d2psi4_x = d2psi_x + np.gradient(np.gradient(k3))
    k4 = -(1j * dt / hbar) * (-(hbar ** 2) / (2 * m) * (d2psi4_x) + (V(x) + I(x)) * psi4_x)
    return psi_x + (k1 +2*k2 + 2*k3 + k4)/6


"""
Given old and new samples of the wavefunction at the same set of sample points,
remove samples where the relative change is bigger than some threshold. This is
designed to get rid of noise.
"""
def clean_samples(y, y_prev, threshold=0.1):
    pass


"""

Obtain a fitting method based on SVM.
"""
def get_svm_fitting_method(epsilon=1e-4):
    def fitting_method(x, y):
        return svm_function_approximation.svm_fit(x, y.ravel(), epsilon=epsilon)
    return fitting_method


"""
Obtain a fitting method based on neural networks.
"""
def get_neural_fitting_method(bosonic, U, n_particles, dim_physical, n_layers, layer_size, epochs, batch_size, reg):
    def fitting_method(x, y, analysis_data, iteration, load_weights):
        # return neural_function_approximation_2d.neural_fit(x, y.ravel(), analysis_data, n_layers=n_layers, layer_size=layer_size, epochs=epochs, learning_batch_size=batch_size, reg=reg)
        return nn_fit.neural_fit(x, y.ravel(), analysis_data, iteration, load_weights, bosonic, U, n_particles, dim_physical, n_layers=n_layers, layer_size=layer_size, epochs=epochs, learning_batch_size=batch_size, reg=reg)
    return fitting_method


"""
Perform time propagation.
"""
def propagate_in_time(eval_psi0, eval_V, eval_I, load_weights, U, n_particles, dim, nsamples, t, m, hbar, xmax, n_x, step_size, x0, decorrelation_steps, uniform_ratio, fitting_method, normalize, eta, calculate_energy, bosonic):
    def eval_d2psi0(x):
        psi_ph = np.zeros(x.shape, dtype=complex)
        psi_mh = np.zeros(x.shape,  dtype=complex)
        for i in range(0, dim*n_particles):
            x_ph = x.copy()
            x_mh = x.copy()
            x_ph[:,i] += eta
            x_mh[:,i] -= eta
            psi_ph[:,i] = eval_psi(x_ph)
            psi_mh[:,i] = eval_psi(x_mh)
        return (psi_ph.sum(axis=-1) + psi_mh.sum(axis=-1) - 2 * (dim*n_particles) * eval_psi(x)) / (eta**2)  
    def P0(x):
        return np.real(np.conj(eval_psi0(x)) * eval_psi0(x))
    x0_arr = np.zeros(n_particles*dim)
    step = np.full(n_particles*dim, xmax)
    dt = t[1] - t[0]
    psi_t = np.zeros((nsamples, t.shape[0]), dtype=complex)
    x_t = np.zeros((nsamples, n_particles*dim, t.shape[0]))
    energies_t = np.zeros(t.shape[0])
    mse_t = np.zeros(t.shape[0])

    eval_psi = eval_psi0
    eval_d2psi = eval_d2psi0
    P = P0
    start = datetime.now()

    for i in range(t.shape[0]):
    #for i in range(20):
        print("i=", i, "time=", datetime.now()-start)
        samples = sample_distribution_Nd.sample_mixed(P, x0_arr, step, xmax, nsamples, decorrelation_steps, uniform_ratio)
        print("samples shape: ", samples.shape)
        x_t[:, :, i] = samples
        psi_t[:, i] = eval_psi(samples)
        X_symm = np.zeros((nsamples*factorial(n_particles), n_particles, dim), dtype=np.float64)
        new_psi_t_symm = np.zeros((nsamples*factorial(n_particles)), dtype=complex)

        if (calculate_energy):
            def Hpsi(x):
                return -(hbar ** 2) / (2 * m) * eval_d2psi(x) + eval_V(x) * eval_psi(x)  + eval_I(x) * eval_psi(x)
            energies_t[i], mse_t[i] = sample_wavefunction_Nd.vec_sample_energy(eval_psi, eval_d2psi, Hpsi, x0_arr, step, nsamples, decorrelation_steps, xmax)
            print("Energy: ", energies_t[i], "Mse: ", mse_t[i])

        new_psi_t = propagate_samples(eval_psi, eval_d2psi, eval_V, eval_I, dt, samples, m, hbar, n_particles, dim)
        
        X = samples.reshape(nsamples, n_particles, dim)
        l = list(permutations(range(0, n_particles)))
        for k in range(len(l)):
            p = l[k] 
            for j in range(len(p)):
                X_symm[(k*nsamples):((k+1)*nsamples),j] = X[:,p[j]]
        
            if bosonic == 1:
                new_psi_t_symm[(k*nsamples):(k+1)*nsamples] = new_psi_t
            else:
                new_psi_t_symm[(k*nsamples):(k+1)*nsamples] = ((-1)**k)*new_psi_t

        X_symm = X_symm.reshape(nsamples*factorial(n_particles), n_particles*dim)

        average_value = np.max(np.abs(new_psi_t_symm))
        if normalize:
            new_psi_t_symm = new_psi_t_symm / average_value

        loss = 1
        while loss >= 1e-3 or math.isnan(loss):
            analysis_data = {}
            fitfunc = fitting_method(X_symm, np.real(new_psi_t_symm).ravel(), analysis_data, i, load_weights)
            history = analysis_data['history']
            loss = history.history['loss'][-1]
        def fit_psi(x):
            f = fitfunc(x)[:,0]
            #return f*average_value
            return f
        def fit_d2psi(x):
            psi_ph = np.zeros(x.shape, dtype=complex)
            psi_mh = np.zeros(x.shape,  dtype=complex)
            for i in range(0, dim*n_particles):
                x_ph = x.copy()
                x_mh = x.copy()
                x_ph[:,i] += eta
                x_mh[:,i] -= eta
                psi_ph[:,i] = eval_psi(x_ph)
                psi_mh[:,i] = eval_psi(x_mh)
            return (psi_ph.sum(axis=-1) + psi_mh.sum(axis=-1) - 2 * (dim*n_particles) * eval_psi(x)) / (eta**2)  
        def fit_P(x):
            return np.abs(fit_psi(x) ** 2)

        eval_psi = fit_psi
        eval_d2psi = fit_d2psi
        P = fit_P
        
        # File output
        results = {
            'x' : x_t,
            't' : t,
            'psi_t' : psi_t,
            'energies_t' : energies_t,
            'mse_t' : mse_t,
        }
        if bosonic == 1:
            output =  "temp_neural_{}_{}d_{}N_U={}.pkl".format('bosons', dim, n_particles, U)
            pickle.dump(results, open( output, "wb" ))
        else:
            output =  "temp_neural_{}_{}d_{}N_U={}.pkl".format('fermions', dim, n_particles, U)
            pickle.dump(results, open( output, "wb" ))

    print("total_time=", datetime.now()-start)
    return x_t, psi_t, energies_t, mse_t

if __name__ == '__main__':
    import harmonic_oscillator_3d_Nparticles as ho
    import matplotlib.pyplot as plt
    U = 0.0
    nu = 0.01
    hbar = 1.0
    m = 1.0
    omega = 1.0
    tmax = 1.0
    xmax = 5.0
    offset = [2.0, -2.0]
    n_t = 100
    nsamples = 5000
    bosonic = False
    eta= 1e-2
    n_particles = 2
    dim = 2
    
    # Generate data by sampling
    def wavefunction(x):
        return ho.eigenfunction_samples(0, x, offset, hbar, m, omega, bosonic)
    def P0(x):
        return np.real(np.conj(wavefunction(x)) * wavefunction(x))
    a = xmax
    x0 = np.zeros(n_particles*dim)
    step = np.full(n_particles*dim, xmax)
    decorrelation_steps = 5
    uniform_ratio = 0.3
    samples = sample_distribution_3d_Nparticles.sample_mixed(P0, x0, step, xmax, nsamples, decorrelation_steps, uniform_ratio)
    #samples = sample_distribution_3d_Nparticles.sample_uniform(a, [int(uniform_ratio * nsamples), x0.shape[0]])
    samples = samples.T

    t = -1j * np.linspace(0, tmax, n_t)
    dt = t[1] - t[0]

    # Stochastic propagation with exact wavefunction
    ################################################
    def eval_psi(x):
        return ho.eigenfunction_samples(0, x, offset, hbar, m, omega, bosonic)
    def eval_d2psi(x, eta=1e-5):
        psi_ph = np.zeros(x.shape, dtype=complex)
        psi_mh = np.zeros(x.shape,  dtype=complex)
        for i in range(0, dim*n_particles):
             x_ph = x.copy()
             x_mh = x.copy()
             x_ph[i] += eta
             x_mh[i] -= eta
             psi_ph[i] = eval_psi(x_ph)
             psi_mh[i] = eval_psi(x_mh)
        return (psi_ph.sum(axis=0) + psi_mh.sum(axis=0) - 2 * (dim*n_particles) * eval_psi(x)) / (eta**2)  
    def eval_V(x):
        return ho.potential(x, m, omega)
    def eval_I(x):
        return ho.coulomb(x, U, nu)

    psi0 = eval_psi(samples)
    psi_t = propagate_samples(eval_psi, eval_d2psi, eval_V, eval_I, dt, samples, m, hbar, n_particles, dim)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[0], samples[2], np.real(psi0))
    ax.scatter(samples[0], samples[2], np.real(psi_t))
    plt.show()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[0].flatten(), X[1].flatten(), np.real(psi0))
    # ax.scatter(X[0].flatten(), X[1].flatten(), np.real(psi_t))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(samples_x[0], samples_x[1], np.real(samples_psi0))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(samples_x[0], samples_x[1], np.real(samples_psi_t))
    # ax.scatter(samples_x[0], samples_x[1], np.real(RK_samples_psi_t))

   # plt.show()
