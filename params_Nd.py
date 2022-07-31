# Physical parameters
U = 0.0
nu = 0.001
hbar = 1.0
m = 1.0
omega = 1.0
tmax = 2.0
xmax = 5.0
offset = [1.0, -1.0]
normalize = True
bosonic = 1
calculate_energy = True
n_particles = 2
dim_physical = 2
load_weights = 0

# Numerical parameters
iteration = 0
n_t = 100
n_x = 100
nsamples = 5000
decorrelation_steps = 10
uniform_ratio = 0.2
step_size = xmax
x0 = 0.0
eta = 1e-2  # Discretization for finite difference
perm_subset = 2

# Neural network parameters
n_layers = 2
layer_size = 4*128
epochs = 500  
batch_size = 128  
reg = 1e-8

# Set up initial condition and potential evaluation:
import harmonic_oscillator_Nd as ho
import numpy as np


def eval_psi0(x):
    return ho.eigenfunction_samples(0, x, n_particles, dim_physical, offset,
                                    hbar, m, omega, 1)


def eval_V(x):
    return ho.potential(x, m, omega)


def eval_I(x):
    return ho.coulomb(x, U, nu)


def eval_exact_energy(x):
    return ho.energy(0, x, U, nu, hbar, omega)
