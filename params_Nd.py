# Physical parameters
U = 0.0
nu = 0.001
hbar = 1.0
m = 1.0
omega = 1.0
tmax = 2.0
xmax = 5.0
offset = [1.0, -1.0]
#offset = [0.0, -0.0]
normalize = True
bosonic = 1
calculate_energy = True 
n_particles = 2
dim_physical = 2
load_weights = 0 

# Numerical parameters
n_t = 100 #100
n_x = 100
nsamples = 10000 #10000
decorrelation_steps = 5
uniform_ratio = 0.2
step_size = xmax
x0 = 0.0
eta = 1e-2 # Discretization for finite difference

# SVM parameters
# epsilon = 1e-3

# Neural network parameters
n_layers = 3
layer_size = 128
epochs = 100 # 100
batch_size = 128 # 400
reg = 1e-8

# Set up initial condition and potential evaluation:
import harmonic_oscillator_Nd as ho
import numpy as np
def eval_psi0_sample(x):
    return ho.eigenfunction_samples(0, x, n_particles, dim_physical, offset, hbar, m, omega, 1)
def eval_V(x):
    return ho.potential(x, m, omega)
def eval_I(x):
    return ho.coulomb(x, U, nu)
def eval_exact_energy(x):
    return ho.energy(0, x, U, nu, hbar, omega)
