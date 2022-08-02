### Physical parameters ###
U = 0.0  # Coulomb interaction
nu = 0.001  # offset of Coulomb interaction to avoid divergence
hbar = 1.0
m = 1.0
omega = 1.0
tmax = 2.0  # maximal time of imaginary time propagation
xmax = 5.0  # cutoff for generating samples in space
offset = [1.0, -1.0]  # offset of initial non-interacting wavefunction
normalize = True  # normalize maximal value of wavefunction to 1
bosonic = False  # choose between bosonic or fermionic symmetry
calculate_energy = True  # calculate energy at every step of the imaginary time propagation
n_particles = 2  # number of particles
dim_physical = 2  # dimensions in space

### Numerical parameters ###
iteration = 0  # initial step in the imaginary time propagation; the default is to start with 0
n_t = 100  # imaginary time propagation steps
n_x = 100  # size of grid in space
nsamples = 5000  # number of samples that represent the wavefunction stochastically
decorrelation_steps = 10  # decorrelation of samples in Monte Carlo
uniform_ratio = 0.2  # ratio of uniformly distributes samples
step_size = xmax
x0 = 0.0
eta = 1e-2  # discretization for finite difference
perm_subset = 2  # number of permutations to be considered at every iteration step; maximum value is n_particles!
load_weights = 0  # load neural network weights from previous fit

### Neural network parameters ###
n_layers = 2
layer_size = 4 * 128
epochs = 100
batch_size = 128
reg = 1e-8

### Set up initial condition and potential evaluation: ###
import harmonic_oscillator_Nd as ho
import numpy as np
"""
Define functions to pass on to the propagation algorithm. 
"""


def eval_psi0(x):
    return ho.eigenfunction_samples(0, x, n_particles, dim_physical, offset,
                                    hbar, m, omega, bosonic)


def eval_V(x):
    return ho.potential(x, m, omega)


def eval_I(x):
    return ho.coulomb(x, U, nu)


def eval_exact_energy(x):
    return ho.energy(0, x, U, nu, hbar, omega)
