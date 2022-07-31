#!/usr/bin/env python
import numpy as np
from numpy import sqrt, pi, exp
from scipy.special import hermite
from numpy.polynomial.hermite import hermvander
from scipy.linalg import expm
from math import factorial
from itertools import permutations
from numpy import linalg as LA
from sample_distribution_Nd import sample_uniform, sample_from_distribution, vec_sample_from_distribution, sample_mixed


def perm_parity(lst):
    '''\
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    '''
    parity = 1
    for i in range(0, len(lst) - 1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i, len(lst)), key=lst.__getitem__)
            lst[i], lst[mn] = lst[mn], lst[i]
    return parity


def eigenfunction_samples(n, x, n_particles, dim_physical, offset, hbar, m,
                          omega, bosonic):

    def psi_single_particle(x, n):
        return 1.0 / sqrt((2 ** n) * factorial(n)) * ((m * omega / pi / hbar) ** 0.25) \
            * exp(-m * omega * (x ** 2) / (2 * hbar)) * hermite(n)(sqrt(m * omega / hbar) * x)

    nsamples = x.shape[0]
    n_perm = factorial(n_particles)
    psi = np.zeros(nsamples, dtype=complex)
    x = x.reshape(nsamples, n_particles, dim_physical)
    for p in permutations(range(0, n_particles)):
        l = list(p)
        parity = perm_parity(l)
        row = np.zeros((dim_physical, n_particles, nsamples), dtype=complex)
        for n_p in range(n_particles):
            x_p = x[:, n_p] + offset[p[n_p]]
            for d in range(dim_physical):
                if (bosonic):
                    row[d, n_p] = psi_single_particle(x_p[:, d], n)
                else:
                    if d == (dim_physical - 1):
                        row[d,
                            n_p] = psi_single_particle(x_p[:, d], n + p[n_p])
                    else:
                        row[d, n_p] = psi_single_particle(x_p[:, d], n)
        if bosonic == 1:
            row = np.prod(np.prod(row, axis=0), axis=0)
        else:
            row = parity * np.prod(np.prod(row, axis=0), axis=0)
        psi += row / np.sqrt(n_perm)
    return psi


def energy(n, X, U, nu, hbar, m, omega):
    H = hamiltonian(X, U, nu, hbar, m, omega)
    w, v = LA.eig(H)
    w.sort()
    return hbar * omega * np.real(w[n])


def laplacian_2D(N):
    I = np.eye(N)
    L1 = -3.0 * np.eye(N)
    L2 = -4.0 * np.eye(N)
    for i in range(N - 1):
        L1[i, i + 1] = L1[i + 1, i] = 1
        L2[i, i + 1] = L2[i + 1, i] = 1
    L1[0, 0] = L1[-1, -1] = -2
    L2[0, 0] = L2[-1, -1] = -3

    L = np.zeros((N * N, N * N))
    for n in range(0, N):
        start = n * N
        end = (n + 1) * N
        L[n * N:(n + 1) * N, n * N:(n + 1) * N] = L2
        # fill in off diagonal elements
        if n < (N - 1):
            L[n * N:(n + 1) * N, (n + 1) * N:(n + 2) * N] = I
            L[(n + 1) * N:(n + 2) * N, n * N:(n + 1) * N] = I

    L[0:N, 0:N] = L1
    L[(N - 1) * N:N * N, (N - 1) * N:N * N] = L1
    return L


def potential(X, m, omega):
    return 0.5 * m * (omega**2) * np.sum(X * X, axis=-1)


def coulomb(X, U, nu):
    r = np.sqrt(np.sum(X * X, axis=-1))
    return U * 1 / (r + nu)


def coulomb_e_e(X, Ze, nu):
    r12 = (X[:, 1] - X[:, 0])
    r = np.sqrt(np.sum(r12 * r12, axis=-1))
    return Ze / (r + nu)


def coulomb_e_n(X, Z, nu):
    r = np.sqrt(np.sum(X * X, axis=-1))
    return np.sum((-Z / (r + nu)), axis=-1)


def H2_plus_I(X, R, Z, nu):
    R = np.full(np.shape(X), 2)
    R[0] = 0
    R[1] = 0
    r_p = np.sqrt(np.sum((X - R / 2)**2))
    r_m = np.sqrt(np.sum((X + R / 2)**2))
    return Z / (r_p + nu) + Z / (r_m + nu)


def hamiltonian(X, U, nu, hbar, m, omega):
    dx = X[0][0, 1] - X[0][0, 0]
    n = X[0].shape[1]
    T = -hbar**2 / (2 * m) / (dx**2) * laplacian_2D(n)
    V = np.diag(potential(X, m, omega).flatten())
    I = np.diag(coulomb(X, U, nu).flatten())
    return T + V + I


def propagator(X, U, nu, hbar, m, omega, dt):
    H = hamiltonian(X, U, nu, hbar, m, omega)
    return expm(-1j * dt / hbar * H)


def propagate_in_time(psi0,
                      X,
                      t,
                      U,
                      nu,
                      hbar,
                      m,
                      omega,
                      method="exact",
                      normalize=False):
    # Setup
    n_x, n_y, n_t = X[0].shape[1], X[1].shape[0], t.shape[0]
    dx = X[0][0, 1] - X[0][0, 0]
    dy = X[1][1, 0] - X[1][0, 0]
    dt = t[1] - t[0]

    psi_t = np.zeros((n_x * n_y, n_t), dtype=complex)

    # Initial condition
    psi_t[:, 0] = psi0

    # Perform propagation
    if method == "exact":
        U_dt = propagator(X, U, nu, hbar, m, omega, dt)
        for i in range(1, n_t):
            psi_t[:, i] = U_dt @ psi_t[:, i - 1]
            average_value = np.sqrt(np.sum(
                np.abs(psi_t[:, i])**2 * np.abs(dx)))
            if normalize:
                psi_t[:, i] = psi_t[:, i] / average_value
    if method == "euler":
        H = hamiltonian(X, U, nu, hbar, m, omega)
        for i in range(1, n_t):
            psi_t[:,
                  i] = psi_t[:,
                             i - 1] - (1j * dt / hbar) * (H @ psi_t[:, i - 1])
            average_value = np.sqrt(np.sum(
                np.abs(psi_t[:, i])**2 * np.abs(dx)))
            if normalize:
                psi_t[:, i] = psi_t[:, i] / average_value

    return psi_t


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    U = 0.0
    nu = 0.001
    hbar = 1.0
    m = 1.0
    omega = 1.0
    tmax = 1.0
    xmax = ymax = zmax = 5.0
    offset = [0.0, -0.0]
    n_t = 10
    n_x = n_y = n_z = 100
    bosonic = 0
    n_particles = 2
    dim_physical = 1

    x = np.linspace(-xmax, xmax, n_x)
    t = -1j * np.linspace(0, tmax, n_t)

    def wavefunction(x):
        return eigenfunction_samples(0, x, n_particles, dim_physical, offset,
                                     hbar, m, omega, bosonic)

    def P0(x):
        return np.real(np.conj(wavefunction(x)) * wavefunction(x))

    nsamples = 10000
    a = xmax
    uniform_ratio = 0.2
    step = np.full((dim_physical * n_particles), xmax, dtype=float)
    x0 = np.zeros((dim_physical * n_particles), dtype=float)
    decorrelation_steps = 5
    samples = vec_sample_from_distribution(P0, x0, nsamples,
                                           decorrelation_steps, xmax)
    psi0 = eigenfunction_samples(0,
                                 samples,
                                 n_particles,
                                 dim_physical,
                                 offset,
                                 hbar,
                                 m,
                                 omega,
                                 bosonic=bosonic)
