import numpy as np, matplotlib.pyplot as plt, harmonic_oscillator_Nd as ho
from mpl_toolkits.mplot3d import Axes3D


def vec_sample_energy(psi,
                      d2psi,
                      Hpsi,
                      x0,
                      step,
                      nsamples,
                      decorrelation_steps,
                      xmax=float('inf')):

    def P(x):
        return np.real(np.conj(psi(x)) * psi(x))

    vec_batch = 100
    nsamples = nsamples
    decorrelation_steps = decorrelation_steps
    x = np.random.rand(vec_batch, x0.shape[0])
    step = np.full(x.shape, xmax)
    Px = P(x)
    samples_obtained = 0
    sampled_energy = 0.0
    energy_n = []
    sampled_norm = 0.0
    steps = 0
    while samples_obtained < nsamples:
        new_x = x + step * (np.random.rand(step.shape[0], step.shape[1]) - 0.5)
        Pnew_x = P(new_x)
        a = np.random.rand()
        condition = np.vstack(
            (Pnew_x / Px >= a, np.all(np.abs(new_x) < xmax, axis=1)))
        condition = np.all(condition, axis=0)
        x = np.where(
            np.full((x.shape[0], x.shape[1]),
                    np.repeat(condition,
                              x0.shape[0]).reshape(x.shape[0], x.shape[1])),
            new_x, x)
        Px = np.where(condition, Pnew_x, Px)
        if steps % decorrelation_steps == 0:
            if samples_obtained == 0:
                E = np.conj(psi(x)) * Hpsi(x) / Px
                energy_n = E.real
                sampled_energy += np.sum(E.real)
            else:
                E = np.conj(psi(x)) * Hpsi(x) / Px
                np.append(energy_n, E.real)
                sampled_energy += np.sum(E.real)
            sampled_norm += 1.0 * vec_batch
            samples_obtained += 1 * vec_batch
        steps += 1

    av_sampled_energy = sampled_energy / sampled_norm
    mse = np.std(energy_n) / np.sqrt(sampled_norm)
    return (av_sampled_energy, mse)


def sample_energy(psi,
                  d2psi,
                  Hpsi,
                  x0,
                  step,
                  nsamples,
                  decorrelation_steps,
                  xmax=float('inf')):

    def P(x):
        return np.real(np.conj(psi(x)) * psi(x))

    x = x0
    Px = P(x)
    samples_obtained = 0
    sampled_energy = 0.0
    energy_n = []
    sampled_norm = 0.0
    steps = 0
    while samples_obtained < nsamples:
        new_x = x + 2 * step * (np.random.rand(step.shape[0]) - 0.5)
        Pnew_x = P(new_x)
        if np.random.rand() <= Pnew_x / Px and (np.abs(new_x) < xmax).all():
            x = new_x
            Px = Pnew_x
        if steps % decorrelation_steps == 0:
            E = np.conj(psi(x)) * Hpsi(x) / Px
            energy_n.append(E.real)
            sampled_energy += E.real
            sampled_norm += 1.0
            samples_obtained += 1
        steps += 1

    av_sampled_energy = sampled_energy / sampled_norm
    var = [a - av_sampled_energy for a in energy_n]
    var = np.sum([a**2 for a in var]) / sampled_norm**2
    return (av_sampled_energy, np.sqrt(var / nsamples))


if __name__ == '__main__':
    U = 2.0
    nu = 0.0001
    xmax = 5.0
    step = np.array([xmax, xmax, xmax])
    x0 = step
    nsamples = 5000
    decorrelation_steps = 5
    hbar = 1.0
    m = 1.0
    omega = 1.0
    n = 0
    eta = 0.001
    offset = [0.0, 0.0, 0.0]
    n_particles = 3
    bosonic = True

    def psi(x):
        return np.exp(-U * np.sqrt(np.sum(x**2, axis=0)))

    def P(x):
        return np.real(np.conj(psi(x)) * psi(x))

    def d2psi(x):
        psi_ph = np.zeros(x.shape, dtype=complex)
        psi_mh = np.zeros(x.shape, dtype=complex)
        for i in range(0, n_particles):
            x_ph = x.copy()
            x_mh = x.copy()
            x_ph[i] += eta
            x_mh[i] -= eta
            psi_ph[i] = psi(x_ph)
            psi_mh[i] = psi(x_mh)

        return (psi_ph.sum(axis=0) + psi_mh.sum(axis=0) -
                2 * n_particles * psi(x)) / eta**2

    def V(x):
        return ho.potential(x, m, omega)

    def I(x):
        return ho.coulomb(x, U, nu)

    def Hpsi(x):
        return -hbar**2 / (2 * m) * d2psi(x) + I(x) * psi(x)

    E, v = sample_energy(psi,
                         d2psi,
                         Hpsi,
                         x0,
                         step,
                         nsamples,
                         decorrelation_steps,
                         xmax=float('inf'))
    print('Monte Carlo energy:', E.real, v.real)
    n_x = 25
    x = np.linspace(-xmax, xmax, n_x)
    X, Y, Z = np.meshgrid(x, x, x)
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
    x_grid = np.array([X, Y, Z])
    psi_grid = psi(x_grid)
    d2_psi_grid = d2psi(x_grid)
    psi_grid = psi_grid.reshape(n_x, n_x, n_x)
    d2_psi_grid = d2_psi_grid.reshape(n_x, n_x, n_x)
