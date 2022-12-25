#!/usr/bin/env python
import pickle
import matplotlib.pyplot as plt
import numpy as np

results = pickle.load(open("results.pkl", "rb"))
tau = -results['t'].imag
plt.plot(tau,
         results['energies_t'].real,
         label="Stochastic wavefunction result")
plt.plot(tau,
         3.0 * np.ones_like(tau),
         '--',
         label="Exact (2 noninteracting fermions in 2D)")
plt.xlabel(r'$\tau$')
plt.ylabel(r'$E_0$')
plt.legend()
plt.show()
