#!/usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '')
filename = sys.argv[1]
results = pickle.load( open( filename, "rb" ) )

tau = -results['t'].imag
print(results['energies_t'].real, results['mse_t'].real)
# plt.plot(tau, results['energies_t'].real, label="two fermions")
# plt.plot(tau, np.ones(tau.shape[0]) * eval_exact_energy(), label="Exact")
# plt.xlabel(r'$\tau$')
# plt.ylabel(r'$E_0$')
# plt.savefig('energy_curve.pdf')
# plt.show()
