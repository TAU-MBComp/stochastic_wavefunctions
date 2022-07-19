#!/usr/bin/env python
from figures import *
import pickle

data1 = pickle.load(open("data1Ha2.pkl", "rb"))
N = 3000
symmetry_loss1 = data1[0][:N]
energy_loss1 = data1[1][:N]
finalE1 = data1[2]
finaldE1 = data1[3]
print(finalE1.numpy(), "+-", finaldE1.numpy())

data1 = pickle.load(open("data1Ha3.pkl", "rb"))
N = 3000
symmetry_loss2 = data1[0][:N]
energy_loss2 = data1[1][:N]
finalE2 = data1[2]
finaldE2 = data1[3]
print(finalE2.numpy(), "+-", finaldE2.numpy())
fig = create_figure(height_multiplier=1.0)
ax1, ax2 = create_horizontal_split(fig, 2, xlabel=('Epoch', 'Epoch'), ylabel=(r'$E_{\mathrm{VMC}}/E_{\mathrm{Exact}}$', 'Asymmetry'),
                                 palette=('viridis', 'viridis'), numcolors=(5, 5))
ax1.plot((np.array(energy_loss1))/2., color='indigo', lw=0.5, ls=":")
ax2.semilogy(symmetry_loss1, color='indigo', lw=0.5, ls=":")
ax1.plot((np.array(energy_loss2))/4.5, color='teal', lw=0.5)
ax1.legend(['2 fermions', '3 fermions'], loc=4)
ax1.axhline(y=1., color = 'k', linewidth=0.25, linestyle = '--')
ax2.semilogy(symmetry_loss2, color='teal', lw=0.5)
fig.savefig('plot.pdf')
