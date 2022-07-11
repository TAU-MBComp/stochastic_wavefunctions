#!/usr/bin/env python
from figures import *
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colorbar import ColorbarBase
import matplotlib.font_manager
import pickle
import sys
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

# file_b = sys.argv[1]
# file_f = sys.argv[2]
# results_b = pickle.load( open( file_b, "rb" ) )
# results_f = pickle.load( open( file_f, "rb" ) )
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
# Create a figure with sane defaults for publication.
fig = create_figure(height_multiplier=1.0)
# ax1, ax2, ax3, ax4 = create_quad_split(fig,
#                        xlabel=('Epoch', 'Epoch'),
#                        ylabel=(r'$E_\mathrm{VMC}$', 'Symmetry'), palette=('viridis', 'viridis', 'viridis', 'viridis'),
#                       numcolors=(5, 5, 5, 5)
#                        )

ax1, ax2 = create_horizontal_split(fig, 2, xlabel=('Epoch', 'Epoch'), ylabel=(r'$E_{\mathrm{VMC}}/E_{\mathrm{Exact}}$', 'Asymmetry'),
                                 palette=('viridis', 'viridis'), numcolors=(5, 5))


#ax.errorbar(tau_b, E_b, fmt='.', yerr=mse_b, markersize=1.0, capsize=2.01, color='royalblue')
#ax.axhline(y = 2.0, color = 'k', linewidth=0.5, linestyle = '--', label='exact energy')
# ax.errorbar(tau_f, E_f, fmt='.', yerr=mse_f, markersize=1.0, capsize=2.0, color='teal')
ax1.plot((np.array(energy_loss1))/2., color='indigo', lw=0.5, ls=":")
#ax.set_ylim(2.5, 6.2)
#ax.set_xlim(-0.02, 1.0)

ax2.semilogy(symmetry_loss1, color='indigo', lw=0.5, ls=":")
#im = mpimg.imread('plot_intro_i=0-1.png')
#newax = fig.add_axes([0.5, 0.5, 0.2, 0.2], anchor='NE', zorder=-1)
#newax = fig.add_axes([0.25, 0.6, 0.5, 0.5])
#newax.imshow(im)
#newax.axis('off')
#imagebox = OffsetImage(im, zoom=0.1)
#ab = AnnotationBbox(imagebox, (0.2, 0.2), xybox=(120., -80.))
#ax.add_artist(ab)
ax1.plot((np.array(energy_loss2))/4.5, color='teal', lw=0.5)
ax1.legend(['2 fermions', '3 fermions'], loc=4)

ax1.axhline(y=1., color = 'k', linewidth=0.25, linestyle = '--')

# ax1.axhline(y=4.5, color = 'k', linewidth=0.25, linestyle = '--', label='exact energy')
#ax.set_ylim(2.5, 6.2)
#ax.set_xlim(-0.02, 1.0)

ax2.semilogy(symmetry_loss2, color='teal', lw=0.5)
# Clean up and save:
fig.savefig('plot.pdf')
# plt.show()
