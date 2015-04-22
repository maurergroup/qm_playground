############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.rpmdbasis import *          #
from qmp.potential import Potential        #
from qmp.pot_tools import *                #
############################################

import scipy.linalg as la

### SIMULATION CELL ### 
cell = [[0., 40.0]]

### POTENTIAL ### 
pot = Potential( cell, f=create_potential(cell, name='harmonic') )


### INITIALIZE MODEL ### 
rpmd1d = Model(
         ndim=1,
         mode='rpmd',
         basis='rpmd_basis',
         integrator='RPMD_VelocityVerlet',
        )

### SET POTENTIAL ###
rpmd1d.set_potential(pot)

### SET INITIAL VALUES ###
rs = [[20.]]#,[18.]]
vs = [[0.]]#,[.2]]
masses = [2.]#, 2.]
n_beads = 5
Temp = [200.]#, 250.]

b = bead_basis(rs, vs, masses, n_beads, T=Temp)
rpmd1d.set_basis(b)

print rpmd1d


### DYNAMICS PARAMETERS ###
dt =  .2
steps = 1E6

### THERMOSTAT ###
thermostat = {'name' : 'Andersen',
              'cfreq' : 1E-3,
              'T_set' : 400.,
             }


### EVOLVE SYSTEM ###
rpmd1d.run(steps,dt, thermostat=thermostat)
print 'INTEGRATED'

## gather information
r_t = rpmd1d.data.rpmd.r_t
rb_t = rpmd1d.data.rpmd.rb_t
v_t = rpmd1d.data.rpmd.v_t
vb_t = rpmd1d.data.rpmd.vb_t
E_t = rpmd1d.data.rpmd.E_t
E_kin = rpmd1d.data.rpmd.E_kin_t
E_pot = rpmd1d.data.rpmd.E_pot_t

### VISUALIZATION ###
V_x = rpmd1d.pot(np.linspace(0.,40.,600))

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = plt.subplot2grid((4,1), (1,0), rowspan=3, colspan=2)
ax1 = plt.subplot2grid((4,1), (0,0))

wave_plot1, = ax.plot(r_t[0], rpmd1d.pot(r_t[0]), ls='',marker='o',mfc='r',mec='r',ms=8.5)
#wave_plot2, = ax.plot(r_t[1], rpmd1d.pot(r_t[1]), label='$r_2(t)$',ls='',marker='o',mfc='k',mec='k',ms=8.5)


def _init_():
    ax.plot(np.linspace(0.,40.,600), V_x, c='b', ls=':', label='$V(x)$', zorder=3)
    ax1.plot(np.linspace(0., len(E_t[0])*dt, len(E_t[0])), E_t[0], c='b', ls='-', label='$E_1(t)$')
    #ax1.plot(np.linspace(0., len(E_t[1])*dt, len(E_t[1])), E_t[1], c='r', ls='-', label='$E_2(t)$')
    if E_kin is not None:
        ax1.plot(np.linspace(0., len(E_t[0])*dt, len(E_t[0])), E_kin[0], c='g', label='$E^{kin}_1(t)$ $[a.u.]$')
    #    ax1.plot(np.linspace(0., len(E_t[1])*dt, len(E_t[1])), E_kin[1], c='k', label='$E^{kin}_2(t)$ $[a.u.]$')

    if E_pot is not None:
        ax1.plot(np.linspace(0., len(E_t[0])*dt, len(E_t[0])), E_pot[0], c='cyan', label='$E^{pot}_1(t)$ $[a.u.]$')
    #    ax1.plot(np.linspace(0., len(E_t[1])*dt, len(E_t[1])), E_pot[1], c='orange', label='$E^{pot}_2(t)$ $[a.u.]$')

    ax1.legend(loc='best')
    ax1.set_xlim([0., len(E_t[0])*dt])
    ax1.set_xlabel('$t$ $[a.u.]$')
    ax1.xaxis.tick_top()
    ax1.set_xticks(ax1.get_xticks()[1:])
    ax1.xaxis.set_label_position('top')
    if (E_kin is None) and (E_pot is None):
        ax1.set_ylim(min(E_t.flatten())-0.01, max(E_t.flatten())+0.01)

    ax.set_xlabel('$x$ $[a.u.]$')
    ax.set_ylim(min(V_x)-0.1*max(V_x), max(V_x)+0.2)
    ax.legend(loc=2)
    return wave_plot1, #wave_plot2,

def animate(i):
    wave_plot1.set_ydata(rpmd1d.pot(r_t[0,i]))  # update data
    wave_plot1.set_xdata(r_t[0,i])
    wave_plot1.set_label('$r_1(t$ $=$ ${0:>4.1f}$ $au)$' .format(i*dt))
    #wave_plot2.set_ydata(rpmd1d.pot(r_t[1,i]))  # update data
    #wave_plot2.set_xdata(r_t[1,i])
    ax.legend(loc=2,numpoints=1)

    return wave_plot1, #wave_plot2,

ani = animation.FuncAnimation(fig, animate, np.arange(0, len(r_t[0])), init_func=_init_, \
                              interval=50, blit=False)

plt.show()


