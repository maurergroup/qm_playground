############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.phasespace_basis import *   #
from qmp.potential.pot_tools import *      #
from qmp.tools.visualizations import *     #
from qmp.integrator.dyn_tools import EOM_morse_analyt
from qmp.tools.utilities import kB
############################################


### SIMULATION CELL ### 
cell = [[0., 40.0]]

### POTENTIAL ###
#a = 0.5
#D = 5.
#steps = 2000
#t, dt = np.linspace(0.,100.,steps, retstep=True)
#m = 2.
#pos= 10.
#T = 2000.
#r_analyt = EOM_morse_analyt(a,D,m,t,pos, Temp=T)

f = create_potential(cell,
                     name='double_well',
                     double_well_barrier=.008,
                     double_well_asymmetry=0.,
                     double_well_width=3.,
                     )

pot = Potential( cell, f=f )


### INITIALIZE MODEL ### 
traj1d = Model(
         ndim=1,
         mode='traj',
         basis='phasespace',
         integrator='vel_verlet',
        )

### SET POTENTIAL ###
traj1d.set_potential(pot)

### SET INITIAL VALUES ###
rs = [[17.],[17.],[18.],[18.],[16.],[16.],[17.5],[17.5],[16.5],[16.5]]
vs = [[0.002797],[-0.002797],[0.002270],[-0.002270],[0.001610],[-0.001610],[0.002649],[-0.002649],[0.002588],[-0.002588]]
masses = [1850.,]*10

b = phasespace(rs, vs, masses)
traj1d.set_basis(b)

print traj1d


### DYNAMICS PARAMETERS ###
dt =  8.1
steps = 1E3


### EVOLVE SYSTEM ###
traj1d.run(steps,dt)

## gather information
r_t = traj1d.data.traj.r_t.flatten()
v_t = traj1d.data.traj.v_t.flatten()
E_t = traj1d.data.traj.E_t.flatten()
E_kin = traj1d.data.traj.E_kin_t.flatten()
E_pot = traj1d.data.traj.E_pot_t.flatten()
#print min(r_analyt)
#print min(r_t)
#print max(r_analyt)
#print max(r_t)
#print np.mean(r_analyt)
print np.mean(r_t)
print np.mean(E_t)

h = np.histogram(r_t,bins=np.arange(traj1d.data.cell[0][0], traj1d.data.cell[0][1], 0.1), density=True)
rbins = h[1]
prop_dist = h[0]

V_x = traj1d.pot(np.linspace(0.,cell[0][1],600))

### VISUALIZATION ###

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.gca()
#n, bins, bla = plt.hist(r_t.flatten(),200.,normed=1)
#ax.clear()
#ax.plot(np.linspace(min(bins)+(bins[1]-bins[0])/2.,max(bins)-(bins[1]-bins[0])/2.,len(n)), n)
ax.plot(np.linspace(min(rbins)+(rbins[1]-rbins[0])/2.,max(rbins)-(rbins[1]-rbins[0])/2.,len(prop_dist)), prop_dist)
plt.savefig('prop_dist_classical_'+str(int(steps))+'steps.pdf')
#raise SystemExit


import matplotlib.animation as animation

fig = plt.figure()
ax = plt.subplot2grid((4,1), (1,0), rowspan=3, colspan=2)
ax1 = plt.subplot2grid((4,1), (0,0))

wave_plot1, = ax.plot(r_t[0], traj1d.pot(r_t[0]), ls='',marker='o',mfc='r',mec='r',ms=8.5)
#wave_plot2, = ax.plot(r_analyt[0], traj1d.pot(r_analyt[0]), ls='',marker='d',mfc='k',mec='k',ms=8.5)


def _init_():
    ax.plot(np.linspace(0.,cell[0][1],600), V_x, ls='-', c='b', label='$V(x)$', zorder=3)
    ax1.plot(np.linspace(0., len(E_t)*dt, len(E_t)), E_t, c='b', ls='-', label='$E_1(t)$')
    if E_kin is not None:
        ax1.plot(np.linspace(0., len(E_t)*dt, len(E_t)), E_kin, c='g', label='$E^{kin}_1(t)$ $[a.u.]$')

    if E_pot is not None:
        ax1.plot(np.linspace(0., len(E_t)*dt, len(E_t)), E_pot, c='r', label='$E^{pot}_1(t)$ $[a.u.]$')

    ax1.legend(loc='best')
    ax1.set_xlim([0., len(E_t)*dt])
    ax1.set_xlabel('$t$ $[a.u.]$')
    ax1.xaxis.tick_top()
    ax1.set_xticks(ax1.get_xticks()[1:])
    ax1.xaxis.set_label_position('top')
    if (E_kin is None) and (E_pot is None):
        ax1.set_ylim(min(E_t.flatten())-0.01, max(E_t.flatten())+0.01)

    ax.set_xlabel('$x$ $[a.u.]$')
    ax.set_ylim(min(V_x)-0.1*max(V_x), max(V_x)+0.2)
    ax.legend(loc=2, numpoints=1)
    return wave_plot1, #wave_plot2, 

def animate(i):
    wave_plot1.set_data(r_t[i], traj1d.pot(r_t[i]))  # update data
    wave_plot1.set_label('$r_1(t$ $=$ ${0:>4.1f}$ $au)$' .format(i*dt))
  #  wave_plot2.set_data(r_analyt[i], traj1d.pot(r_analyt[i]))  # update data
    ax.legend(loc=2,numpoints=1)

    return wave_plot1, #wave_plot2, 

ani = animation.FuncAnimation(fig, animate, np.arange(0, len(r_t)), init_func=_init_, \
                              interval=50, blit=False)

plt.show()


