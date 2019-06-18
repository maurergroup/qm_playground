############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.phasespace_basis import *   #
from qmp.potential.pot_tools import *      #
from qmp.tools.visualizations import *     #
############################################


### SIMULATION CELL ###
cell = [[0., 40.0]]

### POTENTIAL ###
pot = Potential( cell, f=create_potential(cell, name='mexican_hat') )


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
rs = [[17.],[18.]]
vs = [[.6],[.2]]
masses = [1., 2.]

#rs = [[13.],[33.],[5.],[21.],[18.],[9.],[27.]]
#vs = [[3.],[1.],[-.5],[1.],[5.],[.5],[-2.]]
#masses = [4., 2., 1., 2., 3., 1., 1.]

b = phasespace(rs, vs, masses)
traj1d.set_basis(b)

print(traj1d)


### DYNAMICS PARAMETERS ###
dt =  .5
steps = 1000


### EVOLVE SYSTEM ###
traj1d.run(steps,dt)

## gather information
r_t = traj1d.data.traj.r_t
v_t = traj1d.data.traj.v_t
E_t = traj1d.data.traj.E_t
E_kin = traj1d.data.traj.E_kin_t
E_pot = traj1d.data.traj.E_pot_t


V_x = traj1d.pot(np.linspace(0.,40.,600))

### VISUALIZATION ###

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = plt.subplot2grid((4,1), (1,0), rowspan=3, colspan=2)
ax1 = plt.subplot2grid((4,1), (0,0))

wave_plot1, = ax.plot(r_t[0,0], traj1d.pot(r_t[0,0]), ls='',marker='o',mfc='r',mec='r',ms=8.5)
wave_plot2, = ax.plot(r_t[0,1], traj1d.pot(r_t[0,1]), ls='',marker='o',mfc='k',mec='k',ms=8.5)
#wave_plot3, = ax.plot(r_t[0,2], traj1d.pot(r_t[0,2]), label='$r_3(t)$',ls='',marker='o',mfc='b',mec='b',ms=8.5)
#wave_plot4, = ax.plot(r_t[0,3], traj1d.pot(r_t[0,3]), label='$r_4(t)$',ls='',marker='o',mfc='g',mec='g',ms=8.5)
#wave_plot5, = ax.plot(r_t[0,4], traj1d.pot(r_t[0,4]), label='$r_5(t)$',ls='',marker='o',mfc='y',mec='y',ms=8.5)
#wave_plot6, = ax.plot(r_t[0,5], traj1d.pot(r_t[0,5]), label='$r_6(t)$',ls='',marker='o',mfc='grey',mec='grey',ms=8.5)
#wave_plot7, = ax.plot(r_t[0,6], traj1d.pot(r_t[0,6]), label='$r_7(t)$',ls='',marker='o',mfc='orange',mec='orange',ms=8.5)


def _init_():
    ax.plot(np.linspace(0.,40.,600), V_x, ls='-', c='b', label='$V(x)$', zorder=3)
    ax1.plot(np.linspace(0., len(E_t[0])*dt, len(E_t[0])), E_t[0], c='b', ls='-', label='$E_1(t)$')
    ax1.plot(np.linspace(0., len(E_t[1])*dt, len(E_t[1])), E_t[1], c='cyan', ls='-', label='$E_2(t)$')
    if E_kin is not None:
        ax1.plot(np.linspace(0., len(E_t[0])*dt, len(E_t[0])), E_kin[0], c='g', label='$E^{kin}_1(t)$ $[a.u.]$')
        ax1.plot(np.linspace(0., len(E_t[0])*dt, len(E_t[0])), E_kin[1], c='grey', label='$E^{kin}_2(t)$ $[a.u.]$')

    if E_pot is not None:
        ax1.plot(np.linspace(0., len(E_t[0])*dt, len(E_t[0])), E_pot[0], c='r', label='$E^{pot}_1(t)$ $[a.u.]$')
        ax1.plot(np.linspace(0., len(E_t[0])*dt, len(E_t[0])), E_pot[1], c='orange', label='$E^{pot}_2(t)$ $[a.u.]$')

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
    return wave_plot1, wave_plot2,# wave_plot3, wave_plot4, wave_plot5, wave_plot6, wave_plot7,

def animate(i):
    wave_plot1.set_ydata(traj1d.pot(r_t[0,i]))  # update data
    wave_plot1.set_xdata(r_t[0,i])
    wave_plot1.set_label('$r_1(t$ $=$ ${0:>4.1f}$ $au)$' .format(i*dt))
    wave_plot2.set_ydata(traj1d.pot(r_t[1,i]))  # update data
    wave_plot2.set_xdata(r_t[1,i])
    wave_plot2.set_label('$r_2(t$ $=$ ${0:>4.1f}$ $au)$' .format(i*dt))
    ax.legend(loc=1,numpoints=1)
    #wave_plot3.set_ydata(traj1d.pot(r_t[i,2]))  # update data
    #wave_plot3.set_xdata(r_t[i,2])
    #wave_plot4.set_ydata(traj1d.pot(r_t[i,3]))  # update data
    #wave_plot4.set_xdata(r_t[i,3])
    #wave_plot5.set_ydata(traj1d.pot(r_t[i,4]))  # update data
    #wave_plot5.set_xdata(r_t[i,4])
    #wave_plot6.set_ydata(traj1d.pot(r_t[i,5]))  # update data
    #wave_plot6.set_xdata(r_t[i,5])
    #wave_plot7.set_ydata(traj1d.pot(r_t[i,6]))  # update data
    #wave_plot7.set_xdata(r_t[i,6])
    return wave_plot1, wave_plot2,# wave_plot3, wave_plot4, wave_plot5, wave_plot6, wave_plot7,

ani = animation.FuncAnimation(fig, animate, np.arange(0, len(r_t[0])), init_func=_init_, \
                              interval=50, blit=False)

plt.show()
