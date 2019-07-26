import numpy as np
import sys
from qmp import Model
from qmp.potential.pot_tools import *
from qmp.potential import Potential
from qmp.tools.visualizations import *
from qmp.integrator.trajintegrators import VelocityVerlet
from qmp.systems.phasespace import PhaseSpace


# SIMULATION CELL
cell = [[0., 40.0]]

# DYNAMICS PARAMETERS
dt = 8.1      # time step
steps = 1E4    # number of steps
# position(s)
rs = [[17.]]
# velocity(ies)
vs = [[0.002797]]
# mass(es)
masses = [1850.]

# POTENTIAL
f = create_potential(cell,
                     name='double_well',
                     double_well_barrier=.008,
                     double_well_asymmetry=0.,
                     double_well_width=3.,
                     )

pot = Potential(cell, f=f)
integrator = VelocityVerlet(dt)
system = PhaseSpace(rs, vs, masses)


# INITIALIZE MODEL
traj1d = Model(
         system=system,
         potential=pot,
         integrator=integrator,
         mode='traj',
        )


# PRINT INFORMATION
print(traj1d)

# EVOLVE SYSTEM
traj1d.run(steps)

## gather information
r_t = traj1d.data.r_t.flatten()
v_t = traj1d.data.v_t.flatten()
E_t = traj1d.data.E_t.flatten()
E_kin = traj1d.data.E_kin_t.flatten()
E_pot = traj1d.data.E_pot_t.flatten()

# PROCESS DATA FOR PLOTTING/VISUALIZATION
h = np.histogram(r_t, bins=np.arange(cell[0][0], cell[0][1], 0.1), density=True)
rbins = h[1]
prop_dist = h[0]

r_sampled = np.linspace(0.95*min(r_t), 1.05*max(r_t), 600)
V_r = traj1d.potential(r_sampled)

# VISUALIZATION
# probability distribution
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.gca()
ax.set_xlabel('$r$ $[a.u.]$')
ax1 = ax.twinx()
ax1.plot(r_sampled, V_r, c='r', lw=1.25, label='$V(r)$')
ax.plot(np.linspace(min(rbins)+(rbins[1]-rbins[0])/2.,max(rbins)-(rbins[1]-rbins[0])/2.,len(prop_dist)), \
        prop_dist, c='b', label=r'$\mathcal{P}(r)$')
ax1.set_ylim(min(V_r)-0.1*max(V_r), max(V_r)*1.1)
plt.xlim(r_sampled[0], r_sampled[-1])
ax1.set_ylabel(r'$\mathcal{V}(r)$ $[a.u.]$', color='r')
for tl in ax1.get_yticklabels():
    tl.set_color('r')
ax.set_ylabel(r'$\mathcal{P}(r)$', color='b')
for tl in ax.get_yticklabels():
    tl.set_color('b')

bla = '%E' %steps
txtsteps = bla.split('E')[0].rstrip('0').rstrip('.') + 'E' + bla.split('E')[1]
plt.title('Propability distribution for '+txtsteps+' steps'.format(steps))

plt.savefig('prop_dist_classical_'+str(int(steps))+'steps.pdf')


# visualization of dynamics
import matplotlib.animation as animation
fig = plt.figure()
ax = plt.subplot2grid((4,1), (1,0), rowspan=3, colspan=2)
ax1 = plt.subplot2grid((4,1), (0,0))

wave_plot1, = ax.plot(r_t[0], traj1d.potential(r_t[0]), ls='',marker='o',mfc='b',mec='b',ms=8.5)


def _init_():
    ax.plot(r_sampled, V_r, ls='-', c='r', lw=1.25, zorder=3)
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

    ax.set_xlabel('$r$ $[a.u.]$')
    ax.set_ylim(min(V_r)-0.1*max(V_r), max(V_r)*1.1)
    ax.set_xlim(r_sampled[0], r_sampled[-1])
    ax.legend(loc='best', numpoints=1)
    ax.set_ylabel(r'$\mathcal{V}(r)$ $[a.u.]$', color='r')
    for tl in ax.get_yticklabels():
        tl.set_color('r')

    return wave_plot1,

def animate(i):
    wave_plot1.set_data(r_t[i], traj1d.potential(r_t[i]))  # update data
    wave_plot1.set_label('$r(t$ $=$ ${0:>4.1f}$ $au)$' .format(i*dt))
    ax.legend(loc='best',numpoints=1)

    return wave_plot1,

ani = animation.FuncAnimation(fig, animate, np.arange(0, len(r_t)), init_func=_init_, \
                              interval=20, blit=False)

plt.show()
