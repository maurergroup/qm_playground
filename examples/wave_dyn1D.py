import numpy as np
from qmp.potential import Potential
from qmp.integrator.dyn_tools import create_gaussian
from qmp.potential import preset_potentials
from qmp.tools.visualizations import *
from qmp.integrator.waveintegrators import SOFT_Propagator
from qmp.systems.grid import Grid1D
from qmp import Model


# SIMULATION CELL
cell = [[-20., 20.0]]
N = 400
mass = 1800
dt = 82.


# POTENTIAL
harm = preset_potentials.Harmonic(1, minimum=[0.], omega=[0.005])
pot = Potential(cell, f=harm())
integrator = SOFT_Propagator(dt)
system = Grid1D(mass, cell[0][0], cell[0][1], N)

# initial wave functions
sigma = 1./2.
psi_0 = create_gaussian(system.x, x0=0., p0=1.0, sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))

system.set_initial_wvfn(psi_0)

# NUMBER OF BASIS STATES
# for propagation in eigenbasis
states = 128

# INITIALIZE MODEL
tik1d = Model(
        system=system,
        potential=pot,
        integrator=integrator,
        mode='wave',
        solver='scipy',
        )

print(tik1d)
print('Grid points:', N, '\n')

# INITIAL WAVE FUNCTION AND DYNAMICS PARAMETERS
# time step, number of steps
steps = 2E2

# EVOLVE SYSTEM
tik1d.run(steps)

# GATHER INFO
# info time evolution
psi_t = tik1d.data.psi_t
# c_t = tik1d.data..c_t
E_t = tik1d.data.E_t
E_kin_t = tik1d.data.E_kin_t
E_pot_t = tik1d.data.E_pot_t

rho_t = np.sum(psi_t*np.conjugate(psi_t),1)
rho_r_mean = np.mean(psi_t*np.conjugate(psi_t), 0)
r_mean = np.dot(tik1d.system.x, rho_r_mean)
f = open('wave_dyn_'+str(N)+'gridpts_'+str(int(steps))+'steps.log', 'w')
f.write('wave simulation in {0:d} by {1:d} cell\n\n'.format(int(cell[0][0]), int(cell[0][1])))
f.write("E_mean = {0} Ha\n".format(np.real(np.mean(E_t))))
f.write('r_mean = {0:16.8f} a_0\n'.format(r_mean))
f.close()
V_x = tik1d.potential(tik1d.system.x)

### VISUALIZATION ###

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.gca()
ax0 = ax.twinx()
ax.plot(tik1d.system.x, rho_r_mean)
ax0.plot(tik1d.system.x, tik1d.potential(tik1d.system.x), ls=':', c='r')
ax0.set_ylim(min(tik1d.potential(tik1d.system.x)), 11.)
#plt.savefig('wave_dyn_'+str(N)+'gridpts_'+str(int(steps))+'steps.pdf')

#plt.plot(np.conjugate(c_t[:,1])*c_t[:,1], label=r'$\Vert c_1(t)\Vert^2$')
#plt.plot(np.conjugate(c_t[:,2])*c_t[:,2], label=r'$\Vert c_2(t)\Vert^2$')
#plt.plot(np.conjugate(c_t[:,3])*c_t[:,3], label=r'$\Vert c_3(t)\Vert^2$')
#plt.legend(loc='best')
#plt.show()


## view animation
wave_movie1D(tik1d.system.x, psi_t*np.conjugate(psi_t), V_x, dt=dt, E_arr=E_t, rho_tot_arr=rho_t, E_kin_arr=E_kin_t, E_pot_arr=E_pot_t)
