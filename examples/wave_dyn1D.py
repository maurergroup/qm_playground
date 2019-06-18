import numpy as np
import sys
sys.path.append('..')
from qmp import *
from qmp.basis.gridbasis import onedgrid
from qmp.integrator.dyn_tools import *
from qmp.potential.pot_tools import *
from qmp.potential import preset_potentials
from qmp.tools.visualizations import *
from qmp.tools.termcolors import *


### SIMULATION CELL ###
cell = [[-20., 20.0]]

harm = preset_potentials.Harmonic(1, minimum=[0.], omega=[0.005])
### POTENTIAL ###
pot = Potential(cell, f=harm())

### NUMBER OF BASIS STATES ###
## for propagation in eigenbasis
states = 128

### INITIALIZE MODEL ###
tik1d = Model(
        ndim=1,
        mass=1850.0,
        mode='wave',
        solver='scipy',
        integrator='SOFT',
        #TODO instead of states which is eigenstate specific, use nonadiabatic yes or no
        states=states,
        )
### SET POTENTIAL ###
tik1d.set_potential(pot)

### SET BASIS ###
## number of grid points
N=400
b = onedgrid(cell[0][0], cell[0][1],N)
tik1d.set_basis(b)

print(tik1d)
print('grid points:',N)
print('')

### INITIAL WAVE FUNCTION AND DYNAMICS PARAMETERS ###
## time step, number of steps
dt =  82.
steps = 2E2

#tik1d.solve()

## initial wave functions
sigma = 1./2.
psi_0 = create_gaussian(tik1d.basis.x, x0=0., p0=1.0, sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))

### EVOLVE SYSTEM ###
tik1d.run(steps,dt, psi_0=psi_0)#, additional='coefficients')

## GATHER INFO ###
## info time evolution
psi_t = tik1d.data.wvfn.psi_t
#c_t = tik1d.data.wvfn.c_t
E_t = tik1d.data.wvfn.E_t
if tik1d.parameters['integrator'] == 'SOFT':
    E_kin_t = tik1d.data.wvfn.E_kin_t
    E_pot_t = tik1d.data.wvfn.E_pot_t
else:
    E_kin_t = None
    E_pot_t = None

rho_t = np.sum(psi_t*np.conjugate(psi_t),1)
rho_r_mean = np.mean(psi_t*np.conjugate(psi_t), 0)
r_mean = np.dot(tik1d.basis.x, rho_r_mean)
f = open('wave_dyn_'+str(N)+'gridpts_'+str(int(steps))+'steps.log', 'w')
f.write('wave simulation in {0:d} by {1:d} cell\n\n'.format(int(cell[0][0]), int(cell[0][1])))
f.write("E_mean = {0} Ha\n".format(np.real(np.mean(E_t))))
f.write('r_mean = {0:16.8f} a_0\n'.format(r_mean))
f.close()
V_x = tik1d.pot(tik1d.basis.x)

### VISUALIZATION ###

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.gca()
ax0 = ax.twinx()
ax.plot(tik1d.basis.x, rho_r_mean)
ax0.plot(tik1d.basis.x, tik1d.pot(tik1d.basis.x), ls=':', c='r')
ax0.set_ylim(min(tik1d.pot(tik1d.basis.x)), 11.)
#plt.savefig('wave_dyn_'+str(N)+'gridpts_'+str(int(steps))+'steps.pdf')

#plt.plot(np.conjugate(c_t[:,1])*c_t[:,1], label=r'$\Vert c_1(t)\Vert^2$')
#plt.plot(np.conjugate(c_t[:,2])*c_t[:,2], label=r'$\Vert c_2(t)\Vert^2$')
#plt.plot(np.conjugate(c_t[:,3])*c_t[:,3], label=r'$\Vert c_3(t)\Vert^2$')
#plt.legend(loc='best')
#plt.show()


## view animation
wave_movie1D(tik1d.basis.x, psi_t*np.conjugate(psi_t), V_x, dt=dt, E_arr=E_t, rho_tot_arr=rho_t, E_kin_arr=E_kin_t, E_pot_arr=E_pot_t)


#--EOF--#
