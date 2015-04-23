"""
time independent 
solutions to the 2D Particle in different 
potentials
"""

import numpy as np

import sys
sys.path.append('..')
from qmp import *
from qmp.basis.gridbasis import twodgrid
from qmp.potential import Potential2D
from qmp.integrator.dyn_tools import create_gaussian2D
from qmp.visualizations import wave_movie2D
from qmp.pot_tools import create_potential2D


cell = [[0, 0.], [20., 20.]]

pot = Potential2D(cell, f=create_potential2D(cell, name='elbow'))

#initialize the model
tik2d = Model(
        ndim=2,
        mass=1.0,
        mode='wave',
        basis='twodgrid',
        solver='scipy',
        integrator='SOFT',
        states=100,
        )

#set the potential
tik2d.set_potential(pot)

#set basis 
N=200     # spatial discretization
b = twodgrid(cell[0], cell[1], N)
tik2d.set_basis(b)
print tik2d


##prepare initial wavefunction and dynamics
dt =  .1     #whatever the units are ~> a.u.?
steps = 100     #number of steps to propagate

psi_0 = create_gaussian2D(tik2d.basis.xgrid, tik2d.basis.ygrid, x0=[2.,12.], p0=[0.,0.], sigma=[1.,1.])
psi_0 /= np.sqrt(np.conjugate(psi_0.flatten()).dot(psi_0.flatten()))

tik2d.run(steps,dt, psi_0=psi_0)
print 'INTEGRATED'

# number steps: initial + propagated = steps + 1
psi_t = tik2d.data.wvfn.psi_t     #(steps+1, x**ndim)
V_xy = tik2d.pot(tik2d.basis.xgrid, tik2d.basis.ygrid)
rho_t = np.sum(psi_t*np.conjugate(psi_t),1)
E_t = tik2d.data.wvfn.E_t
if tik2d.parameters['integrator'] == 'SOFT':
    E_kin_t = tik2d.data.wvfn.E_kin_t
    E_pot_t = tik2d.data.wvfn.E_pot_t

####VISUALIZATION

import matplotlib.pyplot as plt
plt.plot(E_t)
plt.plot(E_kin_t)
plt.plot(E_pot_t)
plt.plot(rho_t)
plt.show()

wave_movie2D(tik2d.basis.xgrid, tik2d.basis.ygrid, psi_t, pot=V_xy)