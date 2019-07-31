"""
time independent
solutions to the 2D Particle in different
potentials
"""

import numpy as np
from qmp import Model
from qmp.systems.grid import Grid2D
from qmp.potential import Potential
from qmp.integrator.dyn_tools import create_gaussian2D
from qmp.tools.visualizations import wave_movie2D
from qmp.potential.pot_tools import create_potential2D
from qmp.potential.preset_potentials import Elbow
from qmp.integrator.waveintegrators import SOFT_Propagator


cell = [[0, 0.], [20., 20.]]
mass = 1
N = 200
dt = .1
steps = 100

elbow = Elbow(2)
pot = Potential(cell, f=elbow())
system = Grid2D(mass, cell[0], cell[1], N)
integrator = SOFT_Propagator(dt)

# INITIALIZE MODEL
tik2d = Model(
        mode='wave',
        system=system,
        solver='scipy',
        integrator=integrator,
        potential=pot
        )

print(tik2d)

# initial wavefunction
psi_0 = create_gaussian2D(tik2d.system.xgrid, tik2d.system.ygrid,
                          x0=[2., 12.], p0=[0., 0.], sigma=[1., 1.])
psi_0 /= np.sqrt(np.conjugate(psi_0.flatten()).dot(psi_0.flatten()))

system.set_initial_wvfn(psi_0)

tik2d.run(steps)

# GATHER INFO
psi_t = tik2d.data.psi_t     #(steps+1, x**ndim)
V_xy = tik2d.potential(tik2d.system.xgrid, tik2d.system.ygrid)
rho_t = np.sum(psi_t*np.conjugate(psi_t),1)
E_t = tik2d.data.E_t
# if tik2d.parameters['integrator'] == 'SOFT':
#     E_kin_t = tik2d.data.E_kin_t
#     E_pot_t = tik2d.data.E_pot_t


### VISUALIZATION ###
wave_movie2D(tik2d.system.xgrid, tik2d.system.ygrid, np.real(psi_t), pot=V_xy)
