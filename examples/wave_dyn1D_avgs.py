import numpy as np
from qmp.potential import Potential
from qmp.integrator.dyn_tools import create_gaussian
from qmp.potential import preset_potentials
from qmp.tools.visualizations import *
from qmp.integrator.waveintegrators import SOFT_AverageProperties
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
integrator = SOFT_AverageProperties(dt)
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
# c_t = tik1d.data..c_t
E_t = tik1d.data.E_tot
E_kin_t = tik1d.data.E_kin
E_pot_t = tik1d.data.E_pot
print(E_t, E_kin_t, E_pot_t)
