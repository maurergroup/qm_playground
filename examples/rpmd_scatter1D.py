import numpy as np
from qmp.potential import Potential
from qmp.potential import preset_potentials
from qmp.integrator.rpmdintegrators import RPMD_Scattering
from qmp.systems.rpmd import RPMD
from qmp import Model


# SIMULATION CELL
cell = [[-10., 10.0]]
mass = [1800] * 2
r = [[0], [0]]
v = [[0.002], [0.001]]
dt = 82.


# POTENTIAL
wall = preset_potentials.Wall(1, position=[5.],
                              width=np.array([5]), height=[10000])
pot = Potential(cell, f=wall())
integrator = RPMD_Scattering(dt)
system = RPMD(r, v, mass)

# INITIALIZE MODEL
tik1d = Model(
        system=system,
        potential=pot,
        integrator=integrator,
        mode='wave'
        )

print(tik1d)

# INITIAL WAVE FUNCTION AND DYNAMICS PARAMETERS
# time step, number of steps
steps = 2E2

# EVOLVE SYSTEM
tik1d.run(steps)

print(tik1d.data.p_trans)
