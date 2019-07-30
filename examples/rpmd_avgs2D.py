import numpy as np
from qmp import Model
from qmp.potential import Potential
from qmp.integrator.rpmdintegrators import RPMD_EquilibriumProperties
from qmp.systems.rpmd import RPMD
from qmp.potential.preset_potentials import Elbow
from qmp.tools.visualizations import probability_distribution2D
from qmp.tools.visualizations import contour_movie2D

# SIMULATION CELL
cell = [[0., 20.], [0., 20.]]

# DYNAMICS PARAMETERS
dt = 2.
steps = 1E4

# SET INITIAL VALUES
rs = [[15., 4.5], [15., 3.5], [15, 4.0]]
vs = [[0., 0.], [0., 0.], [0., 0.]]
masses = [1860.] * 3
n_beads = 8
Temp = [300.] * 3

# POTENTIAL
f = Elbow(2, elbow_scale=0.005)

# THERMOSTAT
thermo = {
         'name':  'Andersen',
         'cfreq': 1E-4,
         'T_set': 100.,
         }

pot = Potential(cell, f=f())
integrator = RPMD_EquilibriumProperties(dt)
system = RPMD(rs, vs, masses, n_beads, init_type='velocity')

# INITIALIZE MODEL
rpmd2d = Model(
         mode='rpmd',
         integrator=integrator,
         system=system,
         potential=pot
        )

print(rpmd2d)


# EVOLVE SYSTEM
rpmd2d.run(steps)  # , dyn_T='Rugh')

# gather information
print('Average Properties:')
print('Positions: ', rpmd2d.data.r_mean)
print('Kinetic Energy: ', rpmd2d.data.E_kin)
print('Potential Energy: ', rpmd2d.data.E_pot)
print('Total Energy: ', rpmd2d.data.E_t)
