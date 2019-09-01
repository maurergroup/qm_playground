"""
time independent
solutions to the 2D Particle in different
potentials
"""

import qmp
import numpy as np
from qmp.tools.visualizations import wave_slideshow2D

# SIMULATION CELL
N = 512
cell = np.array([[-10, 10], [-10., 10.]])
mass = 1
system = qmp.systems.Grid(mass, cell, N)

# POTENTIAL
f = qmp.potential.presets.Harmonic(2)
pot = qmp.potential.Potential(cell, f=f())

# INITIALIZE MODEL
# number of lowest eigenstates to be solved for
states = 20

tik2d = qmp.Model(
        mode='wave',
        system=system,
        potential=pot,
        states=states,
        )

print(tik2d)

tik2d.solve()

psi = tik2d.system.basis
V_xy = tik2d.potential(*tik2d.system.mesh)

# VISUALIZATION
wave_slideshow2D(*tik2d.system.mesh, psi, pot=V_xy)
