"""
time independent
solutions to the 1D Particle in different
potentials
"""

import numpy as np
import qmp
from qmp.tools.visualizations import wave_slideshow1D

N = 512
cell = np.array([[-10, 10]])
mass = 1
system = qmp.systems.Grid(mass, cell, N)


# POTENTIAL
f = qmp.potential.presets.Harmonic(1)
pot = qmp.potential.Potential(cell, f=f())

# INITIALIZE MODEL
# number of lowest eigenstates to be solved for
states = 30

wave_model = qmp.Model(
            potential=pot,
            system=system,
            mode='wave',
            states=states,
            )

print(wave_model)

wave_model.solve()

psi = wave_model.system.basis
x = wave_model.system.mesh[0]

# VISUALIZATION
wave_slideshow1D(x, psi, wave_model.potential(x))
