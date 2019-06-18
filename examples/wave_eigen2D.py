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
from qmp.potential import Potential, preset_potentials
from qmp.tools.visualizations import wave_slideshow2D


### SIMULATION CELL ###
cell = [[0, 0.], [20., 20.]]

### POTENTIAL ###
f = preset_potentials.Harmonic(2, minimum=[10, 10])
pot = Potential(cell, f=f())

### INITIALIZE MODEL ###
## number of lowest eigenstates to be solved for
states = 20

tik2d = Model(
        ndim=2,
        mass=1.0,
        mode='wave',
        basis='twodgrid',
        solver='scipy',
        states=states,
        )

## set potential
tik2d.set_potential(pot)

### BASIS ###
## spatial discretization
N=200
b = twodgrid(cell[0], cell[1], N)
tik2d.set_basis(b)
print(tik2d)

tik2d.solve()

# GATHER INFORMATION ###
psi = tik2d.data.wvfn.psi
V_xy = tik2d.pot(tik2d.basis.xgrid, tik2d.basis.ygrid)

# VISUALIZATION ###
wave_slideshow2D(tik2d.basis.xgrid, tik2d.basis.ygrid, psi, pot=V_xy)
