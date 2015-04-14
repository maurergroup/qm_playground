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
from qmp.visualizations import slideshow2D


## 2D harmonic potential
def f_harm(x,y):
    omx, omy = .5, .5
    return omx*((x-10.)**2) + omy*((y-10.)**2)

## 2D "mexican hat potential"
def f_mexican(x,y):
    sigma = 1.
    pref = (1./(np.pi*sigma**4))
    brak = .5-(((x-10.)**2+(y-10.)**2)/(2*sigma**2))
    f = pref*brak*np.exp(-(((x-10.)**2+(y-10)**2)/(2.*sigma**2)))
    return f - min(f.flatten())
        

cell = [[0, 0.], [20., 20.]]

pot = Potential2D(cell, f=f_mexican)

#initialize the model
tik2d = Model(
        ndim=2,
        mass=1.0,
        mode='wave',
        basis='twodgrid',
        #solver='alglib',
        solver='scipy',
        states=20,
        )

#set the potential
tik2d.set_potential(pot)

#set basis 
N=200  # spatial discretization
b = twodgrid(cell[0], cell[1], N)
tik2d.set_basis(b)

print tik2d

tik2d.solve()
print 'SOLVED'

psi = tik2d.data.wvfn.psi
V_xy = tik2d.pot(tik2d.basis.xgrid, tik2d.basis.ygrid)

####VISUALIZATION

slideshow2D(tik2d.basis.xgrid, tik2d.basis.ygrid, psi, pot=V_xy)
