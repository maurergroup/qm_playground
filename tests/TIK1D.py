"""
time dependent and time independent 
solutions to the 1D Particle in a box
"""

import numpy as np
from qmp import *

#we dont define a  potential

def f(x):
    x = np.array([x]).flatten()
    for i,xx in enumerate(x):
        if xx>4.0 and xx<6.0:
            x[i]= 5.0
        else:
            x[i]= 0.0
    return x

cell = [[0, 10.0]]

pot = Potential(cell, f=f)

#initialize the model
tik1d = Model(
        ndim=1,
        mass=1.0,
        mode='wave',
        basis='onedgrid',
        solver='alglib',
        )

#set the potential
tik1d.set_potential(pot)

#set basis 
N=50
b = onedgrid(cell[0][0], cell[0][1],N)
tik1d.set_basis(b)

print tik1d

tik1d.solve()

from matplotlib import pyplot as plt

print tik1d.basis.construct_Tmatrix()
print tik1d.basis.construct_Vmatrix(tik1d.pot)
print tik1d.data.E
print tik1d.data.psi

plt.plot(tik1d.basis.x,tik1d.data.psi[:,5])
plt.show()

from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons


