"""
time dependent and time independent 
solutions to the 1D Particle in a box
"""

import numpy as np
from qmp import *

#we dont define a  potential

cell = [[0, 10.0]]

pot = Potential(cell)

#initialize the model
tik1d = Model(
        ndim=1,
        mass=1.0,
        mode='wave',
        basis='onedgrid',
        solver='numpy',
        )

#set the potential
tik1d.set_potential(pot)

#set basis 
N=50
b = onedgrid(cell[0][0], cell[0][1],N)
tik1d.set_basis(b)

print tik1d

H = tik1d.b.construct_Tmatrix()
[E,V] = np.linalg.eig(H)

from matplotlib import pyplot as plt

plt.plot(tik1d.b.x,V[:,-5])
plt.show()



