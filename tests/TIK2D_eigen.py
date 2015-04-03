"""
time independent 
solutions to the 2D Particle in different 
potentials
"""

import numpy as np

import sys
sys.path.append('..')
from qmp import *
from qmp.basis.gridbasis import *
from qmp.potential2D import *


## 2D harmonic potential
def f(x,y):
    omx, omy = 1., 1.
    return omx*((x-5.)**2) + omy*((y-5.)**2)

cell = [[0, 0.], [20., 20.]]

pot = Potential2D(cell, f=f)

#initialize the model
tik2d = Model(
        ndim=2,
        mass=1.0,
        mode='wave',
        basis='twodgrid',
        #solver='alglib',
        solver='scipy',
        )

#set the potential
tik2d.set_potential(pot)

#set basis 
N=200  # of states
b = twodgrid(cell[0], cell[1], N)
tik2d.set_basis(b)

print tik2d

tik2d.solve()

print tik2d.data.wvfn.E

####VISUALIZATION

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.ones(len(tik2d.data.wvfn.E))
plt.plot(x, tik2d.data.wvfn.E, ls='', marker = '_', mew=8, ms=6)
plt.show()

#generate figure
#fig = plt.figure()
#plt.subplots_adjust(bottom=0.2)

fig = plt.figure()
ax = fig.gca(projection='3d')
l, = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                     np.reshape(tik2d.data.wvfn.psi[:,0], (N,N)))


from matplotlib.widgets import Slider, Button, RadioButtons
#BUTTON DEFINITIONS
class Index:
    ind = 0
    def next(self, event):
        self.ind += 1
        if self.ind == N:
            self.ind = 0
        l.set_ydata(tik2d.data.wvfn.psi[:,self.ind])
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind == -1:
            self.ind = N-1
        l.set_ydata(tik2d.data.wvfn.psi[:,self.ind])
        plt.draw()

callback = Index()
pos_prev_button = plt.axes([0.7,0.05,0.1,0.075])
pos_next_button = plt.axes([0.81,0.05,0.1,0.075])
button_next = Button(pos_next_button, 'Next Wvfn')
button_next.on_clicked(callback.next)
button_prev = Button(pos_prev_button, 'Prev Wvfn')
button_prev.on_clicked(callback.prev)

plt.show()

