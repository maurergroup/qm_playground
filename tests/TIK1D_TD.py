"""
time independent 
solutions to the 1D Particle in different 
potentials
"""

import numpy as np

import sys
sys.path.append('..')
from qmp import *
from qmp.basis.gridbasis import onedgrid
from qmp.integrator.dyn_tools import *

#we dont define a  potential

##free particle 
#def f(x):
    #return np.zeros_like(x)

def f_wall(x):
    x = np.array([x]).flatten()
    for i, xx in enumerate(x):
        if xx <= 1. or xx > 29.:
            x[i] = 10000000000000.
        elif xx >= 19. and xx < 21.:
            x[i] = 5
        else:
            x[i] = 0.
    return x

##particle in deep well 
def f(x):
    x = np.array([x]).flatten()
    for i,xx in enumerate(x):
        if xx<= 9.0 and xx>1.0:
            x[i]= 0.0
        else:
            x[i]= 1000000.0
    return x
##particle in two square wells 
def f_double_square(x):
    x = np.array([x]).flatten()
    for i,xx in enumerate(x):
        if (xx<= 4.0 and xx>2.0) or (xx<= 8.0 and xx>6.0) :
            x[i]= 0.0
        else:
            x[i]= 2.5
    return x
##particle in two close lying wells
def f_close_wells(x):
    x = np.array([x]).flatten()
    for i,xx in enumerate(x):
        if (xx<= 4.8 and xx>3.0) or (xx<= 7.0 and xx>5.2) :
            x[i]= 0.0
        elif xx>4.8 and xx<=5.2:
            x[i]=20.0
        else:
            x[i]= 1000000.0
    return x
## harmonic potential
def f_harm(x):
    omega = 1.5
    x = np.array([x]).flatten()
    for i, xx in enumerate(x):
        x[i] = omega* (xx -10.0)**2

    return x
## morse potential
def f_morse(x):
    a = 0.5
    D = 5.0
    x = np.array([x]).flatten()
    for i, xx in enumerate(x):
        x[i] = D* (1-np.exp(-a*(xx -10.0)))**2

    return x

cell = [[0, 30.0]]

pot = Potential(cell, f=f_wall)

#initialize the model
tik1d = Model(
        ndim=1,
        mass=1.0,
        mode='wave',
        basis='onedgrid',
        solver='scipy',
        #solver='alglib',
        integrator='primprop',
        )

#set the potential
tik1d.set_potential(pot)

#set basis 
N=200  # spatial discretization
b = onedgrid(cell[0][0], cell[0][1],N)
tik1d.set_basis(b)

#number of basis states to consider
states = 20

print tik1d

#calculate eigenvalues and eigenfunctions
tik1d.solve()

print tik1d.data.wvfn.E

##prepare initial wavefunction and dynamics
dt =  1  #whatever the units are ~> a.u.?
steps = 100

tik1d.run(0, dt)

#prepare wvfn
#tik1d.data.c[0] = 1
#tik1d.data.c[1] = 1
tik1d.data.c = project_gaussian(tik1d.data.wvfn.psi, tik1d.basis.x, 1., 18.5)
norm = np.dot(tik1d.data.c,tik1d.data.c)
tik1d.data.c /= np.sqrt(norm)

tik1d.run(steps,dt)


#####VISUALIZATION

from matplotlib import pyplot as plt
##generate figure
fix, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
#plt.plot(tik1d.basis.x,tik1d.data.wvfn.psi[0])
l, = plt.plot(tik1d.basis.x,tik1d.data.wvfn.psi_t[0,:]*np.conjugate(tik1d.data.wvfn.psi_t[0,:]))
k, = plt.plot(tik1d.basis.x,tik1d.pot(tik1d.basis.x))
ax.set_ylim([-0.6,.4])
#plt.show()
from matplotlib.widgets import Slider, Button, RadioButtons
##BUTTON DEFINITIONS
class Index:
    ind = 0
    def next(self, event):
        self.ind += 1
        if self.ind == steps:
            self.ind = 0
        l.set_ydata(tik1d.data.wvfn.psi_t[self.ind,:]*np.conjugate(tik1d.data.wvfn.psi_t[self.ind,:]))
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind == -1:
            self.ind = steps-1
        l.set_ydata(tik1d.data.wvfn.psi_t[self.ind,:]*np.conjugate(tik1d.data.wvfn.psi_t[self.ind,:]))
        plt.draw()

callback = Index()
pos_prev_button = plt.axes([0.7,0.05,0.1,0.075])
pos_next_button = plt.axes([0.81,0.05,0.1,0.075])
button_next = Button(pos_next_button, 'Next Wvfn')
button_next.on_clicked(callback.next)
button_prev = Button(pos_prev_button, 'Prev Wvfn')
button_prev.on_clicked(callback.prev)

plt.show()

