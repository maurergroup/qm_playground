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
from qmp.integrator.dyn_tools import project_gaussian2D


## 2D harmonic potential
def f(x,y):
    omx, omy = .5, .5
    return omx*((x-10.)**2) + omy*((y-10.)**2)

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
        integrator='primprop',
        )

#set the potential
tik2d.set_potential(pot)

#set basis 
N=200     # spatial discretization
b = twodgrid(cell[0], cell[1], N)
tik2d.set_basis(b)

print tik2d

tik2d.solve()

#print tik2d.data.wvfn.E.shape

##prepare initial wavefunction and dynamics
dt =  0.5     #whatever the units are ~> a.u.?
steps = 10     #number of steps to propagate

tik2d.run(0, dt)

#prepare wvfn
#tik2d.data.c[0] = 1
#tik2d.data.c[1] = 1
tik2d.data.c = project_gaussian2D(tik2d.data.wvfn.psi, tik2d.basis.xgrid, \
                     tik2d.basis.ygrid, amplitude=10., sigma=1., x0=[8.,8.])
norm = np.dot(tik2d.data.c,tik2d.data.c)
tik2d.data.c /= np.sqrt(norm)

tik2d.run(steps,dt)

# number steps: initial + propagated = steps + 1
psi_t = tik2d.data.wvfn.psi_t     #(steps+1, x**ndim)


####VISUALIZATION

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('QT4Agg')

#generate figure
fig = plt.figure()
plt.subplots_adjust(bottom=0.2)
ax = fig.gca(projection='3d')
l = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                     np.reshape(psi_t[0,:], (N,N)), alpha=0.75)
k = ax.plot_surface(tik2d.basis.xgrid[50:150,50:150], tik2d.basis.ygrid[50:150,50:150], \
        tik2d.pot(tik2d.basis.xgrid, tik2d.basis.ygrid)[50:150,50:150], color='red', alpha=0.25)
ax.set_zlim(-0.1,0.1)
ax.view_init(elev=4. , azim=-45.)

from matplotlib.widgets import Slider, Button, RadioButtons
#BUTTON DEFINITIONS
class Index:
    ind = 0
    def next(self, event):
        self.ind += 1
        if self.ind == steps+2:
            self.ind = 0
        l = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                     np.reshape(psi_t[self.ind,:], (N,N)), alpha=0.75)
#        l.set_data(np.reshape(psi[:,self.ind]), (N,N))
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind == -1:
            self.ind = steps
        l = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                     np.reshape(psi_t[self.ind,:], (N,N)), alpha=0.75)
#        l.set_data(np.reshape(psi[:,self.ind]), (N,N))
        plt.draw()

callback = Index()
pos_prev_button = plt.axes([0.7,0.05,0.1,0.075])
pos_next_button = plt.axes([0.81,0.05,0.1,0.075])
button_next = Button(pos_next_button, 'Next Wvfn')
button_next.on_clicked(callback.next)
button_prev = Button(pos_prev_button, 'Prev Wvfn')
button_prev.on_clicked(callback.prev)

plt.show()

