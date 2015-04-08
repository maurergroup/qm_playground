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

x = np.linspace(cell[0][0], cell[1][0], 500)
y = np.linspace(cell[0][1], cell[1][1], 500)
xv, yv = np.meshgrid(x,y)
V = f_mexican(xv,yv)

#import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import axes3d
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(xv,yv,V, lw=0., cmap=cm.coolwarm)
#plt.show()

#raise SystemExit

pot = Potential2D(cell, f=f_mexican)

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
N=800  # spatial discretization
b = twodgrid(cell[0], cell[1], N)
tik2d.set_basis(b)

states = 20

print tik2d

tik2d.solve()

#print tik2d.data.wvfn.E.shape
psi = tik2d.data.wvfn.psi


####VISUALIZATION

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#x = np.ones(len(tik2d.data.wvfn.E))
#plt.plot(x, tik2d.data.wvfn.E, ls='', marker = '_')#, mew=8, ms=6)
#plt.show()
#plt.hist(tik2d.data.wvfn.E, 7)
#plt.show()

#generate figure
fig = plt.figure()
plt.subplots_adjust(bottom=0.2)

#fig = plt.figure()
ax = fig.gca(projection='3d')
l = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                     np.reshape(psi[:,0], (N,N)), lw=0., cmap=cm.coolwarm)


from matplotlib.widgets import Slider, Button, RadioButtons
#BUTTON DEFINITIONS
class Index:
    ind = 0
    def next(self, event):
        self.ind += 1
        if self.ind == states:
            self.ind = 0
        ax.clear()
        l = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                     np.reshape(psi[:,self.ind], (N,N)), lw=0., cmap=cm.coolwarm)

        plt.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind == -1:
            self.ind = states-1
        ax.clear()
        l = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                     np.reshape(psi[:,self.ind], (N,N)), lw=0., cmap=cm.coolwarm)
        plt.draw()

callback = Index()
pos_prev_button = plt.axes([0.7,0.05,0.1,0.075])
pos_next_button = plt.axes([0.81,0.05,0.1,0.075])
button_next = Button(pos_next_button, 'Next Wvfn')
button_next.on_clicked(callback.next)
button_prev = Button(pos_prev_button, 'Prev Wvfn')
button_prev.on_clicked(callback.prev)

plt.show()

