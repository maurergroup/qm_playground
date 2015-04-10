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
from qmp.integrator.dyn_tools import create_gaussian2D


## 2D harmonic potential
def f_harm(x,y):
    omx, omy = .025, .025
    return omx*((x-10.)**2) + omy*((y-10.)**2)

## 2D "mexican hat potential"
def f_mexican(x,y):
    sigma = 1.
    pref = 5./(np.pi*sigma**4)
    brak = .5-(((x-10.)**2+(y-10.)**2)/(2*sigma**2))
    f = pref*brak*np.exp(-(((x-10.)**2+(y-10)**2)/(2.*sigma**2)))
    return f - min(f.flatten())



cell = [[0, 0.], [20., 20.]]

#x = np.linspace(0., 20., 200)
#y = np.linspace(0., 20., 200)
#xval, yval = np.meshgrid(x,y)
#V = f_mexican(xval, yval)
#from matplotlib import pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(xval, yval, V, cmap=cm.coolwarm)
#plt.show()
#raise SystemExit

pot = Potential2D(cell, f=f_harm)

#initialize the model
tik2d = Model(
        ndim=2,
        mass=1.0,
        mode='wave',
        basis='twodgrid',
        #solver='alglib',
        solver='scipy',
        integrator='SOFT',
        )

#set the potential
tik2d.set_potential(pot)

#set basis 
N=400     # spatial discretization
b = twodgrid(cell[0], cell[1], N)
tik2d.set_basis(b)
print tik2d


##prepare initial wavefunction and dynamics
dt =  .5     #whatever the units are ~> a.u.?
steps = 120     #number of steps to propagate

psi_0 = create_gaussian2D(tik2d.basis.xgrid, tik2d.basis.ygrid, x0=[8.,9.], p0=[1.,2.], sigma=1.)

tik2d.run(0, dt, psi_0=psi_0)

#prepare wvfn
#tik2d.data.c[0] = 1
#tik2d.data.c[1] = 1
#tik2d.data.c = project_gaussian2D(tik2d.data.wvfn.psi, tik2d.basis.xgrid, \
#                     tik2d.basis.ygrid, amplitude=10., sigma=1., x0=[7.,9.])
#norm = np.dot(tik2d.data.c,tik2d.data.c)
#tik2d.data.c /= np.sqrt(norm)

tik2d.run(steps,dt, psi_0=psi_0)
print 'INTEGRATED'

# number steps: initial + propagated = steps + 1
psi_t = tik2d.data.wvfn.psi_t     #(steps+1, x**ndim)


####VISUALIZATION

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('QT4Agg')
theta = 25.

#generate figure
fig = plt.figure()
plt.subplots_adjust(bottom=0.2)
ax = fig.gca(projection='3d')

frame = None
for i in xrange(steps+1):
    oldframe = frame

    frame = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                    8.*np.reshape(psi_t[i,:]*np.conjugate(psi_t[i,:]), (N,N)), alpha=0.75, \
                    antialiased=False, cmap = cm.coolwarm, lw=0.)
    ax.set_zlim(-0.0015,0.0015)
    ax.view_init(elev=theta , azim=-45.)
    fmng = plt.get_current_fig_manager()
    fmng.window.showMaximized()

    if oldframe is not None:
        ax.collections.remove(oldframe)
        
    plt.pause(0.001)
    

raise SystemExit
from matplotlib.widgets import Slider, Button, RadioButtons
#BUTTON DEFINITIONS
class Index:
    ind = 0
    def next(self, event):
        self.ind += 1
        if self.ind == steps+1:
            self.ind = 0
        ax.clear()
        l = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                    np.reshape(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]), (N,N)), alpha=0.75, \
                    antialiased=False, cmap = cm.coolwarm)
        #k = ax.plot_surface(tik2d.basis.xgrid[50:150,50:150], tik2d.basis.ygrid[50:150,50:150], \
        #            tik2d.pot(tik2d.basis.xgrid, tik2d.basis.ygrid)[50:150,50:150], \
        #            color='red', alpha=0.05, linewidth=0.)
        ax.plot([10.,10.,10.], [10.,10.,10.], [-10., 0., 10.], marker='o', mfc='green', ms=8, lw=0.)
        cset = ax.contour(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                  np.reshape(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]), (N,N)), \
                  zdir='z', offset=-0.0025, cmap=cm.coolwarm)
        cset = ax.contour(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                  np.reshape(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]), (N,N)), \
                  zdir='x', offset=cell[0][0], cmap=cm.coolwarm)
        cset = ax.contour(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                  np.reshape(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]), (N,N)), \
                  zdir='y', offset=cell[1][1], cmap=cm.coolwarm)
        ax.set_zlim(-0.0025,0.0025)
        ax.view_init(elev=theta , azim=-45.)

#        l.set_data(np.reshape(psi[:,self.ind]), (N,N))
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind == -1:
            self.ind = steps
        ax.clear()
        l = ax.plot_surface(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                    np.reshape(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]), (N,N)), alpha=0.75, \
                    antialiased=False, cmap = cm.coolwarm)
        #k = ax.plot_surface(tik2d.basis.xgrid[50:150,50:150], tik2d.basis.ygrid[50:150,50:150], \
        #            tik2d.pot(tik2d.basis.xgrid, tik2d.basis.ygrid)[50:150,50:150], \
        #            color='red', alpha=0.05, linewidth=0.)
        ax.plot([10.,10.,10.], [10.,10.,10.], [-10., 0., 10.], marker='o', mfc='green', ms=8, lw=0.)
        cset = ax.contour(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                  np.reshape(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]), (N,N)), \
                  zdir='z', offset=-0.0025, cmap=cm.coolwarm)
        cset = ax.contour(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                  np.reshape(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]), (N,N)), \
                  zdir='x', offset=cell[0][0], cmap=cm.coolwarm)
        cset = ax.contour(tik2d.basis.xgrid, tik2d.basis.ygrid, \
                  np.reshape(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]), (N,N)), \
                  zdir='y', offset=cell[1][1], cmap=cm.coolwarm)
        ax.set_zlim(-0.0025,0.0025)
        ax.view_init(elev=theta , azim=-45.)

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

