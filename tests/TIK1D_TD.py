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
            x[i] = 2.
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

## 1D "mexican hat"
def f_mexican(x):
    sigma = 1.
    pref = 5./(np.sqrt(3*sigma)*np.pi**(1./4.))
    brak = 1.-((x-10.)/sigma)**2
    f = pref*(brak*np.exp(-(1./2.)*((x-10.)/sigma)**2))
    return f - min(f)

cell = [[0, 30.0]]

pot = Potential(cell, f=f_mexican)

#number of basis states to consider
states = 40

#initialize the model
tik1d = Model(
        ndim=1,
        mass=1.0,
        mode='wave',
        basis='onedgrid',
        solver='scipy',
        #solver='alglib',
        integrator='splitopprop',
        states=states,
        )

#set the potential
tik1d.set_potential(pot)

#set basis 
N=500  # spatial discretization
b = onedgrid(cell[0][0], cell[0][1],N)
tik1d.set_basis(b)
print tik1d

#calculate eigenvalues and eigenfunctions
#tik1d.solve()
#print 'SOLVED'

##prepare initial wavefunction and dynamics
dt =  0.5  #whatever the units are ~> a.u.?
steps = 200
psi_0 = create_gaussian(tik1d.basis.x, x0=13.)

tik1d.run(0, dt, psi_0x = psi_0)

#prepare wvfn
#tik1d.data.c[0] = 1
#tik1d.data.c[1] = 1
#tik1d.data.c = project_gaussian(tik1d.data.wvfn.psi, tik1d.basis.x, \
#                                amplitude=1., sigma=1., x0=13.)
#norm = np.dot(tik1d.data.c,tik1d.data.c)
#tik1d.data.c /= np.sqrt(norm)

tik1d.run(steps,dt, psi_0x=psi_0)
print 'INTEGRATED'
psi_t = tik1d.data.wvfn.psi_t
#E_t = tik1d.data.wvfn.E_t


#####VISUALIZATION
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#plt.plot(np.linspace(0., steps*dt, steps+1), E_t)
#plt.show()


fig, ax = plt.subplots()
line, = ax.plot(tik1d.basis.x, psi_t[0,:]*np.conjugate(psi_t[0,:]))
#ax.set_ylim(-0.002, 0.08)

def _init_():
#    ax.set_ylim(-0.002,0.08)
    return line,

def animate(i):
    line.set_ydata(psi_t[i,:]*np.conjugate(psi_t[i,:]))  # update the data
    ax.plot(tik1d.basis.x, tik1d.pot(tik1d.basis.x)/max(tik1d.pot(tik1d.basis.x))*0.15, ls=':', c='r')
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, steps), init_func=_init_, \
                              interval=50, blit=False)

plt.show()

raise SystemExit
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

