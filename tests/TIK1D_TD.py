"""
time dependent 
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
def f_free(x):
    return np.zeros_like(x)

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
def f_well(x):
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
    omega = .5
    x = np.array([x]).flatten()
    for i, xx in enumerate(x):
        x[i] = omega* (xx -15.)**2

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
    pref = 10./(np.sqrt(3*sigma)*np.pi**(1./4.))
    brak = 1.-((x-20.)/sigma)**2
    f = pref*(brak*np.exp(-(1./2.)*((x-20.)/sigma)**2))
    return f - min(f)

def f_gauss(x):
    return create_gaussian(x, sigma=5., x0=15.)

cell = [[0., 40.0]]

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
        integrator='SOFT',
        states=states,
        )

#set the potential
tik1d.set_potential(pot)

#set basis 
N=1000  # spatial discretization
b = onedgrid(cell[0][0], cell[0][1],N)
tik1d.set_basis(b)
print tik1d

##prepare initial wavefunction and dynamics
dt =  .1  #whatever the units are ~> a.u.?
steps = 1000
sigma = 1.
psi_0 = create_gaussian(tik1d.basis.x, x0=15., p0=1., sigma=sigma)

##analytical -- bogus!
#def rho_evol(x, sigma0, x0, p0, t):
#    sigma_t = sigma0*np.sqrt(1. + t**2/4./sigma0**4)
#    return (1./(np.sqrt(2*np.pi)*sigma_t))*np.exp( -(1./2.)*(x-x0-p0*t)**2 )

tik1d.run(0, dt, psi_0 = psi_0)

#prepare wvfn
#tik1d.data.c[0] = 1
#tik1d.data.c[1] = 1
#tik1d.data.c = project_gaussian(tik1d.data.wvfn.psi, tik1d.basis.x, \
#                                amplitude=1., sigma=1., x0=13.)
#norm = np.dot(tik1d.data.c,tik1d.data.c)
#tik1d.data.c /= np.sqrt(norm)

tik1d.run(steps,dt, psi_0=psi_0)
print 'INTEGRATED'
psi_t = tik1d.data.wvfn.psi_t
E_t = tik1d.data.wvfn.E_t
if tik1d.parameters['integrator'] == 'SOFT':
    E_kin_t = tik1d.data.wvfn.E_kin_t
    E_pot_t = tik1d.data.wvfn.E_pot_t
    
rho_t = np.sum(psi_t*np.conjugate(psi_t),1)

#####VISUALIZATION
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = plt.subplot2grid((4,2), (1,0), rowspan=3, colspan=2)
for tl in ax.get_yticklabels():
    tl.set_color('b')
ax0 = ax.twinx()
for tl in ax0.get_yticklabels():
    tl.set_color('r')
ax1 = plt.subplot2grid((4,2), (0,0))
ax2 = plt.subplot2grid((4,2), (0,1))

wave_plot, = ax.plot(tik1d.basis.x, psi_t[0,:]*np.conjugate(psi_t[0,:]), label=r'$\rho_t(x)$')

def _init_():
    ax0.plot(tik1d.basis.x, tik1d.pot(tik1d.basis.x), ls=':', c='r', label='$V(x)$')
    ax0.legend(loc=1)
    ax1.plot(np.linspace(0., (steps+1)*dt, (steps+1)), E_t, c='b', label='$E(t)$ $[a.u.]$')
    if tik1d.parameters['integrator'] == 'SOFT':
        ax1.plot(np.linspace(0., (steps+1)*dt, (steps+1)), E_kin_t, c='g', label='$E_{kin}(t)$ $[a.u.]$')
        ax1.plot(np.linspace(0., (steps+1)*dt, (steps+1)), E_pot_t, c='r', label='$E_{pot}(t)$ $[a.u.]$')
    
    ax1.legend(loc='best')
    #ax1.set_ylim(min(E_t)-0.001, max(E_t)+0.001)
    ax1.set_xlim([0., (steps+1.)*dt])
    ax1.set_xlabel('$t$ $[a.u.]$')
    ax1.xaxis.tick_top()
    ax1.set_xticks(ax1.get_xticks()[1:])
    ax1.xaxis.set_label_position('top')
    ax2.plot(np.linspace(0., (steps+1.)*dt, steps+1), rho_t, c='b', label=r'$\Vert\rho_t\Vert^2$ $[a.u.]$')
    ax2.legend(loc='best')
    ax2.set_ylim(min(rho_t)-0.001, max(rho_t)+0.001)
    ax2.set_xlim(0., (steps+1.)*dt)
    ax2.set_xlabel('$t$ $[a.u.]$')
    ax2.xaxis.tick_top()
    ax2.set_xticks(ax2.get_xticks()[1:])
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.tick_right()
    ax.set_xlabel('$x$ $[a.u.]$')
    ax.set_ylim(-0.0005, max((psi_t*np.conjugate(psi_t)).flatten()))
    ax.legend(loc=2)
    return wave_plot,

def animate(i):
    wave_plot.set_ydata(psi_t[i,:]*np.conjugate(psi_t[i,:]))  # update the data
    return wave_plot,

ani = animation.FuncAnimation(fig, animate, np.arange(1, steps), init_func=_init_, \
                              interval=50, blit=False)

plt.show()

raise SystemExit

t_i = np.arange(0., (steps+1)*dt, dt)
print t_i
from matplotlib import pyplot as plt

##generate figure
fix, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
#plt.plot(tik1d.basis.x,tik1d.data.wvfn.psi[0])
l, = plt.plot(tik1d.basis.x,tik1d.data.wvfn.psi_t[0,:]*np.conjugate(tik1d.data.wvfn.psi_t[0,:]))
k, = plt.plot(tik1d.basis.x,rho_evol(tik1d.basis.x, sigma, 13., 0., t_i[0]), c='r', ls=':')
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
        k.set_ydata(rho_evol(tik1d.basis.x, sigma, 13., 0., t_i[self.ind]))
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind == -1:
            self.ind = steps-1
        l.set_ydata(tik1d.data.wvfn.psi_t[self.ind,:]*np.conjugate(tik1d.data.wvfn.psi_t[self.ind,:]))
        k.set_ydata(rho_evol(tik1d.basis.x, sigma, 13., 0., t_i[self.ind]))
        plt.draw()

callback = Index()
pos_prev_button = plt.axes([0.7,0.05,0.1,0.075])
pos_next_button = plt.axes([0.81,0.05,0.1,0.075])
button_next = Button(pos_next_button, 'Next Wvfn')
button_next.on_clicked(callback.next)
button_prev = Button(pos_prev_button, 'Prev Wvfn')
button_prev.on_clicked(callback.prev)

plt.show()

