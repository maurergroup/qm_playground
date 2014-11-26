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

#we dont define a  potential

##free particle 
#def f(x):
    #return np.zeros_like(x)

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
def f(x):
    x = np.array([x]).flatten()
    for i,xx in enumerate(x):
        if (xx<= 4.0 and xx>2.0) or (xx<= 8.0 and xx>6.0) :
            x[i]= 0.0
        else:
            x[i]= 100.0
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
        integrator='primprop',
        )

#set the potential
tik1d.set_potential(pot)

#set basis 
N=10  # of states
b = onedgrid(cell[0][0], cell[0][1],N)
tik1d.set_basis(b)

print tik1d

#calculate eigenvalues and eigenfunctions
tik1d.solve()

print tik1d.data.wvfn.E

#prepare initial wavefunction and dynamics
dt =  0.0001  #whatever the units are
steps = 0

tik1d.run(steps, dt)

#prepare wvfn
tik1d.data.c[0] = 0.5
tik1d.data.c[1] = 0.5
norm = np.dot(tik1d.data.c,tik1d.data.c)
tik1d.data.c /= np.sqrt(norm)

tik1d.run(2,dt)




#####VISUALIZATION

#from matplotlib import pyplot as plt
##generate figure
#fix, ax = plt.subplots()
#plt.subplots_adjust(bottom=0.2)
#l, = plt.plot(tik1d.basis.x,tik1d.data.psi[:,0])
#k, = plt.plot(tik1d.basis.x,tik1d.pot(tik1d.basis.x))
#ax.set_ylim([-0.6,0.6])

#from matplotlib.widgets import Slider, Button, RadioButtons
##BUTTON DEFINITIONS
#class Index:
    #ind = 0
    #def next(self, event):
        #self.ind += 1
        #if self.ind == N:
            #self.ind = 0
        #l.set_ydata(tik1d.data.psi[:,self.ind])
        #plt.draw()

    #def prev(self, event):
        #self.ind -= 1
        #if self.ind == -1:
            #self.ind = N-1
        #l.set_ydata(tik1d.data.psi[:,self.ind])
        #plt.draw()

#callback = Index()
#pos_prev_button = plt.axes([0.7,0.05,0.1,0.075])
#pos_next_button = plt.axes([0.81,0.05,0.1,0.075])
#button_next = Button(pos_next_button, 'Next Wvfn')
#button_next.on_clicked(callback.next)
#button_prev = Button(pos_prev_button, 'Prev Wvfn')
#button_prev.on_clicked(callback.prev)

#plt.show()

