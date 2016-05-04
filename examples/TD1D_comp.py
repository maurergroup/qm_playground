import numpy as np
import sys
sys.path.append('..')
from qmp import *
from qmp.basis.gridbasis import onedgrid
from qmp.integrator.dyn_tools import *
from qmp.pot_tools import *

cell = [[0., 30.0]]

pot = Potential( cell, f=create_potential(cell, 'mexican_hat', mexican_scale=10.) )

#number of basis states to consider
states = 512

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
N=1024  # spatial discretization
b = onedgrid(cell[0][0], cell[0][1],N)
tik1d.set_basis(b)
print tik1d

##prepare initial wavefunction and dynamics
dt =  .1  #whatever the units are ~> a.u.?
steps = 200
sigma = .2
psi_0 = create_gaussian(tik1d.basis.x, x0=17., p0=0., sigma=sigma)
psi_0 = psi_0/np.sqrt(np.conjugate(psi_0).dot(psi_0))

tik1d.run(steps,dt, psi_0=psi_0)
print 'INTEGRATED'
psi_t = tik1d.data.wvfn.psi_t

E_t_SOFT = tik1d.data.wvfn.E_t
if tik1d.parameters['integrator'] == 'SOFT':
    E_kin_t = tik1d.data.wvfn.E_kin_t
    E_pot_t = tik1d.data.wvfn.E_pot_t
    
rho_t_SOFT = np.sum(psi_t*np.conjugate(psi_t),1)

tik1d_eigen = Model(
        ndim=1,
        mass=1.0,
        mode='wave',
        basis='onedgrid',
        solver='scipy',
        integrator='eigen',
        states=states,
        )

#set the potential
tik1d_eigen.set_potential(pot)

#set basis 
tik1d_eigen.set_basis(b)
print tik1d_eigen

tik1d_eigen.run(0, dt, psi_0 = psi_0)
tik1d_eigen.run(steps,dt, psi_0=psi_0)
print 'INTEGRATED'
psi_t_eigen = tik1d_eigen.data.wvfn.psi_t

E_t_eigen = tik1d_eigen.data.wvfn.E_t
if tik1d_eigen.parameters['integrator'] == 'SOFT':
    E_kin_t = tik1d_eigen.data.wvfn.E_kin_t
    E_pot_t = tik1d_eigen.data.wvfn.E_pot_t

rho_t_eigen = np.sum(psi_t_eigen*np.conjugate(psi_t_eigen),1)

tik1d_prim = Model(
        ndim=1,
        mass=1.0,
        mode='wave',
        basis='onedgrid',
        solver='scipy',
        integrator='primitive',
        states=states,
        )

#set the potential
tik1d_prim.set_potential(pot)

#set basis 
tik1d_prim.set_basis(b)
print tik1d_prim

tik1d_prim.run(steps,dt, psi_0=psi_0)
print 'INTEGRATED'
psi_t_prim = tik1d_prim.data.wvfn.psi_t

E_t_prim = tik1d_prim.data.wvfn.E_t
if tik1d_prim.parameters['integrator'] == 'SOFT':
    E_kin_t = tik1d_prim.data.wvfn.E_kin_t
    E_pot_t = tik1d_prim.data.wvfn.E_pot_t

rho_t_prim = np.sum(psi_t_prim*np.conjugate(psi_t_prim),1)

print E_t_SOFT[:10]
print E_t_eigen[:10]
print E_t_prim[:10]

#####VISUALIZATION

from matplotlib import pyplot as plt

##generate figure
fix, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
#plt.plot(tik1d.basis.x,tik1d.data.wvfn.psi[0])
l, = plt.plot(tik1d.basis.x,psi_t[0,:]*np.conjugate(psi_t[0,:]), c='b')
k, = plt.plot(tik1d_eigen.basis.x,psi_t_eigen[0,:]*np.conjugate(psi_t_eigen[0,:]), c='r', ls='--')
m, = plt.plot(tik1d_prim.basis.x,psi_t_prim[0,:]*np.conjugate(psi_t_prim[0,:]), c='g', ls=':')

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
        l.set_ydata(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]))
        k.set_ydata(psi_t_eigen[self.ind,:]*np.conjugate(psi_t_eigen[self.ind,:]))
        m.set_ydata(psi_t_prim[self.ind,:]*np.conjugate(psi_t_prim[self.ind,:]))
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind == -1:
            self.ind = steps-1
        l.set_ydata(psi_t[self.ind,:]*np.conjugate(psi_t[self.ind,:]))
        k.set_ydata(psi_t_eigen[self.ind,:]*np.conjugate(psi_t_eigen[self.ind,:]))
        m.set_ydata(psi_t_prim[self.ind,:]*np.conjugate(psi_t_prim[self.ind,:]))
        plt.draw()

callback = Index()
pos_prev_button = plt.axes([0.7,0.05,0.1,0.075])
pos_next_button = plt.axes([0.81,0.05,0.1,0.075])
button_next = Button(pos_next_button, 'Next Wvfn')
button_next.on_clicked(callback.next)
button_prev = Button(pos_prev_button, 'Prev Wvfn')
button_prev.on_clicked(callback.prev)

plt.show()

