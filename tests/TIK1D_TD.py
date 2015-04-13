############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.gridbasis import onedgrid   #
from qmp.integrator.dyn_tools import *     #
from qmp.pot_tools import *                #
from qmp.visualizations import *           #
############################################


### SIMULATION CELL ### 
cell = [[0., 40.0]]

### POTENTIAL ### 
pot = Potential( cell, f=create_potential(cell, name='mexican_hat') )

### NUMBER OF BASIS STATES ### 
## for propagation in eigenbasis
states = 40

### INITIALIZE MODEL ### 
tik1d = Model(
        ndim=1,
        mass=1.0,
        mode='wave',
        basis='onedgrid',
        solver='scipy',
        integrator='SOFT',
        states=states,
        )

### SET POTENTIAL ###
tik1d.set_potential(pot)

### SET BASIS ### 
## number of grid points
N=400
b = onedgrid(cell[0][0], cell[0][1],N)
tik1d.set_basis(b)

print tik1d
print 'grid points:',N
print ''

### INITIAL WAVE FUNCTION AND DYNAMICS PARAMETERS ###
## time step, number of steps
dt =  .1
steps = 200

## initial wave functions
sigma = 1.
psi_0 = create_gaussian(tik1d.basis.x, x0=15., p0=1., sigma=sigma)

##analytical -- bogus!
#def rho_evol(x, sigma0, x0, p0, t):
#    sigma_t = sigma0*np.sqrt(1. + t**2/4./sigma0**4)
#    return (1./(np.sqrt(2*np.pi)*sigma_t))*np.exp( -(1./2.)*(x-x0-p0*t)**2 )

### EVOLVE SYSTEM ###
tik1d.run(steps,dt, psi_0=psi_0)
print 'INTEGRATED'

## gather information
psi_t = tik1d.data.wvfn.psi_t
E_t = tik1d.data.wvfn.E_t
if tik1d.parameters['integrator'] == 'SOFT':
    E_kin_t = tik1d.data.wvfn.E_kin_t
    E_pot_t = tik1d.data.wvfn.E_pot_t
else:
    E_kin_t = None
    E_pot_t = None
    
rho_t = np.sum(psi_t*np.conjugate(psi_t),1)
V_x = tik1d.pot(tik1d.basis.x)

### VISUALIZATION ###

## view animation
movie1D(tik1d.basis.x, psi_t, V_x, dt=dt)#, E_arr=E_t, abs_rho_arr=rho_t, E_kin_arr=E_kin_t, E_pot_arr=E_pot_t)

## view slideshow
#slideshow1D(tik1d.basis.x, psi_t, V_x)


