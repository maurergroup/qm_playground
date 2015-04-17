############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.phasespace_basis import *   #
from qmp.pot_tools import *                #
from qmp.visualizations import *           #
from qmp.potential import Potential2D      #
############################################


### SIMULATION CELL ### 
cell = [[0., 20.0]]

### POTENTIAL ###
## 2D harmonic potential
def f_harm(x,y):
    omx, omy = .5, .5
    return omx*((x-10.)**2) + omy*((y-10.)**2)

## 2D "mexican hat potential"
def f_mexican(x,y):
    sigma = 1.
    pref = 20./(np.pi*sigma**4)
    brak = 1.-(((x-10.)**2+(y-10.)**2)/(2*sigma**2))
    f = pref*brak*np.exp(-(((x-10.)**2+(y-10)**2)/(2.*sigma**2)))
    return f - min(f.flatten())

pot = Potential2D( cell, f=f_mexican )


### INITIALIZE MODEL ### 
traj2d = Model(
         ndim=2,
         mode='traj',
         basis='phasespace',
         integrator='vel_verlet',
        )

### SET POTENTIAL ###
traj2d.set_potential(pot)

### SET INITIAL VALUES ###
rs = [[13.,13.],[9.,9.]]
vs = [[-2.,1.],[.4,.45]]
masses = [1., 1.]

b = phasespace(rs, vs, masses)
traj2d.set_basis(b)

print traj2d


### DYNAMICS PARAMETERS ###
dt =  .1
steps = 100


### EVOLVE SYSTEM ###
traj2d.run(steps,dt)
print 'INTEGRATED'

## gather information
r_t = traj2d.data.traj.r_t
v_t = traj2d.data.traj.v_t
E_t = traj2d.data.traj.E_t
E_kin = traj2d.data.traj.E_kin_t
E_pot = traj2d.data.traj.E_pot_t

x = np.linspace(0., 20., 200)
y = np.linspace(0., 20., 200)
xg, yg = np.meshgrid(x,y)
V_xy = traj2d.pot(xg, yg)

### VISUALIZATION ###

contour_movie2D(xg, yg, V_xy, r_t[:,1,:], steps+1, trace=True)

