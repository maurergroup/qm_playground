############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.phasespace_basis import *   #
from qmp.potential.pot_tools import *      #
from qmp.tools.visualizations import *     #
from qmp.potential import Potential2D      #
from qmp.tools.utilities import *          #
############################################


### SIMULATION CELL ### 
cell = [[0.,20.], [0.,20.]]

### POTENTIAL ###
pot = Potential2D( cell, f=create_potential2D(cell, name='elbow') )


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
rs = [[15.,4.5],[15.,3.5], [2.5,8.]]
vs = [[-.001,-0.00001],[0.,0.],[1.,-2.]]
masses = [1., 1., 1.]

b = phasespace(rs, vs, masses)
traj2d.set_basis(b)

print traj2d

### DYNAMICS PARAMETERS ###
dt =  .1
steps = 300


### EVOLVE SYSTEM ###
traj2d.run(steps,dt)

## gather information
r_t = traj2d.data.traj.r_t
v_t = traj2d.data.traj.v_t
E_t = traj2d.data.traj.E_t
E_kin = traj2d.data.traj.E_kin_t
E_pot = traj2d.data.traj.E_pot_t

x = np.linspace(0., 20., 500)
y = np.linspace(0., 20., 500)
xg, yg = np.meshgrid(x,y)
V_xy = traj2d.pot(xg, yg)

### VISUALIZATION ###

contour_movie2D(xg, yg, V_xy, r_t, steps+1, npar=3, trace=True)

