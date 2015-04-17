############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.rpmdbasis import *          #
from qmp.potential import Potential2D      #
from qmp.pot_tools import *                #
from qmp.visualizations import *           #
############################################


### SIMULATION CELL ### 
cell = [[0.,0.], [20.,20.0]]

### POTENTIAL ###
pot = Potential2D( cell, f=create_potential2D(cell, name='elbow') )


### INITIALIZE MODEL ### 
rpmd2d = Model(
         ndim=2,
         mode='rpmd',
         basis='rpmd_basis',
         integrator='rpmd_vel_verlet',
        )

### SET POTENTIAL ###
rpmd2d.set_potential(pot)

### SET INITIAL VALUES ###
rs = [[2.5,8.],[15.,3.5], [15.,4.5]]
vs = [[0.,-1.],[0.,0.],[-0.001,-0.00001]]
masses = [2., 1., 1.]
n_beads = 2
Temp = [200., 393.]

b = bead_basis(rs, vs, masses, n_beads, Temperature=Temp)
rpmd2d.set_basis(b)

print rpmd2d


### DYNAMICS PARAMETERS ###
dt =  .1
steps = 200


### EVOLVE SYSTEM ###
rpmd2d.run(steps,dt)
print 'INTEGRATED'

## gather information
r_t = rpmd2d.data.rpmd.r_t
v_t = rpmd2d.data.rpmd.v_t
#E_t = rpmd2d.data.rpmd.E_t
#E_kin = rpmd2d.data.rpmd.E_kin_t
#E_pot = rpmd2d.data.rpmd.E_pot_t

x = np.linspace(0., 20., 200)
y = np.linspace(0., 20., 200)
xg, yg = np.meshgrid(x,y)
V_xy = rpmd2d.pot(xg, yg)

### VISUALIZATION ###
import matplotlib.pyplot as plt

contour_movie2D(xg, yg, V_xy, r_t, steps+1, npar=3, trace=True)

