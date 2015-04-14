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

pot = Potential2D( cell, f=f_harm )


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
rs = [[10., 14.], [8., 7.]]
vs = [[3.,0.], [0.5, 2.]]
masses = [1., 2.]

b = phasespace(rs, vs, masses)
traj2d.set_basis(b)

print traj2d


### DYNAMICS PARAMETERS ###
dt =  .2
steps = 400


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

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
theta = 25.

#generate figure
fig = plt.figure()
plt.subplots_adjust(bottom=0.2)
ax = fig.gca(projection='3d')

frame = None
for i in xrange(steps+1):
    ax.clear()
    ax.plot_surface(xg, yg, V_xy, alpha=0.75, antialiased=False, cmap = cm.coolwarm, lw=0.)
    ax.scatter(r_t[i,0,0], r_t[i,0,1], traj2d.pot(r_t[i,0,0], r_t[i,0,1])+0.1, marker='o', s=20., c='k')
    ax.scatter(r_t[i,1,0], r_t[i,1,1], traj2d.pot(r_t[i,1,0], r_t[i,1,1])+0.1, marker='o', s=20., c='k')
        
    plt.pause(0.0005)
    
    
    