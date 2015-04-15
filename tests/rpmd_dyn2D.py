############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.rpmdbasis import *          #
from qmp.potential import Potential2D      #
from qmp.pot_tools import *                #
############################################


### SIMULATION CELL ### 
cell = [[0.,0.], [20.,20.0]]

### POTENTIAL ###
## 2D harmonic potential
def f_harm(x,y):
    omx, omy = .5, .5
    return omx*((x-10.)**2) + omy*((y-10.)**2)

pot = Potential2D( cell, f=f_harm )


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
rs = [[13.,13.],[8.,10.]]
vs = [[-2.,1.],[0.,5.]]
masses = [1., 2.]
n_beads = 5
Temp = [200., 300.]

b = bead_basis(rs, vs, masses, n_beads, Temperature=Temp)
rpmd2d.set_basis(b)

print rpmd2d


### DYNAMICS PARAMETERS ###
dt =  .1
steps = 50


### EVOLVE SYSTEM ###
rpmd2d.run(steps,dt)
print 'INTEGRATED'

## gather information
r_t = rpmd2d.data.rpmd.r_t
v_t = rpmd2d.data.rpmd.v_t
E_t = rpmd2d.data.rpmd.E_t
E_kin = rpmd2d.data.rpmd.E_kin_t
E_pot = rpmd2d.data.rpmd.E_pot_t

V_t1, V_t2 = [], []
for i in xrange(steps+1):
         V_t1.append(rpmd2d.pot(r_t[0,i,0],r_t[0,i,1]))
         V_t2.append(rpmd2d.pot(r_t[1,i,0],r_t[1,i,1]))

### VISUALIZATION ###
x = np.linspace(0.,20.,600)
y = np.linspace(0.,20.,600)
xv,yv = np.meshgrid(x,y)
V_xy = rpmd2d.pot(xv,yv)

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
theta = 25.

plt.plot(E_t[0].T)
plt.plot(E_t[1].T)
plt.show()

#generate figure
fig = plt.figure()
plt.subplots_adjust(bottom=0.2)
ax = fig.gca(projection='3d')

frame = None
for i in xrange(steps+1):
    ax.clear()
    ax.plot_surface(xv, yv, V_xy, alpha=0.7, antialiased=False, cmap = cm.coolwarm, lw=0.,zorder=3)
    ax.scatter(r_t[0,i,0], r_t[0,i,1], V_t1[i]+0.2, marker='o', s=20., c='k',zorder=1)
    ax.scatter(r_t[1,i,0], r_t[1,i,1], V_t2[i]+0.2, marker='o', s=20., c='r',zorder=2)
        
    plt.pause(0.0005)
    
    
    