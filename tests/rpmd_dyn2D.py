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
rpmd2d = Model(
         ndim=2,
         mode='rpmd',
         basis='rpmd_basis',
         integrator='rpmd_vel_verlet',
        )

### SET POTENTIAL ###
rpmd2d.set_potential(pot)

### SET INITIAL VALUES ###
rs = [[13.,13.],[9.,9.]]
vs = [[-2.,1.],[.4,.45]]
masses = [1., 1.]
n_beads = 5
Temp = [200., 393.]

b = bead_basis(rs, vs, masses, n_beads, Temperature=Temp)
rpmd2d.set_basis(b)

print rpmd2d


### DYNAMICS PARAMETERS ###
dt =  .1
steps = 100


### EVOLVE SYSTEM ###
rpmd2d.run(steps,dt)
print 'INTEGRATED'

## gather information
r_t = rpmd2d.data.rpmd.r_t
v_t = rpmd2d.data.rpmd.v_t
E_t = rpmd2d.data.rpmd.E_t
E_kin = rpmd2d.data.rpmd.E_kin_t
E_pot = rpmd2d.data.rpmd.E_pot_t

x = np.linspace(0., 20., 200)
y = np.linspace(0., 20., 200)
xg, yg = np.meshgrid(x,y)
V_xy = rpmd2d.pot(xg, yg)

### VISUALIZATION ###
import matplotlib.pyplot as plt

plt.plot(E_t[0])
plt.plot(E_t[1])
plt.show()


contour_movie2D(xg, yg, V_xy, r_t[1], steps+1, trace=True)

