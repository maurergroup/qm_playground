############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.rpmdbasis import *          #
from qmp.potential import Potential2D      #
from qmp.pot_tools import *                #
from qmp.tools.visualizations import *     #
############################################

import scipy.linalg as la

### SIMULATION CELL ### 
cell = [[0.,0.], [20.,20.0]]

### POTENTIAL ###
f=create_potential2D(cell,
                     name='elbow',
                     elbow_scale=0.005,
                     #harmonic_omega_x=1.,
                     #harmonic_omega_y=2.,
                    )

pot = Potential2D( cell, f=f )


### INITIALIZE MODEL ### 
rpmd2d = Model(
         ndim=1,
         mode='rpmd',
         basis='rpmd_basis',
         integrator='RPMD_VelocityVerlet',
        )

### SET POTENTIAL ###
rpmd2d.set_potential(pot)

### SET INITIAL VALUES ###
rs = [[15.,4.5]]#,[15.,3.5], [2.5,8.]]
vs = [[0.,0.]]#,[0.,0.],[1.,-2.]]
masses = [200.]#, 1., 1.]
n_beads = 4
Temp = [300.]#, 393., 1000.]

b = bead_basis(rs, vs, masses, n_beads, T=Temp)
rpmd2d.set_basis(b)

print rpmd2d

### DYNAMICS PARAMETERS ###
dt =  2.
steps = 1E3

### THERMOSTAT ###
thermo = {
         'name':  'Andersen',
         'cfreq': 1E-4,
         'T_set': 10000.,
         }

### EVOLVE SYSTEM ###
rpmd2d.run(steps,dt,thermostat=thermo)
print 'INTEGRATED'

## gather information
r_t = rpmd2d.data.rpmd.r_t
v_t = rpmd2d.data.rpmd.v_t
E_t = rpmd2d.data.rpmd.E_t
E_kin = rpmd2d.data.rpmd.E_kin_t
E_pot = rpmd2d.data.rpmd.E_pot_t

rb_t = rpmd2d.data.rpmd.rb_t
vb_t = rpmd2d.data.rpmd.vb_t
Eb_t = rpmd2d.data.rpmd.Eb_t
Eb_kin = rpmd2d.data.rpmd.Eb_kin_t
Eb_pot = rpmd2d.data.rpmd.Eb_pot_t

x = np.linspace(0., 20., 200)
y = np.linspace(0., 20., 200)
xg, yg = np.meshgrid(x,y)
V_xy = rpmd2d.pot(xg, yg)

### VISUALIZATION ###

contour_movie2D(xg, yg, V_xy, r_t, steps+1, npar=1, trace=True)

