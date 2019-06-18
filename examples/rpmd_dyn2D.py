############### IMPORT STUFF ###############
import numpy as np                         #
import sys                                 #
sys.path.append('..')                      #
from qmp import *                          #
from qmp.basis.rpmdbasis import *          #
from qmp.potential import Potential        #
from qmp.potential.preset_potentials import Elbow
from qmp.potential.pot_tools import *      #
from qmp.tools.visualizations import *     #
############################################

import scipy.linalg as la

### SIMULATION CELL ###
cell = [[0.,20.], [0.,20.]]

### POTENTIAL ###
f = Elbow(2, elbow_scale=0.005)

pot = Potential(cell, f=f())


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
masses = [1860.]#, 1., 1.]
n_beads = 8
Temp = [300.]#, 393., 1000.]

b = bead_basis(rs, vs, masses, n_beads, T=Temp)
rpmd2d.set_basis(b)

print(rpmd2d)

### DYNAMICS PARAMETERS ###
dt =  2.
steps = 1E4

### THERMOSTAT ###
thermo = {
         'name':  'Andersen',
         'cfreq': 1E-4,
         'T_set': 100.,
         }

### EVOLVE SYSTEM ###
rpmd2d.run(steps,dt,thermostat=thermo)

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
bins = rpmd2d.data.rpmd.prop_bins

xax = np.arange(bins[0][0], bins[0][-1]+0.1, 0.1)
yax = np.arange(bins[1][0], bins[1][-1]+0.1, 0.1)
xg, yg, = np.meshgrid(xax, yax)
V_xy = rpmd2d.pot(xg, yg)

### VISUALIZATION ###
propability_distribution2D(rpmd2d, show_plot=True, nlines_pot=6, add_contour_labels=True)

contour_movie2D(xg, yg, V_xy, r_t, steps+1, npar=1, trace=True)
