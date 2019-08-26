import numpy as np
from qmp.tools.visualizations import probability_distribution2D
from qmp.tools.visualizations import contour_movie2D
import qmp

# SIMULATION CELL
cell = [[0., 20.], [0., 20.]]

# DYNAMICS PARAMETERS
dt = 5.
steps = 1E3

# SET INITIAL VALUES
rs = [[15., 4.5], [15., 3.5], [2.5, 8.]]
vs = [[0., 0.], [0., 0.], [1., -2.]]
masses = [1860.] * 3
n_beads = 8
Temp = [300.] * 3

# POTENTIAL
f = qmp.potential.presets.Elbow(2, elbow_scale=0.005)

# THERMOSTAT
thermo = {
         'name':  'Andersen',
         'cfreq': 1E-4,
         'T_set': 100.,
         }

pot = qmp.potential.Potential(cell, f=f())
integrator = qmp.integrator.RPMD_VelocityVerlet(dt)
system = qmp.systems.RPMD(rs, vs, masses, n_beads, init_type='velocity')

# INITIALIZE MODEL
rpmd2d = qmp.Model(
         mode='rpmd',
         integrator=integrator,
         system=system,
         potential=pot
        )

print(rpmd2d)


# EVOLVE SYSTEM
rpmd2d.run(steps)  # , dyn_T='Rugh')

# gather information
r_t = rpmd2d.data.r_t
v_t = rpmd2d.data.v_t
E_t = rpmd2d.data.E_t
E_kin = rpmd2d.data.E_kin_t
E_pot = rpmd2d.data.E_pot_t

rb_t = rpmd2d.data.rb_t
vb_t = rpmd2d.data.vb_t
Eb_t = rpmd2d.data.Eb_t
Eb_kin = rpmd2d.data.Eb_kin_t
Eb_pot = rpmd2d.data.Eb_pot_t
bins = rpmd2d.data.prob_bins

xax = np.arange(bins[0][0], bins[0][-1]+0.1, 0.1)
yax = np.arange(bins[1][0], bins[1][-1]+0.1, 0.1)
xg, yg, = np.meshgrid(xax, yax)
V_xy = rpmd2d.potential(xg, yg)

# VISUALIZATION
probability_distribution2D(rpmd2d, show_plot=True,
                           nlines_pot=6, add_contour_labels=True)

# contour_movie2D(xg, yg, V_xy, r_t[:, 0], int(steps), npar=3, trace=False)
