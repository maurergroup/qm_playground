import qmp
import numpy as np
from qmp.tools.visualizations import contour_movie2D

# SIMULATION CELL
cell = [[-5., 5.]]

rs = [[4.], [-1.]]
vs = [[0.], [0.]]
masses = [1., 1.]
dt = 0.05

# POTENTIAL
harm = qmp.potential.presets.Harmonic(1)
pot = qmp.potential.Potential(cell, f=harm())
integ = qmp.integrator.VelocityVerlet(dt)
sys = qmp.systems.PhaseSpace(rs, vs, masses)

# INITIALIZE MODEL
traj2d = qmp.Model(
         system=sys,
         potential=pot,
         integrator=integ,
         mode='traj',
        )

print(traj2d)

# DYNAMICS PARAMETERS
steps = 3000

# EVOLVE SYSTEM
traj2d.run(steps)

# gather information
r_t = traj2d.data.r_t
v_t = traj2d.data.v_t
E_t = traj2d.data.E_t
E_kin = traj2d.data.E_kin_t
E_pot = traj2d.data.E_pot_t
print(r_t)

# x = np.linspace(0., 20., 500)
# y = np.linspace(0., 20., 500)
# xg, yg = np.meshgrid(x, y)
# V_xy = traj2d.potential(xg, yg)

# VISUALIZATION
# contour_movie2D(xg, yg, V_xy, r_t, steps+1, npar=3, trace=True)
