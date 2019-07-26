import numpy as np
from qmp.systems.phasespace import PhaseSpace
from qmp.potential import Potential
from qmp import Model
from qmp.tools.visualizations import contour_movie2D
from qmp.potential.preset_potentials import Elbow
from qmp.integrator.trajintegrators import VelocityVerlet


# SIMULATION CELL
cell = [[0., 20.], [0., 20.]]

rs = [[15., 4.5], [15., 3.5], [2.5, 8.]]
vs = [[-.001, -0.00001], [0., 0.], [1., -2.]]
masses = [1., 1., 1.]
dt = 0.1

# POTENTIAL
elbow = Elbow(2)
pot = Potential(cell, f=elbow())
integ = VelocityVerlet(dt)
sys = PhaseSpace(rs, vs, masses)

# INITIALIZE MODEL
traj2d = Model(
         system=sys,
         potential=pot,
         integrator=integ,
         mode='traj',
        )

print(traj2d)

# DYNAMICS PARAMETERS
steps = 300

# EVOLVE SYSTEM
traj2d.run(steps)

# gather information
r_t = traj2d.data.r_t
v_t = traj2d.data.v_t
E_t = traj2d.data.E_t
E_kin = traj2d.data.E_kin_t
E_pot = traj2d.data.E_pot_t

x = np.linspace(0., 20., 500)
y = np.linspace(0., 20., 500)
xg, yg = np.meshgrid(x, y)
V_xy = traj2d.potential(xg, yg)

# VISUALIZATION
contour_movie2D(xg, yg, V_xy, r_t, steps+1, npar=3, trace=True)
