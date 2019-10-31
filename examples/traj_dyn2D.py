import qmp

# SIMULATION CELL
cell = [[0., 20.], [0., 20.]]

rs = [[15., 4.5], [15., 3.5], [2.5, 8.]]
vs = [[-.001, -0.00001], [0., 0.], [1., -2.]]
masses = [1., 1., 1.]
dt = 0.05

# POTENTIAL
elbow = qmp.potential.presets.Elbow(2)
pot = qmp.potential.Potential(cell, f=elbow())
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
