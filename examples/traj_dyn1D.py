import qmp

# SIMULATION CELL
cell = [[-5., 5.]]

rs = [[4.], [-1.]]
vs = [[0.], [0.]]
masses = [1., 1.]
dt = 0.05
steps = 3000

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

# EVOLVE SYSTEM
traj2d.run(steps)
