import qmp

# SIMULATION CELL
cell = [[-0.5, 0.5]]

rs = [[-0.], [-0.2]]
vs = [[0.01], [0]]
masses = [2000., 2000.]
dt = 5
steps = 3000
n_beads = 8

# POTENTIAL
harm = qmp.potential.presets.Harmonic(1)
pot = qmp.potential.Potential(cell, f=harm())
integ = qmp.integrator.RPMD_VelocityVerlet(dt)
sys = qmp.systems.RPMD(rs, vs, masses, n_beads, T=10,
                       init_type='velocity')

# INITIALIZE MODEL
traj2d = qmp.Model(
         system=sys,
         potential=pot,
         integrator=integ,
         mode='rpmd',
        )

print(traj2d)

# EVOLVE SYSTEM
traj2d.run(steps, output_freq=1)
