import qmp

# SIMULATION CELL
cell = [[0., 20.], [0., 20.]]

# DYNAMICS PARAMETERS
dt = 5.
steps = 1E3

# SET INITIAL VALUES
rs = [[5., 5], [15., 3.5], [2.5, 8.]]
vs = [[0.01, 0.], [0., 0.], [0.001, -0.002]]
masses = [1860.] * 3
n_beads = 8

# POTENTIAL
f = qmp.potential.presets.Elbow(2, elbow_scale=0.005)
pot = qmp.potential.Potential(cell, f=f())
integrator = qmp.integrator.RPMD_VelocityVerlet(dt)
system = qmp.systems.RPMD(rs, vs, masses, n_beads, init_type='velocity', T=20)

# INITIALIZE MODEL
rpmd2d = qmp.Model(
         mode='rpmd',
         integrator=integrator,
         system=system,
         potential=pot
        )

print(rpmd2d)

# EVOLVE SYSTEM
rpmd2d.run(steps, output_freq=1)
