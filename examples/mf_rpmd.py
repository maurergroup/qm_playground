import numpy as np

import qmp

# Set simulation parameters:
cell = np.array([[-5, 5]])
mass = np.array([1])
dt = 0.1
steps = 1e3
x = np.array([0])
v = np.array([0])
initial_state = 1
n_beads = 16

T = 1/16 * qmp.tools.dyn_tools.atomic_to_kelvin

# Choose potential:
potential = qmp.potential.spin_boson.SpinBoson(cell=cell, gamma=0.1)

# Choose integrator:
integrator = qmp.integrator.MF_RPMD_Propagator(dt=dt)

system = qmp.systems.MF_RPMD(x, v, mass, start_file='nvt.traj',
                             equilibration_end=100,
                             n_beads=n_beads, T=T)

# Create the model:
wave_model = qmp.Model(
                system=system,
                potential=potential,
                integrator=integrator,
                mode='nrpmd'
                )

# Run the simulation:
wave_model.run(steps, output_freq=1)

wave_model.write_output()
