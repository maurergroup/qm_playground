import qmp
import numpy as np

# Set simulation parameters:
cell = np.array([[-15., 15.0]])
N = 512
mass = 1800
dt = 1
steps = 1.5e4
output_freq = 150
system = qmp.systems.Grid(mass, cell, N)


# Choose potential:
wall = qmp.potential.presets.Wall(1, position=[0.],
                                  width=np.array([2]),
                                  height=[0.001])
# free = qmp.potential.presets.Free(1)
pot = qmp.potential.Potential(cell, f=wall())

# Choose an integrator:
integrator = qmp.integrator.SOFT_Propagator(dt)
# integrator = qmp.integrator.PrimitivePropagator(dt)

# Prepare initial wavefunction:
sigma = 1./2.
psi_0 = qmp.tools.create_gaussian(*system.mesh, x0=-8., p0=2., sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))
system.set_initial_wvfn(psi_0)

# Create the model:
wave_model_1D = qmp.Model(system=system,
                          potential=pot,
                          integrator=integrator,
                          mode='wave',
                          states=N
                          )

# Print model information:
print(wave_model_1D)
print('Grid points:', N, '\n')

# Run the simulation:
wave_model_1D.run(steps, output_freq=output_freq)
print()
print('Results:')
print('Reflect N=1', 'Transmit N=1')
print(wave_model_1D.data.outcome)
