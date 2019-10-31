import numpy as np
import qmp

# Set simulation parameters:
cell = np.array([[-9., 9]])
N = 1024
mass = 2000
dt = 1
steps = 5E4
system = qmp.systems.Grid(mass, cell, N, states=2)

# Choose potential:
# pot = qmp.potential.tullymodels.TullySimpleAvoidedCrossing(cell=cell)
pot = qmp.potential.tullymodels.TullyDualAvoidedCrossing(cell=cell)
# pot = qmp.potential.tullymodels.TullyExtendedCoupling(cell=cell)

# Choose integrator:
integrator = qmp.integrator.SOFT_Propagator(dt)
# integrator = qmp.integrator.PrimitivePropagator(dt)

# Create and set the initial wavefunction:
x = -6
p = 30
sigma = 20 / p
psi_0 = qmp.tools.create_gaussian(system.mesh[0], x0=x, p0=p, sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))
system.set_initial_wvfn(psi_0)

# Create the model:
wave_model = qmp.Model(
                system=system,
                potential=pot,
                integrator=integrator,
                mode='wave'
                )

# Print model information:
print(wave_model)
print('Grid points:', N, '\n')

# Run the simulation:
wave_model.run(steps, output_freq=8)

# Print the scattering results:
print('Reflection', 'Transmission')
print(wave_model.data.outcome)
