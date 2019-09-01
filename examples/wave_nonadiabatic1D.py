import numpy as np
import qmp

# Set simulation parameters:
cell = np.array([[-15., 15.0]])
N = 256
mass = 2000
dt = 1
steps = 5E4
system = qmp.systems.Grid(mass, cell, N, states=2)

# Choose potential:
pot = qmp.potential.tullymodels.TullySimpleAvoidedCrossing(cell=cell)
# pot = qmp.potential.tullymodels.TullyExtendedCoupling(cell=cell)


# Choose integrator:
# integrator = qmp.integrator.SOFT_Propagator(dt)
integrator = qmp.integrator.PrimitivePropagator(dt)

# Create and set the initial wavefunction:
sigma = 1./2.
psi_0 = qmp.tools.create_gaussian(system.mesh[0], x0=-8, p0=5.0, sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))
system.set_initial_wvfn(psi_0)

# Create the model:
tik1d = qmp.Model(
        system=system,
        potential=pot,
        integrator=integrator,
        mode='wave'
        )

# Print model information:
print(tik1d)
print('Grid points:', N, '\n')

# Run the simulation:
tik1d.run(steps, output_freq=50)

# Print the scattering results:
print('Reflect N=1', 'Transmit N=1', 'Reflect N=2', 'Transmit N=2')
print(tik1d.data.outcome)
