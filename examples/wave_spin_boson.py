import numpy as np
import qmp
import matplotlib.pyplot as plt

# Set simulation parameters:
cell = np.array([[-10, 10]])
N = 1024
mass = 1
dt = 0.03
steps = 1000
system = qmp.systems.Grid(mass, cell, N, states=2)

# Choose potential:
# pot = qmp.potential.tullymodels.TullySimpleAvoidedCrossing(cell=cell)
# pot = qmp.potential.tullymodels.TullyDualAvoidedCrossing(cell=cell)
pot = qmp.potential.spin_boson.SpinBoson(cell=cell, gamma=0.1)
# pot = qmp.potential.tullymodels.TullyExtendedCoupling(cell=cell)

# Choose integrator:
integrator = qmp.integrator.SOFT_Propagator(dt, absorb=False,
                                            output_adiabatic=False)
# integrator = qmp.integrator.PrimitivePropagator(dt)
# Create and set the initial wavefunction:
x = 0
p = 0
sigma = 1
psi_0 = qmp.tools.create_gaussian(system.mesh[0], x0=x, p0=p, sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))
system.set_initial_wvfn(psi_0, n=2)

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
wave_model.run(steps, output_freq=1)

# Print the scattering results:
print('Reflection', 'Transmission')
print(wave_model.data.outcome)
wave_model.write_output()

rho_t = wave_model.data.rho_t
res = np.zeros(steps+1)
for i in range(steps+1):
    split = np.array(np.split(rho_t[i], 2))
    summed = np.sum(split, axis=1)
    summed[0] *= -1
    res[i] = np.sum(summed)

plt.plot(np.arange(steps+1)*dt, res)
plt.show()
