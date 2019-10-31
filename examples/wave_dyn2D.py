import qmp
import numpy as np
import matplotlib.pyplot as plt

# SIMULATION CELL
cell = np.array([[-12, 18], [-8, 8]])
N = 64
mass = 1800
dt = 1
steps = 2e2
output_freq = 10


# POTENTIAL
doubleslit = qmp.potential.presets.DoubleSlit(2)

pot = qmp.potential.Potential(cell, f=doubleslit())
system = qmp.systems.Grid(mass, cell, N)

plt.contour(*system.mesh, pot(*system.mesh))
plt.show()

# Choose an integrator:
integrator = qmp.integrator.SOFT_Propagator(dt)

# Prepare initial wavefunction:
sigma = 2
psi_0 = qmp.tools.create_gaussian2D(*system.mesh, x0=[-5., 0], p0=[5, 0],
                                    sigma=[sigma, sigma])
psi_0 /= np.sqrt(np.conjugate(psi_0.flatten()).dot(psi_0.flatten()))
system.set_initial_wvfn(psi_0)

# INITIALIZE MODEL
wave_model_1D = qmp.Model(system=system,
                          potential=pot,
                          integrator=integrator,
                          mode='wave',
                          states=N
                          )

print(wave_model_1D)
print('Grid points:', N, '\n')

# EVOLVE SYSTEM
wave_model_1D.run(steps, output_freq=1)
