import qmp
import numpy as np
from qmp.tools.visualizations import wave_movie2D

# SIMULATION CELL
cell = np.array([[-3., 3.0], [-3, 3]])
N = 50
mass = 1800
dt = 1
steps = 1e3
output_freq = 10


# POTENTIAL
# wall = qmp.potential.presets.Wall(1, position=[5.],
#                                   width=np.array([2]),
#                                   height=[0.001])
free = qmp.potential.presets.Free(2)

pot = qmp.potential.Potential(cell, f=free())
system = qmp.systems.Grid(mass, cell, N)

# Choose an integrator:
integrator = qmp.integrator.SOFT_Propagator(dt)
# integrator = qmp.integrator.SOFT_Scattering(dt)
# integrator = qmp.integrator.PrimitivePropagator(dt)
# integrator = qmp.integrator.EigenPropagator(dt)

# Prepare initial wavefunction:
sigma = 1./2.
psi_0 = qmp.tools.create_gaussian2D(*system.mesh, x0=[0., 0], p0=[0, 0],
                                    sigma=[sigma, sigma])
# psi_0 = qmp.tools.create_gaussian(system.x, x0=0., p0=5., sigma=sigma)
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

# GATHER INFO
psi_t = np.real(wave_model_1D.data.psi_t)
# rho_t = wave_model_1D.data.rho_t
# norm_t = np.sum(rho_t, 1)

# print(rho_t.shape)
# print(psi_t.shape)
# print(E_t.shape)
# V_x = wave_model_1D.potential(wave_model_1D.system.x)

# view animation
# print(wave_model_1D.data.outcome
# wave_movie2D(*wave_model_1D.system.mesh, psi_t)
