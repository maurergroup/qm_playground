import qmp
import numpy as np
from qmp.tools.visualizations import wave_movie1D

# SIMULATION CELL
cell = [[-10., 15.0]]
N = 256
mass = 1800
dt = 5
steps = 5E3
output_freq = 10


# POTENTIAL
wall = qmp.potential.presets.Wall(1, position=[5.],
                                  width=np.array([2]),
                                  height=[0.001])
pot = qmp.potential.Potential(cell, f=wall())
system = qmp.systems.Grid1D(mass, cell[0][0], cell[0][1], N)

# Choose an integrator:
# integrator = qmp.integrator.SOFT_Propagator(dt)
# integrator = qmp.integrator.SOFT_Scattering(dt)
# integrator = qmp.integrator.PrimitivePropagator(dt)
# integrator = qmp.integrator.EigenPropagator(dt)

# Prepare initial wavefunction:
sigma = 1./2.
psi_0 = qmp.tools.create_gaussian(system.x, x0=0., p0=2.0, sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))
system.set_initial_wvfn(psi_0)

# INITIALIZE MODEL
wave_model_1D = qmp.Model(system=system,
                          potential=pot,
                          integrator=integrator,
                          mode='wave',
                          states=200
                          )

print(wave_model_1D)
print('Grid points:', N, '\n')


# EVOLVE SYSTEM
wave_model_1D.run(steps, output_freq=10)

# GATHER INFO
psi_t = wave_model_1D.data.psi_t
E_t = wave_model_1D.data.E_t
rho_t = wave_model_1D.data.rho_t
norm_t = np.sum(rho_t, 1)

V_x = wave_model_1D.potential(wave_model_1D.system.x)

# view animation
wave_movie1D(wave_model_1D.system.x, rho_t[:, :N], V_x, dt=dt, E_arr=E_t,
             rho_tot_arr=norm_t)
