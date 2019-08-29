import numpy as np
from qmp.tools.visualizations import wave_movie1D
import qmp

# SIMULATION CELL
cell = [[-8., 8.0]]
N = 256
mass = 2000
dt = 0.1
steps = 1E4

# POTENTIAL
pot = qmp.potential.tullymodels.TullySimpleAvoidedCrossing()
# pot = qmp.potential.tullymodels.TullyExtendedCoupling()
integrator = qmp.integrator.SOFT_Propagator(dt)
# integrator = qmp.integrator.SOFT_Scattering(dt)
# integrator = qmp.integrator.PrimitivePropagator(dt)

system = qmp.systems.Grid1D(mass, cell[0][0], cell[0][1], N, states=2)

# initial wave functions
sigma = 1./2.
psi_0 = qmp.tools.create_gaussian(system.x, x0=-5, p0=-10.0, sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))

system.set_initial_wvfn(psi_0)

# INITIALIZE MODEL
tik1d = qmp.Model(
        system=system,
        potential=pot,
        integrator=integrator,
        mode='wave'
        )

print(tik1d)
print('Grid points:', N, '\n')

# EVOLVE SYSTEM
tik1d.run(steps, output_freq=200)

# print(tik1d.data.probs)
# GATHER INFO
# psi_t = tik1d.data.psi_t
# E_t = tik1d.data.E_t
# rho_t = tik1d.data.rho_t
# norm_t = np.sum(rho_t, 1)
# V_x = tik1d.potential(tik1d.system.x)

# wave_movie1D(tik1d.system.x, rho_t[:, :N], V_x, dt=dt, E_arr=E_t,
#              rho_tot_arr=norm_t)
# wave_movie1D(tik1d.system.x, rho_t[:, N:], V_x, dt=dt)
