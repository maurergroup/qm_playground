import numpy as np
from qmp.potential import tullymodels
from qmp.integrator.dyn_tools import create_gaussian
from qmp.tools.visualizations import *
from qmp.integrator.waveintegrators import SOFT_Propagator
from qmp.integrator.waveintegrators import PrimitivePropagator
from qmp.systems.grid import Grid1D
from qmp import Model


# SIMULATION CELL
cell = [[-15., 15.0]]
N = 256
mass = 2000
dt = 0.05


# POTENTIAL
pot = tullymodels.TullySimpleAvoidedCrossing()
# integrator = SOFT_Propagator(dt)
integrator = PrimitivePropagator(dt)
system = Grid1D(mass, cell[0][0], cell[0][1], N, states=2)

# initial wave functions
sigma = 1./2.
psi_0 = create_gaussian(system.x, x0=-5, p0=10.0, sigma=sigma)
psi_0 /= np.sqrt(np.conjugate(psi_0).dot(psi_0))

system.set_initial_wvfn(psi_0)

# INITIALIZE MODEL
tik1d = Model(
        system=system,
        potential=pot,
        integrator=integrator,
        mode='wave'
        )

print(tik1d)
print('Grid points:', N, '\n')

# INITIAL WAVE FUNCTION AND DYNAMICS PARAMETERS
# time step, number of steps
steps = 5E4

# EVOLVE SYSTEM
tik1d.run(steps, output_freq=1000)

# GATHER INFO
# info time evolution
psi_t = tik1d.data.psi_t
E_t = tik1d.data.E_t

# E_kin_t = tik1d.data.E_kin_t
# E_pot_t = tik1d.data.E_pot_t
rho_t = tik1d.data.rho_t
# rho_mean = tik1d.data.rho_mean

norm_t = np.sum(rho_t, 1)

# rho_mean = np.mean(rho_t, 0)
# r_mean = np.dot(tik1d.system.x, rho_mean)

V_x = tik1d.potential(tik1d.system.x)

wave_movie1D(tik1d.system.x, rho_t[:, :N], V_x, dt=dt, E_arr=E_t,
        rho_tot_arr=norm_t)
             # E_kin_arr=E_kin_t, E_pot_arr=E_pot_t)
wave_movie1D(tik1d.system.x, rho_t[:, N:], V_x, dt=dt)
