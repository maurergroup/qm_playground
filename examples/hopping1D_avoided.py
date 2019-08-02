from qmp.potential import tullymodels
from qmp.model import Model
from qmp.systems.hopping import Hopping
from qmp.integrator.hoppingintegrators import HoppingIntegrator
import numpy as np


x = np.array([-5.])
max_steps = 1e5
dt = 20.0
mass = np.array([2000.])
initial_state = 0

pot = tullymodels.TullySimpleAvoidedCrossing()
integ = HoppingIntegrator()

momenta = np.linspace(3, 30, 20)
velocities = momenta / mass

results = np.zeros((len(momenta), 5))

for i, vel in enumerate(velocities):
    system = Hopping(x, [vel], mass, initial_state, pot)
    surfhop = Model(potential=pot, mode='hop', system=system, integrator=integ)
    surfhop.run(max_steps, dt=dt, ntraj=200)
    dat = surfhop.data
    results[i, 0] = momenta[i]
    results[i, 1] = dat.reflect_lower
    results[i, 2] = dat.reflect_upper
    results[i, 3] = dat.transmit_lower
    results[i, 4] = dat.transmit_upper

np.savetxt('avoided_crossing.txt', results)
