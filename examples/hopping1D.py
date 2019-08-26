import qmp
import numpy as np

x = np.array([-10.])
max_steps = 1e5
ntraj = 20
dt = 20.0
mass = np.array([2000.])
initial_state = 0

pot = qmp.potential.tullymodels.TullySimpleAvoidedCrossing()
integ = qmp.integrator.HoppingIntegrator()

momenta = np.linspace(1, 32, 5)
velocities = momenta / mass

results = np.zeros((len(momenta), 5))

for i, vel in enumerate(velocities):
    system = qmp.systems.Hopping(x, [vel], mass, initial_state, pot)
    surfhop = qmp.Model(potential=pot, mode='hop',
                        system=system, integrator=integ)
    surfhop.run(max_steps, dt=dt, ntraj=ntraj)
    dat = surfhop.data
    results[i, 0] = momenta[i]
    results[i, 1] = dat.reflect_lower
    results[i, 2] = dat.reflect_upper
    results[i, 3] = dat.transmit_lower
    results[i, 4] = dat.transmit_upper

np.savetxt('avoided_crossing.txt', results)
