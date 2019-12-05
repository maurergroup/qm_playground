import numpy as np

from .integrator import Integrator, SimulationTerminated


class HoppingIntegrator:

    def __init__(self, dt=1):
        """Class is initialised with a timestep as a single argument."""
        self.dt = dt

    def run(self, system, steps, potential, data, **kwargs):

        self.system = system
        self._read_kwargs(kwargs)

        self._initialise_start(data, steps)

        for i in range(self.ntraj):
            self.traj.run(system, steps, potential, data, **kwargs)

        self._assign_data(data)

        print(f'{self.ntraj} successful trajectories completed.')

    def _read_kwargs(self, kwargs):
        self.dt = kwargs.get('dt', self.dt)
        self.ntraj = kwargs.get('ntraj', 2000)

    def _initialise_start(self, data, steps):
        data.state_t = np.zeros(steps+1)
        data.outcome = np.zeros((self.system.nstates, 2))
        self.traj = SingleTrajectoryHopper(self.dt)
        momentum = self.system.initial_v * self.system.masses
        print(f'Running {self.ntraj} surface hopping trajectories'
              + f' for momentum = {momentum}')

    def _assign_data(self, data):
        data.outcome = data.outcome / self.ntraj
        data.state_t = data.state_t / self.ntraj


class SingleTrajectoryHopper(Integrator):
    """Single trajectory surface hopping.

    This class must be used in conjunction with a managing class, otherwise the
    outcome attribute of the data object has not been initilised.
    """

    def _initialise_start(self):
        """Reset system and prepare logging variables."""
        self.r_t = [self.system.r]
        self.v_t = [self.system.v]
        self.state_t = [self.system.initial_state]

        self.system.reset_system(self.potential)
        self.outcome = np.zeros((self.system.nstates, 2))
        self.current_acc = self.system.compute_acceleration(self.potential)

    def _perform_timestep(self, iteration):
        self._propagate_system()

        self.system.propagate_density_matrix(self.dt)

        self._execute_hopping()

        if (iteration+1) % self.output_freq == 0:
            self._store_result()

        if self._is_finished():
            raise SimulationTerminated


    def _is_finished(self):
        """ Checks criteria for a successful trajectory.

        Check if the particle has left the cell, if so, the outcome is
        recorded.
        """
        exit = False
        if self.system.has_reflected():
            self.outcome[self.system.current_state, 0] = 1
            exit = True
        elif self.system.has_transmitted():
            self.outcome[self.system.current_state, 1] = 1
            exit = True
        return exit

    def _execute_hopping(self):
        """Carry out the hop between surfaces."""

        g = self.system.get_probabilities(self.dt)
        can_hop = self._check_possible_hop(g)

        if can_hop:
            self.system.attempt_hop()

    def _check_possible_hop(self, g):
        """Determine whether a hop should occur.

        A random number is compared with the hopping probability, if the
        probability is higher than the random number, a hop occurs. This
        currently only works for a two level system.

        Parameters
        ----------
        g : float
            The hopping probability.
        """
        zeta = np.random.uniform()
        return g > zeta

    def _assign_data(self, data):
        """Add to the cumulative outcome."""
        data.outcome += self.outcome
        for i in range(len(self.state_t)):
            data.state_t[i] += self.state_t[i]

    def _store_result(self):
        """Store trajectory data but this is currently not used."""
        self.r_t.append(self.system.r)
        self.v_t.append(self.system.v)
        self.state_t.append(self.system.current_state)


class RingHoppingIntegrator(HoppingIntegrator):

    def _initialise_start(self, data, steps):
        super()._initialise_start(data, steps)
        self.system.initialise_propagators(self.dt)
