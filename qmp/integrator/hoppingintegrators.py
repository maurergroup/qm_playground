import numpy as np
from .trajintegrators import AbstractVelocityVerlet


class HoppingIntegrator:

    def __init__(self, dt=1):
        """Class is initialised with a timestep as a single argument."""
        self.dt = dt

    def run(self, system, steps, potential, data, **kwargs):

        dt = kwargs.get('dt', self.dt)
        self.ntraj = kwargs.get('ntraj', 2000)

        self.system = system

        data.outcome = np.zeros((self.system.nstates, 2))

        momentum = self.system.initial_v * self.system.masses
        print(f'Running {self.ntraj} surface hopping trajectories'
              + f' for momentum = {momentum}')
        for i in range(self.ntraj):

            self.system.reset_system(potential)
            traj = SingleTrajectoryHopper(dt)
            traj.run(system, steps, potential, data, **kwargs)

        data.outcome = data.outcome / self.ntraj

        print(f'{self.ntraj} successful trajectories completed.')


class SingleTrajectoryHopper(AbstractVelocityVerlet):
    """Single trajectory surface hopping.

    This class must be used in conjunction with a managing class, otherwise the
    outcome attribute of the data object has not been initilised.
    """

    def initialise_start(self):
        """Reset system and prepare logging variables."""
        self.r_t = [self.system.r]
        self.v_t = [self.system.v]

        self.system.reset_system(self.potential)
        self.outcome = np.zeros((self.system.nstates, 2))

    def integrate(self, steps):
        """Carry out main integration loop.

        Parameters
        ----------
        steps : int
            The number of steps.
        """

        self.current_acc = self.system.compute_acceleration(self.potential)
        for i in range(steps):

            self.propagate_system()

            self.system.propagate_density_matrix(self.dt)

            self.execute_hopping()

            if (i+1) % self.output_freq == 0:
                self.store_result()

            if self.is_finished():
                break

    def is_finished(self):
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

    def execute_hopping(self):
        """Carry out the hop between surfaces."""

        g = self.system.get_probabilities(self.dt)
        can_hop, desired_state = self.check_possible_hop(g)

        if can_hop:
            self.system.attempt_hop(desired_state)

    def check_possible_hop(self, g):
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
        for i, prob in enumerate(g):
            if prob > zeta:
                return True, i
        return False, -1

    def assign_data(self, data):
        """Add to the cumulative outcome."""
        data.outcome += self.outcome

    def store_result(self):
        """Store trajectory data but this is currently not used."""
        self.r_t.append(self.system.r)
        self.v_t.append(self.system.v)
